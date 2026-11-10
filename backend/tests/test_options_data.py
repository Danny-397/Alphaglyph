"""
Market data for the options backtester: the VIX term structure from Cboe's own
history files. Fully offline.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402

CSV = """DATE,OPEN,HIGH,LOW,CLOSE
01/02/2024,13.20,14.00,13.10,13.50
01/03/2024,13.60,14.40,13.50,14.00
01/04/2024,0,0,0,0
"""


def test_cboe_history_files_parse():
    s = ob.cboe_frame(CSV)
    assert list(s.index) == [pd.Timestamp('2024-01-02'), pd.Timestamp('2024-01-03')]   # zero prints dropped
    assert list(s) == [13.5, 14.0]
    assert ob.cboe_frame('a,b\n1,2\n') is None


def test_vix_curve_comes_from_cboe_without_the_slow_fallback(monkeypatch):
    idx = pd.bdate_range('2024-01-02', periods=5)
    levels = {'VIX': 15.0, 'VIX9D': 14.0, 'VIX3M': 16.5, 'VIX6M': 18.0}
    monkeypatch.setattr(ob, 'cboe_closes', lambda name: pd.Series(levels[name], index=idx))

    def boom(*a, **k):
        raise AssertionError('fell back to the price fetcher')
    monkeypatch.setattr(ob.feat, 'fetch_ohlcv', boom)
    c = ob.vix_curve('2024-01-02', '2024-01-08')
    assert c[9].iloc[0] == pytest.approx(14 / 15)
    assert c[93].iloc[0] == pytest.approx(16.5 / 15)
    assert c[183].iloc[0] == pytest.approx(18 / 15)
    assert len(c) == 5


def test_vix_curve_falls_back_when_cboe_is_down(monkeypatch):
    idx = pd.bdate_range('2024-01-02', periods=3)
    monkeypatch.setattr(ob, 'cboe_closes', lambda name: None)
    monkeypatch.setattr(ob.feat, 'fetch_ohlcv', lambda sym, **k: pd.DataFrame({'Close': [20.0, 20.0, 20.0]}, index=idx))
    c = ob.vix_curve('2024-01-02', '2024-01-04')
    assert (c[9] == 1.0).all()


def test_on_the_server_earnings_skip_yahoo(monkeypatch):
    monkeypatch.setenv('RENDER', 'true')
    ob._EARN_CACHE.pop('ZZZZ', None)
    import yfinance

    def boom(*a, **k):
        raise AssertionError('asked Yahoo from the server')
    monkeypatch.setattr(yfinance, 'Ticker', boom)
    assert ob.earnings_events('ZZZZ') is None          # not in the saved file or the snapshot
    assert ob.earnings_events('AAPL')                  # the saved file still answers
