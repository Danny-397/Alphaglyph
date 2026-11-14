"""
The shared daily price store (features._download_daily) and how a refused
request is reported. Offline: the data sources are stubs.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import features as feat  # noqa: E402


def history(lo='2014-01-01', hi=None):
    idx = pd.bdate_range(lo, hi or pd.Timestamp.today().normalize())
    close = 100 + np.arange(len(idx)) * 0.1
    return pd.DataFrame({'Open': close, 'High': close, 'Low': close, 'Close': close, 'Volume': 1e6}, index=idx)


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    monkeypatch.setattr(feat, '_STORE', {})
    monkeypatch.setattr(feat, '_RATE_LIMITED', {})
    monkeypatch.setattr(feat, '_store_db', lambda: None)      # memory only in tests


def counting_source(monkeypatch, df):
    calls = []

    def source(ticker, period, interval, start, end):
        calls.append((ticker, start, end))
        return df
    monkeypatch.setattr(feat, '_download_sources', source)
    return calls


def test_one_fetch_serves_every_range_of_a_ticker(monkeypatch):
    calls = counting_source(monkeypatch, history())
    a = feat.fetch_ohlcv('SPY', start='2020-01-01', end='2021-01-01')
    b = feat.fetch_ohlcv('spy', start='2023-03-01', end='2024-06-30')
    c = feat.fetch_ohlcv('SPY', period='6mo')
    assert len(calls) == 1                                    # one request, three answers
    assert a.index.min() >= pd.Timestamp('2020-01-01') and a.index.max() <= pd.Timestamp('2021-01-01')
    assert b.index.min() >= pd.Timestamp('2023-03-01')
    assert c is not None and not c.empty


def test_a_range_older_than_the_stored_one_fetches_again(monkeypatch):
    calls = counting_source(monkeypatch, history('2000-01-03'))
    feat.fetch_ohlcv('SPY', start='2020-01-01', end='2021-01-01')
    feat.fetch_ohlcv('SPY', start='2005-01-03', end='2006-01-01')
    assert len(calls) == 2


def test_busy_sources_fall_back_to_the_saved_copy(monkeypatch):
    counting_source(monkeypatch, history())
    feat.fetch_ohlcv('SPY', start='2020-01-01', end='2021-01-01')
    stamp, lo, df = feat._STORE['SPY']
    feat._STORE['SPY'] = (stamp - 3 * 24 * 3600, lo, df)       # three days stale

    def busy(*a):
        raise RuntimeError('rate_limited')
    monkeypatch.setattr(feat, '_download_sources', busy)
    out = feat.fetch_ohlcv('SPY', start='2020-01-01', end='2021-01-01')
    assert out is not None and not out.empty
    assert not feat.was_rate_limited('SPY')


def test_busy_sources_with_nothing_saved_are_reported_as_busy(monkeypatch):
    def busy(*a):
        raise RuntimeError('rate_limited')
    monkeypatch.setattr(feat, '_download_sources', busy)
    assert feat.fetch_ohlcv('SPY', start='2020-01-01', end='2021-01-01') is None
    assert feat.was_rate_limited('spy')


def test_tiingo_out_of_quota_is_not_read_as_an_unknown_symbol(monkeypatch):
    """Yahoo's empty answer from a cloud IP must not turn a quota error into 'no such ticker'."""
    monkeypatch.setattr(feat, '_CACHE', {})
    monkeypatch.setattr(feat, '_try_tiingo', lambda *a: (None, 'rate_limited'))
    monkeypatch.setattr(feat, '_try_yfinance', lambda *a: (None, 'nodata'))
    monkeypatch.setattr(feat, '_try_stooq', lambda *a: None)
    with pytest.raises(RuntimeError):
        feat._download_sources('SPY', '6mo', '1d', '2020-01-01', '2021-01-01')


def test_tiingo_allocation_reply_counts_as_a_rate_limit(monkeypatch):
    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'detail': 'Error: You have run over your hourly request allocation.'}
    monkeypatch.setenv('TIINGO_API_KEY', 'x')
    monkeypatch.setattr(feat.requests, 'get', lambda *a, **k: Resp())
    assert feat._try_tiingo('SPY', '6mo', '2020-01-01', '2021-01-01') == (None, 'rate_limited')


def test_fresh_store_is_reused_and_stale_store_is_refreshed(monkeypatch):
    calls = counting_source(monkeypatch, history())
    feat.fetch_ohlcv('SPY', period='1y')
    feat.fetch_ohlcv('SPY', period='1y')
    assert len(calls) == 1
    stamp, lo, df = feat._STORE['SPY']
    feat._STORE['SPY'] = (time.time() - feat._STORE_FRESH_SECS - 1, lo, df)
    feat.fetch_ohlcv('SPY', period='1y')
    assert len(calls) == 2
