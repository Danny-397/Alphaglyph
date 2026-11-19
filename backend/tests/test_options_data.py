"""
Market data for the options backtester: the VIX term structure from Cboe's own
history files. Fully offline.
"""

import os
import sys

import numpy as np
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


def test_pricing_context_prices_exactly_as_the_backtest_does():
    """The quotes check prices with pricing_context; it must give the backtest's own entry prices."""
    import numpy as np
    from datetime import date
    idx = pd.bdate_range('2023-01-02', periods=320)
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0003, 0.012, 320))), index=idx)
    dv = pd.Series({idx[100]: 0.5, idx[163]: 0.5})
    m = {'close': close, 'spy': close * 1.01, 'vix': pd.Series(18.0, index=idx), 'rate': pd.Series(4.0, index=idx),
         'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx), 'raw_close': close,
         'dividends': dv, 'splits': pd.Series(dtype=float),
         'earnings': [{'date': idx[150], 'when': 'amc'}, {'date': idx[213], 'when': 'amc'}]}
    r = ob.run_preset('iron_condor', 'XYZ', idx[60].date().isoformat(), idx[-1].date().isoformat(), market=m)
    ctx = ob.pricing_context(m, 'XYZ')
    checked = 0
    for t in r['trades'][:6]:
        c = ob.context_at(ctx, t['opened'])
        for leg in t['legs']:
            T = (date.fromisoformat(leg['expiry']) - date.fromisoformat(t['opened'])).days / 365.0
            price, _ = ob.model_price(leg['type'], c, leg['strike'], T)
            assert price == pytest.approx(leg['open'], abs=0.006)
            checked += 1
    assert checked >= 20


def test_implied_vol_inverts_black_scholes():
    p = ob.bs_price('put', 100 * np.exp(-0.01 * 0.25), 95, 0.25, 0.04, 0.27)
    assert ob.implied_vol('put', p, 100, 95, 0.25, 0.04, 0.01) == pytest.approx(0.27, abs=1e-6)
    assert ob.implied_vol('call', 0.0, 100, 95, 0.25, 0.04) is None


def _market_with_reports(events):
    idx = pd.bdate_range('2023-01-02', periods=320)
    rets = np.random.default_rng(2).normal(0.0003, 0.012, 320)
    rets[151], rets[214] = 0.08, -0.08                  # two past reports, each an 8% reaction
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
    return idx, {'close': close, 'spy': close * 1.01, 'vix': pd.Series(18.0, index=idx),
                 'rate': pd.Series(4.0, index=idx), 'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx),
                 'raw_close': close, 'dividends': pd.Series(dtype=float), 'splits': pd.Series(dtype=float),
                 'earnings': events}


def test_pricing_context_sees_a_report_after_the_last_close():
    """A report whose reaction day has no price yet still counts: an option priced before it
    and expiring after it carries the report's move (it used to be left out entirely)."""
    idx, _ = _market_with_reports([])
    last = idx[-1]
    past = [{'date': idx[150], 'when': 'amc'}, {'date': idx[213], 'when': 'amc'}]
    _, m = _market_with_reports(past + [{'date': last, 'when': 'amc'}])
    _, m_without = _market_with_reports(past)
    c = ob.context_at(ob.pricing_context(m, 'XYZ'), last)
    c0 = ob.context_at(ob.pricing_context(m_without, 'XYZ'), last)
    assert c['next_report'] == (last + pd.offsets.BDay(1)).date()     # after the close: the next business day
    assert c0['next_report'] is None
    _, iv_with = ob.model_price('call', c, c['S'], 10 / 365)
    _, iv_without = ob.model_price('call', c0, c['S'], 10 / 365)
    assert iv_with > iv_without * 1.5                                 # an 8% move in 10 days is priced in
    _, short = ob.model_price('call', c, c['S'], 0.5 / 365)          # expires before the reaction day
    _, short0 = ob.model_price('call', c0, c['S'], 0.5 / 365)
    assert short == pytest.approx(short0)


def test_upcoming_reaction_days_follow_the_report_time():
    idx = pd.bdate_range('2026-11-02', periods=10)        # ends Friday 2026-11-13
    close = pd.Series(100.0, index=idx)
    ev = [{'date': pd.Timestamp('2026-11-13'), 'when': 'amc'},    # Friday after the close -> Monday
          {'date': pd.Timestamp('2026-11-18'), 'when': 'bmo'},    # before the open -> that day
          {'date': pd.Timestamp('2026-11-10'), 'when': 'amc'}]    # already in the prices
    assert ob.upcoming_reactions(close, ev) == [pd.Timestamp('2026-11-16'), pd.Timestamp('2026-11-18')]


def test_calibrate_scales_volatility_only_when_asked(monkeypatch):
    idx, m = _market_with_reports([])
    monkeypatch.setattr(ob, 'iv_calibration', lambda t: {'factor': 1.3, 'contracts': 300, 'days': ['a', 'b']})
    kw = dict(market=m)
    start, end = idx[60].date().isoformat(), idx[-1].date().isoformat()
    plain = ob.run_preset('short_put', 'XYZ', start, end, **kw)
    cal = ob.run_preset('short_put', 'XYZ', start, end, calibrate=True, **kw)
    assert 'calibration' not in plain['model'] and cal['model']['calibration']['factor'] == 1.3
    assert cal['model']['iv_scale'] == pytest.approx(1.3)
    # Richer volatility: the first short put collects more premium.
    assert cal['trades'][0]['legs'][0]['open'] > plain['trades'][0]['legs'][0]['open']


def test_a_calibrated_run_reports_the_default_model_alongside(monkeypatch):
    import options_robust as rob
    idx, m = _market_with_reports([])
    monkeypatch.setattr(ob, 'iv_calibration', lambda t: {'factor': 1.3, 'contracts': 300, 'days': ['a', 'b']})
    kw = dict(ticker='XYZ', start=idx[60].date().isoformat(), end=idx[-1].date().isoformat(),
              legs=ob.PRESETS['short_put']['legs'], market=m, calibrate=True)
    base = ob.run(**kw)
    sens = rob.model_sensitivity(kw, base)
    labels = [v['label'] for v in sens['variants']]
    assert labels[0] == 'Without the quote match'
    plain = ob.run(**{**kw, 'calibrate': False})
    assert sens['variants'][0]['cagr'] == plain['metrics']['cagr']
    plain_sens = rob.model_sensitivity({**kw, 'calibrate': False}, plain)
    assert 'Without the quote match' not in [v['label'] for v in plain_sens['variants']]
