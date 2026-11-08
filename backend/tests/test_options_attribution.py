"""
Greeks and the P&L split by Greek in the options backtester. Fully offline.
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob  # noqa: E402


@pytest.mark.parametrize('kind', ['call', 'put'])
def test_greeks_match_finite_differences(kind):
    S, K, T, r, s, h = 100.0, 95.0, 0.2, 0.04, 0.3, 1e-3
    d, g, th, vg = ob.bs_greeks(kind, S, K, T, r, s)
    p = lambda S=S, T=T, s=s: ob.bs_price(kind, S, K, T, r, s)   # noqa: E731
    assert d == pytest.approx((p(S=S + h) - p(S=S - h)) / (2 * h), abs=1e-6)
    assert g == pytest.approx((p(S=S + h) - 2 * p() + p(S=S - h)) / h ** 2, rel=1e-3)
    assert vg == pytest.approx((p(s=s + h) - p(s=s - h)) / (2 * h), rel=1e-5)
    assert th == pytest.approx(-(p(T=T + h) - p(T=T - h)) / (2 * h), rel=1e-5)


def test_expired_option_has_no_time_greeks():
    assert ob.bs_greeks('call', 110, 100, 0, 0.0, 0.2) == (1.0, 0.0, 0.0, 0.0)


def test_attribute_parts_add_up_and_pure_time_is_theta():
    # A flat curve and no skew, so a leg's IV can't roll along the curve as time passes.
    flat = ((9.0, 1.0), (30.0, 1.0), (93.0, 1.0), (183.0, 1.0))
    pos = ob.build_position(ob.PRESETS['short_put']['legs'], date(2024, 1, 2), 100.0, 0.2, 0.0, 30, 'XYZ',
                            False, flat)
    a = ob.position_marks(pos, date(2024, 1, 2), 100.0, 0.2, 0.0, False, flat)
    b = ob.position_marks(pos, date(2024, 1, 3), 100.0, 0.2, 0.0, False, flat)
    split = ob.attribute(a, b)
    assert sum(split.values()) == pytest.approx(b['value'] - a['value'], abs=1e-9)
    assert split['delta'] == 0 and split['gamma'] == 0 and abs(split['vega']) < 1e-9
    assert split['theta'] > 0                                  # a short put collects decay
    assert abs(split['other']) < 0.05 * abs(split['theta'])


def _market(prices, vix):
    idx = pd.bdate_range('2022-01-03', periods=len(prices))
    close = pd.Series(np.asarray(prices, float), index=idx)
    return {'close': close, 'spy': close, 'vix': pd.Series(np.asarray(vix, float), index=idx),
            'rate': pd.Series(2.0, index=idx),
            'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx)}


def _window(m):
    idx = m['close'].index
    return idx[25].date().isoformat(), idx[-1].date().isoformat()


@pytest.mark.parametrize('key', ['short_put', 'covered_call', 'iron_condor', 'long_straddle',
                                 'calendar_call', 'collar'])
def test_split_adds_up_to_every_trade_and_to_the_account(key):
    rng = np.random.default_rng(3)
    n = 400
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    vix = 18 + 6 * np.sin(np.arange(n) / 25) + rng.normal(0, 0.5, n)
    m = _market(prices, vix)
    r = ob.run_preset(key, 'SPY', *_window(m), market=m)
    assert r['trades']
    for t in r['trades']:
        assert sum(t['attribution'].values()) == pytest.approx(t['pnl'], abs=0.02)
    a = r['attribution']
    assert sum(a.values()) == pytest.approx(r['metrics']['final_value'] - 100_000, abs=0.05 * len(r['trades']) + 1)
    assert a['costs'] <= 0 and a['interest'] >= 0


def test_short_put_in_a_quiet_market_earns_from_time_decay():
    n = 300
    m = _market(np.full(n, 100.0), np.full(n, 20.0))
    a = ob.run_preset('short_put', 'SPY', *_window(m), market=m)['attribution']
    assert a['theta'] > 0 and a['theta'] == max(a[k] for k in ('delta', 'gamma', 'theta', 'vega'))


def test_long_straddle_gains_from_rising_volatility():
    n = 300
    m = _market(np.full(n, 100.0), np.linspace(12, 45, n))
    a = ob.run_preset('long_straddle', 'SPY', *_window(m), market=m)['attribution']
    assert a['vega'] > 0 and a['theta'] < 0


def test_equity_rows_carry_exposures_and_are_flat_when_out():
    n = 200
    m = _market(np.linspace(100, 110, n), np.full(n, 20.0))
    r = ob.run_preset('covered_call', 'SPY', *_window(m), market=m)
    held = [e for e in r['equity'] if e['delta'] != 0]
    assert held and all(e['delta'] > 0 for e in held)          # long stock, short call: net long
    assert all(e['theta'] >= 0 for e in held)                  # the short call decays in its favour
    assert any(e['theta'] > 0 for e in held)
    first = r['equity'][0]
    assert {'delta', 'theta', 'vega'} <= set(first)
