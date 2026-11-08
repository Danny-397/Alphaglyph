"""
Dividends and early assignment in the options backtester, and the start-date
check. Fully offline: prices and dividends are synthetic.
"""

import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402
import options_robust as orb       # noqa: E402


def _market(raw, divs, start='2022-01-03', vix=18.0):
    """A market with unadjusted closes `raw` and dividends {trading-day index: amount}."""
    idx = pd.bdate_range(start, periods=len(raw))
    raw = pd.Series(np.asarray(raw, float), index=idx)
    dv = pd.Series({idx[i]: a for i, a in divs.items()}, dtype=float)
    # Total-return (adjusted) closes: each dividend scales every earlier price down.
    adj = raw.copy()
    for d, a in dv.items():
        pos = idx.get_loc(d)
        adj.iloc[:pos] *= 1 - a / raw.iloc[pos - 1]
    return {'close': adj, 'spy': adj, 'vix': pd.Series(vix, index=idx), 'rate': pd.Series(1.0, index=idx),
            'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx), 'raw_close': raw, 'dividends': dv}


def _window(m, skip=25):
    idx = m['close'].index
    return idx[skip].date().isoformat(), idx[-1].date().isoformat()


def test_merton_price_is_black_scholes_on_the_discounted_spot():
    S, K, T, r, q, s = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
    d1 = (math.log(S / K) + (r - q + s * s / 2) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    N = ob._ncdf
    want = S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    assert ob.bs_price('call', S * math.exp(-q * T), K, T, r, s) == pytest.approx(want, abs=1e-9)
    assert want == pytest.approx(9.2270, abs=1e-3)                          # the textbook value


def test_a_yield_makes_calls_cheaper_and_puts_dearer():
    leg_c = {'type': 'call', 'side': 'long', 'qty': 1, 'strike': 100.0, 'expiry': date(2024, 3, 1)}
    leg_p = {**leg_c, 'type': 'put'}
    d = date(2024, 1, 2)
    assert ob.leg_value(leg_c, d, 100, 0.2, 0.03, True, None, 0.02) < ob.leg_value(leg_c, d, 100, 0.2, 0.03, True)
    assert ob.leg_value(leg_p, d, 100, 0.2, 0.03, True, None, 0.02) > ob.leg_value(leg_p, d, 100, 0.2, 0.03, True)


def test_stock_legs_are_paid_dividends_and_the_books_still_balance():
    n = 300
    raw = np.linspace(100, 112, n)
    m = _market(raw, {80: 1.0, 143: 1.0, 206: 1.0, 269: 1.0})
    r = ob.run('XYZ', *_window(m), ob.PRESETS['collar']['legs'], market=m, alloc_pct=100)
    a = r['attribution']
    assert a['dividends'] > 0
    assert sum(t['dividends'] for t in r['trades']) == pytest.approx(a['dividends'], abs=0.05)
    assert sum(a.values()) == pytest.approx(r['metrics']['final_value'] - 100_000, abs=1.0)
    for t in r['trades']:
        assert sum(t['attribution'].values()) == pytest.approx(t['pnl'], abs=0.05)
    assert 'paid $' in r['model']['dividends']


def test_option_only_strategies_get_no_dividends():
    m = _market(np.linspace(100, 110, 300), {80: 1.0, 143: 1.0})
    r = ob.run('XYZ', *_window(m), ob.PRESETS['iron_condor']['legs'], market=m)
    assert r['attribution']['dividends'] == 0


def test_deep_in_the_money_short_call_is_assigned_before_the_ex_date():
    raw = np.r_[np.full(30, 100.0), np.linspace(100, 125, 20), np.full(70, 125.0)]   # the call goes deep ITM
    m = _market(raw, {55: 3.0})                                    # a large dividend, before the first expiry
    legs = [{'type': 'stock', 'side': 'long', 'qty': 1},
            {'type': 'call', 'side': 'short', 'qty': 1, 'strike': {'mode': 'pct', 'value': 2}}]
    r = ob.run('XYZ', *_window(m), legs, market=m, alloc_pct=100, dte=60)
    assigned = [t for t in r['trades'] if t['reason'] == 'assigned early (dividend)']
    assert assigned and r['metrics']['early_assignments'] >= 1
    t = assigned[0]
    assert t['closed'] == m['close'].index[54].date().isoformat()                  # the day before the ex-date
    call = next(leg for leg in t['legs'] if leg['type'] == 'call')
    assert call['close'] == pytest.approx(t['underlying_close'] - call['strike'], abs=0.01)   # at intrinsic
    assert t['dividends'] == 0                                                      # gone before the ex-date


def test_early_assignment_rule():
    leg = {'type': 'call', 'side': 'short', 'qty': 1, 'strike': 100.0, 'expiry': date(2024, 2, 16)}
    d = date(2024, 2, 1)
    assert ob.early_assignment([leg], d, 130, 0.2, 0.03, False, None, 0.0, 1.0)     # deep ITM, little time value
    assert not ob.early_assignment([leg], d, 101, 0.2, 0.03, False, None, 0.0, 1.0)  # near the money
    assert not ob.early_assignment([leg], d, 90, 0.2, 0.03, False, None, 0.0, 5.0)   # out of the money
    assert not ob.early_assignment([{**leg, 'side': 'long'}], d, 130, 0.2, 0.03, False, None, 0.0, 1.0)


def test_without_dividend_data_the_run_says_so():
    m = _market(np.linspace(100, 110, 200), {})
    m = {**m, 'raw_close': None, 'dividends': None}
    r = ob.run('XYZ', *_window(m), ob.PRESETS['covered_call']['legs'], market=m)
    assert r['model']['dividends'].startswith('Dividend data was unavailable')


def test_benchmark_stays_total_return():
    raw = np.full(300, 100.0)
    m = _market(raw, {100: 2.0, 200: 2.0})
    r = ob.run('XYZ', *_window(m), ob.PRESETS['covered_call']['legs'], market=m)
    assert r['benchmark']['total_return'] > 3.5                                      # dividends reinvested


def test_trend_rule_compares_like_with_like():
    raw = np.full(520, 100.0)
    m = _market(raw, {i: 1.0 for i in range(60, 520, 63)})       # flat price, so never above its own average
    r = ob.run('XYZ', *_window(m, 230), ob.PRESETS['short_put']['legs'], market=m, entry={'trend': 'above'})
    assert not r['trades']                                        # adjusted closes would have made it look rising


def test_start_dates_measure_roll_date_luck():
    rng = np.random.default_rng(9)
    raw = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.013, 420)))
    m = _market(raw, {})
    start, end = _window(m)
    p = ob.PRESETS['iron_condor']
    sd = orb.start_dates({'ticker': 'XYZ', 'start': start, 'end': end, 'legs': p['legs'], 'name': p['name'],
                          'alloc_pct': p['alloc'], 'market': m})
    assert len(sd['runs']) == len(orb.START_SHIFTS)
    assert len({x['start'] for x in sd['runs']}) == len(sd['runs'])
    assert sd['cagr_range'] == pytest.approx(sd['cagr_high'] - sd['cagr_low'], abs=0.011)
    s = {'n': 20, 'best': 0, 'best_dsr': 10.0, 'sharpe_bar': 0.85, 'pbo': None, 'start_dates': sd,
         'cells': [{'dte': 30, 'exit_label': 'Hold to expiry', 'sharpe': 0.3}]}
    assert 'different days a week apart' in orb.verdict(s)
