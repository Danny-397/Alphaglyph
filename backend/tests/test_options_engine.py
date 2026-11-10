"""
The options engine beyond one position at a time: the wheel (puts, assignment,
covered calls, called away), staggered positions, and closing (rolling) on a
short option's delta. Fully offline: prices are synthetic.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402


def _market(path):
    idx = pd.bdate_range('2022-01-03', periods=len(path))
    close = pd.Series(np.asarray(path, float), index=idx)
    return {'close': close, 'spy': close, 'vix': pd.Series(20.0, index=idx), 'rate': pd.Series(2.0, index=idx),
            'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx), 'raw_close': close,
            'dividends': pd.Series(dtype=float), 'splits': pd.Series(dtype=float), 'earnings': None}


def _window(m, skip=30):
    idx = m['close'].index
    return idx[skip].date().isoformat(), idx[-1].date().isoformat()


def _down_then_up(n=400):
    """Falls 20% over the first months (puts assigned), then recovers past the start (shares called away)."""
    a, b = n // 3, n // 6
    x = np.concatenate([np.linspace(100, 80, a), np.full(b, 80.0), np.linspace(80, 115, n - a - b)])
    noise = np.random.default_rng(7).normal(0, 0.004, n)
    return x * np.exp(noise)


def test_the_wheel_buys_shares_on_assignment_and_sells_calls_until_called_away():
    m = _market(_down_then_up())
    r = ob.run_preset('wheel', 'XYZ', *_window(m), market=m)
    w = r['metrics']['wheel']
    assert w['assigned'] >= 1 and w['called_away'] >= 1
    reasons = [t['reason'] for t in r['trades']]
    k = reasons.index('expiry: assigned, bought the shares')
    nxt = r['trades'][k + 1]
    assert {leg['type'] for leg in nxt['legs']} == {'stock', 'call'}
    assert nxt['opened'] == r['trades'][k]['closed']                 # calls start the day of assignment
    away = reasons.index('expiry: shares called away', k)
    assert all(leg['type'] == 'put' for leg in r['trades'][away + 1]['legs'])   # then puts again
    # Same number of shares as the put's contracts.
    put_qty = next(leg['qty'] for leg in r['trades'][k]['legs'])
    assert next(leg['qty'] for leg in nxt['legs'] if leg['type'] == 'stock') == put_qty
    assert 'is sold on them every 30 days' in r['strategy']['rules']
    assert r['strategy']['headline'].startswith('The wheel: a cycle backtest')


def test_wheel_books_balance():
    m = _market(_down_then_up())
    r = ob.run_preset('wheel', 'XYZ', *_window(m), market=m)
    eq = r['equity']
    pnl = sum(t['pnl'] for t in r['trades'])
    interest = r['attribution']['interest']
    assert eq[-1]['value'] - 100_000 == pytest.approx(pnl + interest, abs=1.0)


def test_carried_shares_pay_no_slippage_between_calls():
    m = _market(_down_then_up())
    r = ob.run_preset('wheel', 'XYZ', *_window(m), market=m, commission=0.0, spread_pct=0.0)
    calls = [t for t in r['trades'] if any(leg['type'] == 'stock' for leg in t['legs'])]
    assert calls
    # With no option costs, the only cost on a covered-call trade would be stock slippage.
    middle = [t for t in calls if t['reason'] == 'expiry']
    assert all(t['costs'] == 0 for t in middle)


def test_the_wheel_refuses_other_legs_and_staggering():
    m = _market(_down_then_up())
    bad = ob.run('XYZ', *_window(m), ob.PRESETS['short_strangle']['legs'], market=m,
                 wheel=ob.PRESETS['wheel']['wheel'])
    assert bad == {'error': 'The wheel starts from a single short put'}
    assert 'one position at a time' in ob.run_preset('wheel', 'XYZ', *_window(m), market=m, stagger_days=7)['error']


def test_staggered_positions_overlap_and_never_tie_up_more_than_the_account():
    m = _market(np.linspace(100, 120, 300))
    r = ob.run_preset('bull_put_spread', 'XYZ', *_window(m), market=m, alloc_pct=20, stagger_days=7, dte=45)
    assert 'Staggered: up to' in ' '.join(r['notes'])
    events = sorted([(t['opened'], 1) for t in r['trades']] + [(t['closed'], -1) for t in r['trades']])
    live = peak = 0
    for _, x in events:
        live += x
        peak = max(peak, live)
    assert peak >= 4
    one = ob.run_preset('bull_put_spread', 'XYZ', *_window(m), market=m, alloc_pct=20, dte=45)
    assert len(r['trades']) > 3 * len(one['trades'])
    assert 'every 7 days, alongside any still open, 45 days' in r['strategy']['rules']


def test_staggering_respects_free_capital():
    """Cash set aside for open cash-secured puts never exceeds the account."""
    m = _market(np.linspace(100, 101, 200))
    r = ob.run_preset('short_put', 'XYZ', *_window(m), market=m, alloc_pct=60, stagger_days=3)
    days = [e['date'] for e in r['equity']]
    peak_open = 0
    for d in days:
        live = [t for t in r['trades'] if t['opened'] <= d < t['closed']]
        peak_open = max(peak_open, len(live))
        secured = sum(leg['strike'] * leg['qty'] * 100 for t in live for leg in t['legs'])
        value = next(e['value'] for e in r['equity'] if e['date'] == d)
        assert secured <= value * 1.03
    assert peak_open == 2                                             # 60% + what is left


def test_delta_exit_closes_and_rolls_the_same_day():
    x = np.concatenate([np.full(60, 100.0), np.linspace(100, 125, 60), np.full(80, 125.0)])
    m = _market(x)
    r = ob.run_preset('short_strangle', 'XYZ', *_window(m), market=m, exit_delta=0.45)
    rolled = [t for t in r['trades'] if t['reason'] == 'short delta reached 0.45']
    assert rolled
    nxt = {t['opened'] for t in r['trades']}
    assert all(t['closed'] in nxt for t in rolled)                  # a new position opens the same day
    assert "delta reaches 0.45" in r['strategy']['rules']


def test_short_delta_in_the_marks():
    from datetime import date
    pos = [{'type': 'put', 'side': 'short', 'qty': 1, 'strike': 100.0, 'expiry': date(2024, 2, 16), 'price': 2.0},
           {'type': 'put', 'side': 'long', 'qty': 1, 'strike': 90.0, 'expiry': date(2024, 2, 16), 'price': 0.5}]
    mk = ob.position_marks(pos, date(2024, 1, 16), 100.0, 0.2, 0.03, False)
    assert 0.4 < mk['short_delta'] < 0.6                              # at the money


def test_staggering_wording_with_an_entry_rule():
    m = _market(np.linspace(100, 120, 300))
    r = ob.run_preset('bull_put_spread', 'XYZ', *_window(m), market=m, alloc_pct=20, stagger_days=7,
                      entry={'vix_min': 10})
    assert 'every 7 days, alongside any still open' in r['strategy']['rules']
    assert 'the last one closes' not in r['strategy']['rules']
