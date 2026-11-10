"""
Earnings in the options backtester: report-day variance in option prices, the crush
after, report days left out of ordinary volatility, and the earnings entry and exit
rules. Fully offline: prices and report dates are synthetic.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402

N = 320
REPORT_EVERY = 63                   # a report about once a quarter


def _market(jump=0.06, seed=5, earnings=True):
    idx = pd.bdate_range('2021-01-04', periods=N)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.01, N)
    report_idx = list(range(40, N, REPORT_EVERY))
    for k, j in enumerate(report_idx):
        r[j] += jump if k % 2 == 0 else -jump
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    # Reports after the close the day before the move.
    ev = [{'date': idx[j - 1], 'when': 'amc'} for j in report_idx] if earnings else None
    return {'close': close, 'spy': close, 'vix': pd.Series(18.0, index=idx), 'rate': pd.Series(1.0, index=idx),
            'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx), 'raw_close': close,
            'dividends': pd.Series(dtype=float), 'splits': pd.Series(dtype=float), 'earnings': ev}, report_idx


def _run(m, preset='short_strangle', **kw):
    idx = m['close'].index
    p = ob.PRESETS[preset]
    return ob.run('XYZ', idx[30].date().isoformat(), idx[-1].date().isoformat(), p['legs'], name=p['name'],
                  alloc_pct=p['alloc'], market=m, **kw)


def test_reaction_day_follows_the_time_of_day():
    idx = pd.bdate_range('2024-01-01', periods=10)
    close = pd.Series(100.0, index=idx)
    close.iloc[5:] = 110.0                                   # the move happens on idx[5]
    assert ob.reaction_days(close, [{'date': idx[4], 'when': 'amc'}]) == [idx[5]]
    assert ob.reaction_days(close, [{'date': idx[5], 'when': 'bmo'}]) == [idx[5]]
    # Unknown time: whichever of the day and the next moved more.
    assert ob.reaction_days(close, [{'date': idx[4], 'when': None}]) == [idx[5]]
    assert ob.reaction_days(close, [{'date': idx[5], 'when': None}]) == [idx[5]]


def test_event_moves_only_look_back():
    m, rep = _market()
    close = m['close']
    react = [close.index[j] for j in rep]
    var = ob.event_moves(close, react, ob.realized_vol(close, exclude=react))
    rets = np.log(close).diff()
    # The second report's expected move is the first report's actual move.
    assert var[react[1]] == pytest.approx(float(rets[react[0]]) ** 2)
    # The first has no history: three ordinary days.
    assert 0 < var[react[0]] < 0.06 ** 2


def test_an_option_spanning_a_report_is_dearer_and_the_crush_follows():
    cv = ob.day_curve(0.25, None)
    before = ob.with_events(cv, ((5, 0.05 ** 2),))           # report in 5 days
    T = 30 / 365
    iv_before = ob.skewed_iv(0.25, 100, 100, T, True, before)
    assert iv_before == pytest.approx(math.sqrt(ob.term_iv(0.25, T, cv) ** 2 + 0.05 ** 2 / T), rel=1e-9)
    # An option expiring before the report doesn't carry it.
    assert ob.skewed_iv(0.25, 100, 100, 4 / 365, True, before) == ob.skewed_iv(0.25, 100, 100, 4 / 365, True, cv)
    # After the report: back to the ordinary level.
    assert ob.skewed_iv(0.25, 100, 100, T, True, cv) < iv_before - 0.05


def test_report_days_are_left_out_of_ordinary_volatility():
    m, rep = _market(jump=0.15)
    close = m['close']
    react = [close.index[j] for j in rep]
    with_jumps = ob.realized_vol(close)
    without = ob.realized_vol(close, exclude=react)
    after = close.index[rep[1] + 5]
    assert with_jumps[after] > 2 * without[after]


def test_premium_sellers_lose_less_through_reports_when_priced_for_them():
    m, _ = _market(jump=0.08)
    priced = _run(m)
    blind = _run({**m, 'earnings': None})
    assert 'earnings reports in the window' in priced['model']['earnings']
    assert 'No earnings dates were found' in blind['model']['earnings']
    held = [t for t in priced['trades'] if t['earnings']]
    assert held
    # Collecting the report premium: more credit on positions that hold through a report.
    by_open = {t['opened']: t for t in blind['trades']}
    pairs = [(t, by_open[t['opened']]) for t in held if t['opened'] in by_open]
    assert pairs
    assert all(-a['net_open'] > -b['net_open'] for a, b in pairs)


def test_avoid_and_only_rules():
    m, _ = _market()
    avoid = _run(m, entry={'earnings': 'avoid'})
    only = _run(m, entry={'earnings': 'only'})
    assert avoid['trades'] and all(t['earnings'] == 0 for t in avoid['trades'])
    assert only['trades'] and all(t['earnings'] >= 1 for t in only['trades'])
    assert 'would not hold through an earnings report' in avoid['strategy']['rules']


def test_close_the_day_before_earnings():
    m, rep = _market()
    r = _run(m, exit_earnings=True)
    idx = m['close'].index
    assert all(t['earnings'] == 0 for t in r['trades'])
    closes = [t for t in r['trades'] if t['reason'] == 'before earnings']
    assert closes
    report_eves = {idx[j - 1].date().isoformat() for j in rep}
    assert all(t['closed'] in report_eves for t in closes)
    # Nothing reopens on the eve itself.
    assert not any(t['opened'] in report_eves for t in r['trades'])
    assert r['strategy']['rules'].endswith('or closed the day before an earnings report.')


def test_index_funds_have_no_earnings():
    assert ob.earnings_events('SPY') == []
    m, _ = _market()
    idx = m['close'].index
    r = ob.run('SPY', idx[30].date().isoformat(), idx[-1].date().isoformat(), ob.PRESETS['short_put']['legs'],
               market={**m, 'earnings': []})
    assert r['model']['earnings'] == 'SPY is an index fund: no earnings reports.'


def test_clean_entry_accepts_the_earnings_rule():
    assert ob.clean_entry({'earnings': 'avoid'}) == {'earnings': 'avoid'}
    assert ob.clean_entry({'earnings': 'any'}) is None
    with pytest.raises(ValueError):
        ob.clean_entry({'earnings': 'sometimes'})


def test_the_saved_file_covers_popular_names_with_times():
    ev = ob.earnings_events('AAPL')
    assert ev and len(ev) >= 40
    assert ev[0]['date'] < pd.Timestamp('2010-01-01')
    assert sum(e['when'] == 'amc' for e in ev) > len(ev) / 2      # Apple reports after the close
    assert ev == sorted(ev, key=lambda e: e['date'])
