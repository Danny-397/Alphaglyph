"""
The real VIX term structure in the options pricer, and the checks against
Cboe's PutWrite index. Fully offline: every series is synthetic.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402
import options_validation as ov    # noqa: E402

CURVE = ((9.0, 0.9), (30.0, 1.0), (93.0, 1.1), (183.0, 1.2))


def test_curve_reproduces_its_points_and_holds_the_ends():
    for days, ratio in CURVE:
        assert ob.term_iv(0.2, days / 365, CURVE) == pytest.approx(0.2 * ratio)
    assert ob.term_iv(0.2, 3 / 365, CURVE) == pytest.approx(0.2 * 0.9)     # before the first point
    assert ob.term_iv(0.2, 400 / 365, CURVE) == pytest.approx(0.2 * 1.2)   # after the last


def test_curve_interpolates_total_variance_linearly():
    d = 60.0
    w30, w93 = 1.0 ** 2 * 30, 1.1 ** 2 * 93
    want = 0.2 * math.sqrt((w30 + (w93 - w30) * (d - 30) / 63) / d)
    assert ob.term_iv(0.2, d / 365, CURVE) == pytest.approx(want)
    # and monotone between points when the curve slopes up
    vals = [ob.term_iv(0.2, x / 365, CURVE) for x in range(30, 184, 7)]
    assert vals == sorted(vals)


def test_missing_curve_points_fall_back_to_the_modelled_slope():
    cv = ob.day_curve(0.15, (float('nan'), 1.1, float('nan')))
    pts = dict(cv)
    assert pts[30.0] == 1.0 and pts[93.0] == 1.1
    assert pts[9.0] == pytest.approx(ob.model_term_ratio(0.15, 9))
    assert pts[183.0] == pytest.approx(ob.model_term_ratio(0.15, 183))
    assert dict(ob.day_curve(0.15, None))[93.0] == pytest.approx(ob.model_term_ratio(0.15, 93))
    # no curve at all: the old modelled term structure, unchanged
    assert ob.term_iv(0.15, 90 / 365) == pytest.approx(0.15 * ob.model_term_ratio(0.15, 90))


def _market(n=160, curve=True):
    idx = pd.bdate_range('2022-01-03', periods=n)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.1, n)), index=idx)
    m = {'close': close, 'spy': close, 'vix': pd.Series(20.0, index=idx),
         'rate': pd.Series(0.0, index=idx)}
    if curve:
        m['curve'] = pd.DataFrame({9: 0.9, 93: 1.1, 183: 1.2}, index=idx)
    return m


def test_backtest_says_which_term_structure_it_used():
    legs = ob.PRESETS['calendar_call']['legs']
    m = _market()
    start, end = m['close'].index[25].date().isoformat(), m['close'].index[-1].date().isoformat()
    real = ob.run('SPY', start, end, legs, market=m, alloc_pct=10)
    flat = ob.run('SPY', start, end, legs, market={**m, 'curve': None}, alloc_pct=10)
    assert real['model']['term'].startswith('The real VIX term structure each day')
    assert flat['model']['term'].startswith('A modelled slope')
    # a calendar depends on the curve, so the two must differ
    assert real['metrics']['final_value'] != flat['metrics']['final_value']


def test_vix_curve_is_built_from_ratios_and_drops_bad_prints(monkeypatch):
    idx = pd.bdate_range('2024-01-02', periods=5)
    data = {'^VIX': [20, 20, 20, 20, 20], '^VIX9D': [18, 18, 18, 18, 90],
            '^VIX3M': [22, 22, 22, 22, 22], '^VIX6M': [24, 24, 24, 24, 24]}
    monkeypatch.setattr(ob.feat, 'fetch_ohlcv',
                        lambda sym, start=None, end=None: pd.DataFrame({'Close': data[sym]}, index=idx))
    cv = ob.vix_curve('2024-01-01', '2024-01-10')
    assert cv[93].tolist() == pytest.approx([1.1] * 5)
    assert cv[9].iloc[0] == pytest.approx(0.9)
    assert np.isnan(cv[9].iloc[-1])                  # 4.5x the VIX is a bad print


def test_compare_identical_series_is_a_perfect_match():
    idx = pd.bdate_range('2020-01-01', periods=600)
    rng = np.random.default_rng(0)
    s = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx))), index=idx)
    c = ov.compare(s, s)
    assert c['cagr_gap'] == 0 and c['monthly_corr'] == 1.0 and c['tracking_error'] == 0
    assert c['beta'] == pytest.approx(1.0)
    assert c['model'] == c['real']


def test_compare_sees_a_model_that_earns_more():
    idx = pd.bdate_range('2020-01-01', periods=600)
    rng = np.random.default_rng(1)
    real = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(idx))), index=idx)
    model = real * np.exp(np.arange(len(idx)) * 0.0002)          # about +5% a year on top
    c = ov.compare(model, real)
    assert 4 < c['cagr_gap'] < 7
    assert c['monthly_corr'] > 0.95


def test_replica_restores_the_level_it_borrows():
    keep = ob.VIX_TO_ATM
    m = _market()
    start, end = m['close'].index[25].date().isoformat(), m['close'].index[-1].date().isoformat()
    lo = ov.replica(start, end, m, level=0.75)
    hi = ov.replica(start, end, m, level=0.95)
    assert ob.VIX_TO_ATM == keep
    assert hi.iloc[-1] > lo.iloc[-1]          # richer puts, more premium in a rising market


def test_notional_sizing_holds_one_unit_per_100_shares_of_value():
    m = _market()
    start, end = m['close'].index[25].date().isoformat(), m['close'].index[-1].date().isoformat()
    legs = ob.PRESETS['iron_condor']['legs']
    risk = ob.run('SPY', start, end, legs, market=m, alloc_pct=100, initial_capital=1_000_000)
    notional = ob.run('SPY', start, end, legs, market=m, alloc_pct=100, initial_capital=1_000_000, sizing='notional')
    S0 = float(m['close'].iloc[25])
    assert notional['trades'][0]['units'] == int(1_000_000 // (S0 * 100))
    assert risk['trades'][0]['units'] > notional['trades'][0]['units']     # worst-case sizing levers a condor up


def test_every_index_has_a_replica_that_validates():
    for code, spec in ov.INDICES.items():
        legs = ob.clean_legs(spec['legs'])
        assert legs and spec['name'] and spec['what']
        assert any(leg['type'] != 'stock' for leg in legs), code


def test_fit_skew_finds_the_settings_that_generated_the_index(monkeypatch):
    m = _market(n=300)
    monkeypatch.setattr(ov, 'FIT', (m['close'].index[25].date().isoformat(), m['close'].index[-1].date().isoformat()))
    monkeypatch.setattr(ov, 'SKEW_GRID_PUT', [0.1, 0.2, 0.3])
    monkeypatch.setattr(ov, 'SKEW_GRID_CALL', [0.1, 0.25, 0.4])
    # "Real" indices made by the model itself with known skews: the fit has to recover them.
    truth = (0.3, 0.1)
    real = {code: ov.replica(*ov.FIT, m, legs=ov.INDICES[code]['legs'], skew=truth) for code in ('PPUT', 'BXMD')}
    keep = (ob.SKEW_PUT, ob.SKEW_CALL)
    monkeypatch.setattr(ob, 'SKEW_PUT', truth[0])
    monkeypatch.setattr(ob, 'SKEW_CALL', truth[1])
    assert ov.fit_skew(m, real) == {'put': 0.3, 'call': 0.1}
    monkeypatch.setattr(ob, 'SKEW_PUT', keep[0])
    monkeypatch.setattr(ob, 'SKEW_CALL', keep[1])


def test_index_checks_compare_each_available_index():
    m = _market(n=300)
    idx = m['close'].index
    start, end = idx[25].date().isoformat(), idx[-1].date().isoformat()
    real = ov.replica(start, end, m, legs=ov.INDICES['BXM']['legs'])
    rows = ov.index_checks(m, {'BXM': real}, start, end, before=(0.40, 0.15))
    assert [r['code'] for r in rows] == ['BXM']
    assert rows[0]['cagr_gap'] == 0 and rows[0]['monthly_corr'] == 1.0
    assert rows[0]['cagr_gap_before_skew_fit'] is not None
    assert len(rows[0]['chart']['model']) == len(rows[0]['chart']['real'])


def test_third_fridays_and_the_monthly_schedule():
    from datetime import date
    assert ob.third_friday(2020, 3) == date(2020, 3, 20)
    assert ob.third_friday(2026, 11) == date(2026, 11, 20)
    assert ob.monthly_expiry_for(date(2020, 2, 21), 30) == date(2020, 3, 20)      # roll on expiry day
    assert ob.monthly_expiry_for(date(2020, 3, 20), 30) == date(2020, 4, 17)
    assert ob.monthly_expiry_for(date(2026, 12, 18), 30) == date(2027, 1, 15)     # across a year end
    for d in pd.date_range('2024-01-02', '2024-12-31', freq='7D'):
        e = ob.monthly_expiry_for(d.date(), 30)
        assert e > d.date() and e.weekday() == 4 and 15 <= e.day <= 21


def test_monthly_expiries_roll_on_third_fridays():
    m = _market(n=200)
    start, end = m['close'].index[25].date().isoformat(), m['close'].index[-1].date().isoformat()
    r = ob.run('SPY', start, end, ob.PRESETS['short_put']['legs'], market=m, expiries='monthly')
    from datetime import date
    for t in r['trades'][:-1]:
        e = date.fromisoformat(t['legs'][0]['expiry'])
        assert e.weekday() == 4 and 15 <= e.day <= 21
    assert 'monthly (third-Friday) expiry' in r['strategy']['rules']
    w = ob.run('SPY', start, end, ob.PRESETS['short_put']['legs'], market=m)
    assert 'expiries' not in w['strategy']
