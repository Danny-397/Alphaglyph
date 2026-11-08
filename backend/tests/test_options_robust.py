"""
Is an options edge real? Deflated Sharpe on the leaderboard, the settings sweep
with PBO, the faster delta-strike search, and their endpoints. Fully offline.
"""

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402
import options_robust as orb       # noqa: E402


def _market(n=420, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2021-01-04', periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n))), index=idx)
    vix = pd.Series(19 + 5 * np.sin(np.arange(n) / 30), index=idx)
    return {'close': close, 'spy': close, 'vix': vix, 'rate': pd.Series(2.0, index=idx),
            'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx)}


def _window(m):
    idx = m['close'].index
    return idx[25].date().isoformat(), idx[-1].date().isoformat()


def test_delta_strike_search_matches_checking_every_strike():
    rng = np.random.default_rng(11)
    for _ in range(300):
        S = float(rng.uniform(20, 600))
        T = float(rng.uniform(3, 120)) / 365
        iv = float(rng.uniform(0.08, 0.9))
        r = float(rng.uniform(0, 0.06))
        want = float(rng.uniform(0.05, 0.6))
        kind = 'call' if rng.random() < 0.5 else 'put'
        skew = bool(rng.random() < 0.8)
        step = ob.strike_step(S, 'XYZ')
        grid = sorted({k for k in (ob.round_strike(S * (1 + p / 200.0), step) for p in range(-100, 101)) if k > 0})

        def miss(k):
            return abs(abs(ob.bs_delta(kind, S, k, T, r, ob.skewed_iv(iv, S, k, T, skew))) - want)
        fast = ob._delta_strike(grid, kind, want, S, T, r, iv, skew, None)
        assert miss(fast) == pytest.approx(miss(min(grid, key=miss)), abs=1e-12)


def test_deflation_raises_the_bar_with_the_number_tried():
    rng = np.random.default_rng(2)
    rets = [rng.normal(0.0004, 0.01, 750) for _ in range(19)]
    rows = [{} for _ in rets]
    d = orb.deflate(rows, rets)
    assert d['n'] == 19 and d['sharpe_bar'] > 0.5
    assert all(0 <= r['dsr'] <= 100 for r in rows)
    one = [{}]
    assert orb.deflate(one, rets[:1])['sharpe_bar'] == 0
    assert one[0]['dsr'] >= rows[0]['dsr']                 # one try is judged against zero


def test_sweep_runs_every_setting_and_tests_the_best():
    m = _market()
    p = ob.PRESETS['short_put']
    s = orb.sweep('SPY', *_window(m), p['legs'], p['name'], p['alloc'], m, current_dte=30,
                  dte=45, profit_take=10, initial_capital=100_000)       # sweep overrides dte and exits
    assert s['n'] == len(orb.SWEEP_DTE) * len(orb.SWEEP_EXITS) == len(s['cells'])
    assert {(c['dte'], c['exit']) for c in s['cells']} == {(d, e) for d in orb.SWEEP_DTE for e, _ in orb.SWEEP_EXITS}
    assert s['cells'][s['best']]['sharpe'] == max(c['sharpe'] for c in s['cells'])
    assert s['pbo'] is not None and 0 <= s['pbo']['pbo'] <= 1 and s['pbo']['n_combos'] == 252
    assert s['best_dsr'] is not None and s['sharpe_bar'] > 0
    assert s['verdict'].startswith(f"The best of the {s['n']} settings")


def test_exit_rules_reach_the_engine():
    m = _market()
    p = ob.PRESETS['iron_condor']
    s = orb.sweep('SPY', *_window(m), p['legs'], p['name'], p['alloc'], m)
    by = {(c['dte'], c['exit']): c for c in s['cells']}
    # closing halfway to expiry or at a 50% profit means more, shorter positions than holding
    assert by[(30, 'half')]['trades'] > by[(30, 'expiry')]['trades']
    assert by[(30, 'profit50')]['trades'] >= by[(30, 'expiry')]['trades']


def test_a_pass_comes_with_the_price_model_caveat():
    s = {'n': 20, 'best': 0, 'best_dsr': 97.0, 'sharpe_bar': 0.85, 'pbo': None,
         'cells': [{'dte': 7, 'exit_label': 'Hold to expiry', 'sharpe': 1.9}]}
    assert 'not proof of an edge' in orb.verdict(s)
    s['best_dsr'] = 40.0
    assert 'not proof' not in orb.verdict(s)


@pytest.fixture
def client(monkeypatch):
    import app as app_module
    m = _market()
    monkeypatch.setattr(ob, 'load_market', lambda t, s, e: m)
    monkeypatch.setattr(app_module.limiter, 'enabled', False)
    return app_module.app.test_client(), *_window(m)


def test_compare_endpoint_deflates_every_row(client):
    c, start, end = client
    body = c.post('/api/options/compare', json={'ticker': 'SPY', 'start_date': start, 'end_date': end}).get_json()
    assert body['deflation']['n'] == len(ob.PRESETS) and body['deflation']['sharpe_bar'] > 0
    assert all('dsr' in r for r in body['results'])


def test_sweep_endpoint(client):
    c, start, end = client
    r = c.post('/api/options/sweep', json={'ticker': 'SPY', 'preset': 'short_put', 'start_date': start,
                                           'end_date': end, 'dte': 14})
    body = r.get_json()
    assert r.status_code == 200 and body['current_dte'] == 14 and len(body['cells']) == 20


@pytest.mark.parametrize('payload', [
    {'preset': 'nope'},
    {'preset': 'short_put', 'start_date': '2018-01-04', 'end_date': '2024-01-04'},    # over 5 years
    {'preset': 'short_put', 'start_date': '2024-01-04', 'end_date': '2024-06-04'},    # under a year
])
def test_sweep_endpoint_rejects_bad_requests(client, payload):
    c, _, _ = client
    assert c.post('/api/options/sweep', json={'ticker': 'SPY', **payload}).status_code == 400


def test_results_carry_the_t_bill_rate_for_excess_returns():
    m = _market()
    r = ob.run_preset('short_put', 'SPY', *_window(m), market=m)
    assert r['model']['rf_daily'] == pytest.approx(0.02 / 252, rel=1e-6)
    ex = orb.excess_returns(r)
    assert len(ex) == len(r['equity']) - 1
    assert date.fromisoformat(r['equity'][0]['date'])
