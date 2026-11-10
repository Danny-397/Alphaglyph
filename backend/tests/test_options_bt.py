"""
Options strategy backtester: pricing, strike and expiry rules, the descriptions
shown to the user, validation, capital rules, honest P&L accounting and the API.

Fully offline: market data is synthetic and passed in, or monkeypatched.
"""

import math
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob  # noqa: E402


def market(prices, vix=20.0, rate=0.0, start='2022-01-03'):
    idx = pd.bdate_range(start, periods=len(prices))
    close = pd.Series(np.asarray(prices, dtype=float), index=idx)
    spy = close.copy()
    return {'close': close, 'spy': spy,
            'vix': pd.Series(vix, index=idx, dtype=float),
            'rate': pd.Series(rate * 100, index=idx, dtype=float)}


def window(m):
    idx = m['close'].index
    return idx[25].strftime('%Y-%m-%d'), idx[-1].strftime('%Y-%m-%d')


# ── pricing ─────────────────────────────────────────────────

def test_black_scholes_textbook_value_and_parity():
    c = ob.bs_price('call', 100, 100, 1.0, 0.05, 0.2)
    p = ob.bs_price('put', 100, 100, 1.0, 0.05, 0.2)
    assert c == pytest.approx(10.4506, abs=1e-3)
    assert c - p == pytest.approx(100 - 100 * math.exp(-0.05), abs=1e-9)


def test_expired_options_are_worth_intrinsic():
    assert ob.bs_price('call', 110, 100, 0, 0.05, 0.3) == 10
    assert ob.bs_price('put', 110, 100, 0, 0.05, 0.3) == 0
    assert ob.bs_delta('call', 110, 100, 0, 0.0, 0.3) == 1.0


def test_skew_makes_puts_richer_and_calls_cheaper():
    T = 30 / 365
    atm = ob.skewed_iv(0.20, 100, 100, T)
    assert atm == pytest.approx(0.20, abs=1e-9)
    assert ob.skewed_iv(0.20, 100, 92, T) > 0.20 * 1.2         # OTM put
    assert 0.20 * 0.75 <= ob.skewed_iv(0.20, 100, 108, T) < 0.20  # OTM call cheaper, floored at 0.75x
    assert ob.skewed_iv(0.20, 100, 92, T, skew=False) == pytest.approx(0.20)


def test_term_structure_slopes_up_when_calm_and_inverts_in_panics():
    assert ob.term_iv(0.15, 90 / 365) > 0.15
    assert ob.term_iv(0.60, 90 / 365) < 0.60
    assert ob.term_iv(0.15, 30 / 365) == pytest.approx(0.15)


# ── strikes and expiries ────────────────────────────────────

def test_expiries_are_fridays_after_entry_near_the_target():
    for d in pd.bdate_range('2024-01-01', periods=40):
        exp = ob.expiry_for(d.date(), 30)
        assert exp.weekday() == 4 and exp > d.date()
        assert abs((exp - d.date()).days - 30) <= 3
    assert ob.expiry_for(date(2024, 1, 4), 1) == date(2024, 1, 5)


def test_strikes_round_to_listed_spacing():
    assert ob.strike_step(450, 'SPY') == 1.0
    assert ob.strike_step(340, 'AAPL') == 5.0
    assert ob.round_strike(103.7, 2.5) == 102.5


def test_delta_strikes_land_near_the_target_delta():
    pos = ob.build_position([ob._L('put', 'short', {'mode': 'delta', 'value': 0.30})],
                            date(2024, 1, 2), 100.0, 0.20, 0.0, 30, 'XYZ', True)
    leg = pos[0]
    T = (leg['expiry'] - date(2024, 1, 2)).days / 365
    d = ob.bs_delta('put', 100, leg['strike'], T, 0.0, ob.skewed_iv(0.2, 100, leg['strike'], T))
    assert leg['strike'] < 100 and abs(abs(d) - 0.30) < 0.06


def test_offset_legs_build_a_real_spread():
    legs = ob.PRESETS['iron_condor']['legs']
    pos = ob.build_position(legs, date(2024, 1, 2), 100.0, 0.20, 0.0, 30, 'XYZ', True)
    sp, lp, sc, lc = (leg['strike'] for leg in pos)
    assert lp < sp < 100 < sc < lc


# ── what the user is told ───────────────────────────────────

def test_collar_is_described_as_three_legs_with_each_leg_spelled_out():
    p = ob.PRESETS['collar']
    d = ob.describe_strategy(p['name'], p['legs'], 30)
    assert d['legs_count'] == 3
    assert d['headline'] == 'Protective collar: a three-leg backtest (long stock, long put, short call)'
    assert d['legs'][0] == 'long 100 shares'
    assert d['legs'][1].startswith('long 1 put 5% below the price')
    assert d['legs'][2].startswith('short 1 call 5% above the price')


def test_protective_put_is_two_legs_long_stock_and_long_put():
    p = ob.PRESETS['protective_put']
    d = ob.describe_strategy(p['name'], p['legs'], 30)
    assert d['headline'] == 'Protective put: a two-leg backtest (long stock, long put)'


def test_every_preset_describes_itself_and_counts_its_legs():
    for key, p in ob.PRESETS.items():
        d = ob.describe_strategy(p['name'], p['legs'], 30)
        assert d['legs_count'] == len(p['legs']) == len(d['legs'])
        assert 'backtest' in d['headline']
    groups = {p['group'] for p in ob.PRESETS.values()}
    assert groups == {'Single leg', 'Two legs', 'Three legs', 'Four legs', 'Cycle'}


# ── validation ──────────────────────────────────────────────

@pytest.mark.parametrize('bad', [
    [],
    [{'type': 'stock', 'side': 'long'}],
    [{'type': 'future', 'side': 'long'}],
    [{'type': 'call', 'side': 'long', 'qty': 0}],
    [{'type': 'call', 'side': 'long', 'strike': {'mode': 'delta', 'value': 1.5}}],
    [{'type': 'call', 'side': 'long', 'strike': {'mode': 'offset', 'leg': 0, 'value': 5}}],
    [{'type': 'stock', 'side': 'short'}, {'type': 'call', 'side': 'long'}],
    [{'type': 'call', 'side': 'long', 'dte': 999}],
    [{'type': 'call', 'side': 'long'}] * 7,
])
def test_invalid_legs_are_refused(bad):
    with pytest.raises(ValueError):
        ob.clean_legs(bad)


def test_valid_custom_legs_are_normalised():
    legs = ob.clean_legs([{'type': 'put', 'side': 'short', 'strike': {'mode': 'delta', 'value': '0.3'}},
                          {'type': 'put', 'side': 'long', 'strike': {'mode': 'offset', 'leg': 0, 'value': -5}}])
    assert legs[0]['strike'] == {'mode': 'delta', 'value': 0.3}
    assert legs[1]['strike'] == {'mode': 'offset', 'leg': 0, 'value': -5.0}


# ── capital rules ───────────────────────────────────────────

def _pos(key, S=100.0):
    return ob.build_position(ob.PRESETS[key]['legs'], date(2024, 1, 2), S, 0.2, 0.0, 30, 'XYZ', True)


def test_cash_secured_put_ties_up_the_strike_less_the_premium():
    pos = _pos('short_put')
    need, unbounded = ob.capital_required(pos, 100.0)
    assert not unbounded
    assert need == pytest.approx(pos[0]['strike'] * 100 - pos[0]['price'] * 100)


def test_debit_spread_ties_up_its_debit_and_naked_calls_are_flagged():
    pos = _pos('bull_call_spread')
    need, unbounded = ob.capital_required(pos, 100.0)
    debit = (pos[0]['price'] - pos[1]['price']) * 100
    assert need == pytest.approx(debit) and not unbounded
    assert ob.capital_required(_pos('short_call'), 100.0)[1] is True
    assert ob.capital_required(_pos('covered_call'), 100.0)[1] is False


# ── accounting ──────────────────────────────────────────────

def test_profit_and_loss_add_up_exactly_with_no_interest():
    rng = np.random.default_rng(3)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 320)))
    m = market(prices, rate=0.0)
    start, end = window(m)
    for key in ('iron_condor', 'collar', 'long_straddle', 'calendar_call', 'covered_call'):
        r = ob.run_preset(key, 'XYZ', start, end, market=m)
        total_pnl = sum(t['pnl'] for t in r['trades'])
        assert r['metrics']['final_value'] == pytest.approx(100_000 + total_pnl, abs=0.05), key
        assert r['trades'] and r['metrics']['trades'] == len(r['trades'])


def test_long_call_on_a_flat_stock_loses_its_premium_every_cycle():
    m = market([100.0] * 300, rate=0.0)
    start, end = window(m)
    r = ob.run_preset('long_call', 'XYZ', start, end, market=m, spread_pct=0.0, commission=0.0)
    assert all(t['pnl'] < 0 for t in r['trades'])
    for t in r['trades']:
        if t['reason'] == 'expiry':
            assert t['legs'][0]['close'] == 0.0
            assert t['pnl'] == pytest.approx(-t['net_open'], abs=0.01)


def test_costs_only_ever_lower_the_result():
    rng = np.random.default_rng(7)
    m = market(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    start, end = window(m)
    free = ob.run_preset('iron_condor', 'XYZ', start, end, market=m, spread_pct=0.0, commission=0.0)
    paid = ob.run_preset('iron_condor', 'XYZ', start, end, market=m, spread_pct=0.05, commission=1.0)
    assert paid['metrics']['total_costs'] > 0 == free['metrics']['total_costs']
    assert paid['metrics']['final_value'] < free['metrics']['final_value']


def test_no_look_ahead_future_prices_do_not_change_the_first_trade():
    rng = np.random.default_rng(11)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    shocked = base.copy()
    shocked[60:] *= 1.5
    m1, m2 = market(base), market(shocked)
    start, end = window(m1)
    a = ob.run_preset('short_put', 'XYZ', start, end, market=m1)['trades'][0]
    b = ob.run_preset('short_put', 'XYZ', start, end, market=m2)['trades'][0]
    assert a['opened'] == b['opened'] and a['legs'][0]['strike'] == b['legs'][0]['strike']
    assert a['legs'][0]['open'] == b['legs'][0]['open']


def test_profit_target_closes_credit_trades_early():
    m = market([100.0] * 300)
    start, end = window(m)
    r = ob.run_preset('iron_condor', 'XYZ', start, end, market=m, profit_take=50)
    assert any(t['reason'].startswith('profit target') for t in r['trades'])


def test_result_carries_the_model_label_and_an_example_position():
    m = market([100.0 + i * 0.05 for i in range(300)])
    start, end = window(m)
    r = ob.run_preset('collar', 'XYZ', start, end, market=m)
    assert r['model']['priced'] == 'model' and r['model']['limits']
    assert r['strategy']['legs_count'] == 3
    assert len(r['example']['legs']) == 3 and len(r['example']['payoff']) == 61
    assert r['benchmark']['label'] == 'Buy and hold XYZ'


# ── API ─────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    import app as app_module
    rng = np.random.default_rng(5)
    m = market(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 400))))
    monkeypatch.setattr(ob, 'load_market', lambda t, s, e: m)
    monkeypatch.setattr(app_module.limiter, 'enabled', False)
    start, end = window(m)
    return app_module.app.test_client(), start, end


def test_presets_endpoint_lists_every_strategy(client):
    c, _, _ = client
    body = c.get('/api/options/presets').get_json()
    assert len(body['presets']) == len(ob.PRESETS)
    collar = next(p for p in body['presets'] if p['key'] == 'collar')
    assert collar['legs_count'] == 3 and 'three-leg' in collar['headline']


def test_backtest_endpoint_runs_a_preset_and_custom_legs(client):
    c, start, end = client
    r = c.post('/api/options/backtest', json={'ticker': 'xyz', 'preset': 'collar',
                                              'start_date': start, 'end_date': end})
    assert r.status_code == 200
    assert r.get_json()['strategy']['headline'].startswith('Protective collar: a three-leg')
    legs = [{'type': 'put', 'side': 'short', 'strike': {'mode': 'delta', 'value': 0.2}},
            {'type': 'put', 'side': 'long', 'strike': {'mode': 'offset', 'leg': 0, 'value': -5}}]
    r = c.post('/api/options/backtest', json={'ticker': 'XYZ', 'legs': legs, 'name': 'My put spread',
                                              'start_date': start, 'end_date': end, 'profit_take': 50})
    body = r.get_json()
    assert r.status_code == 200 and body['strategy']['legs_count'] == 2
    assert body['strategy']['headline'] == 'My put spread: a two-leg backtest (short put, long put)'


@pytest.mark.parametrize('payload', [
    {'preset': 'nope'},
    {'preset': 'collar', 'ticker': 'not a ticker'},
    {'preset': 'collar', 'dte': 1},
    {'preset': 'collar', 'start_date': '2020-01-01', 'end_date': '2020-01-15'},
    {'legs': [{'type': 'stock', 'side': 'long'}]},
])
def test_backtest_endpoint_rejects_bad_requests(client, payload):
    c, _, _ = client
    assert c.post('/api/options/backtest', json=payload).status_code == 400


def test_compare_endpoint_ranks_every_preset(client):
    c, start, end = client
    body = c.post('/api/options/compare', json={'ticker': 'XYZ', 'start_date': start,
                                                'end_date': end}).get_json()
    rets = [r['total_return'] for r in body['results']]
    assert len(rets) == len(ob.PRESETS) and rets == sorted(rets, reverse=True)
    assert body['benchmark']['label'] == 'Buy and hold XYZ'
