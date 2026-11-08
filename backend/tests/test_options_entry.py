"""
Entry rules (VIX, IV rank, trend) and the model-sensitivity reruns in the
options backtester. Fully offline.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402
import options_robust as orb       # noqa: E402


def _market(prices, vix=18.0, start='2021-01-04'):
    idx = pd.bdate_range(start, periods=len(prices))
    close = pd.Series(np.asarray(prices, float), index=idx)
    vix = pd.Series(np.broadcast_to(np.asarray(vix, float), len(idx)).copy(), index=idx)
    return {'close': close, 'spy': close, 'vix': vix, 'rate': pd.Series(1.0, index=idx),
            'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx)}


def _run(m, key='short_put', skip=230, **kw):
    idx = m['close'].index
    p = ob.PRESETS[key]
    return ob.run('SPY', idx[skip].date().isoformat(), idx[-1].date().isoformat(), p['legs'], p['name'],
                  alloc_pct=p['alloc'], market=m, **kw)


def test_clean_entry_accepts_and_normalises():
    assert ob.clean_entry(None) is None and ob.clean_entry({}) is None
    assert ob.clean_entry({'trend': 'any', 'vix_min': ''}) is None
    assert ob.clean_entry({'vix_min': '20', 'trend': 'Above'}) == {'vix_min': 20.0, 'trend': 'above'}


@pytest.mark.parametrize('raw', [
    {'vix_min': 'lots'}, {'vix_min': 2}, {'ivr_max': 120}, {'trend': 'sideways'},
    {'vix_min': 30, 'vix_max': 20}, {'ivr_min': 80, 'ivr_max': 20}, ['vix'],
])
def test_clean_entry_rejects_bad_rules(raw):
    with pytest.raises(ValueError):
        ob.clean_entry(raw)


def test_describe_entry_reads_as_a_sentence():
    assert ob.describe_entry(None, 'SPY') == ''
    s = ob.describe_entry({'vix_min': 20, 'ivr_max': 50, 'trend': 'above'}, 'SPY')
    assert s == 'only when the VIX is 20 or higher and its IV rank is 50 or lower and SPY is above its 200-day average'
    assert ob.describe_entry({'vix_min': 15, 'vix_max': 25}, 'QQQ') == 'only when the VIX is between 15 and 25'


def test_vix_rule_gates_entries_and_says_how_often_it_allowed_them():
    m = _market(np.linspace(100, 120, 500), vix=15.0)
    base = _run(m)
    none = _run(m, entry={'vix_min': 20})
    assert base['trades'] and not none['trades']
    assert none['notes'][-1].startswith('The entry rules allowed new positions on 0 of')
    same = _run(m, entry={'vix_max': 20})
    assert same['metrics']['final_value'] == base['metrics']['final_value']
    assert 'only when the VIX is 20 or lower' in same['strategy']['rules']


def test_trend_rule_uses_the_200_day_average():
    down = _market(np.linspace(150, 90, 520))
    assert not _run(down, entry={'trend': 'above'})['trades']
    assert _run(down, entry={'trend': 'below'})['trades']


def test_rules_never_close_an_open_position():
    vix = np.r_[np.full(250, 30.0), np.full(270, 12.0)]      # passes the rule, then fails it
    m = _market(np.linspace(100, 110, 520), vix=vix)
    r = _run(m, skip=235, entry={'vix_min': 20})
    assert len(r['trades']) >= 1
    assert all(t['reason'] == 'expiry' for t in r['trades'])
    first_closed = pd.Timestamp(r['trades'][0]['closed'])
    assert first_closed > m['close'].index[250]               # held past the day the VIX fell


def test_iv_rank_rule_full_range_changes_nothing():
    m = _market(np.linspace(100, 120, 520), vix=np.linspace(14, 30, 520))
    base = _run(m)
    assert _run(m, entry={'ivr_min': 0, 'ivr_max': 100})['metrics']['final_value'] == base['metrics']['final_value']
    high = _run(m, entry={'ivr_min': 90})                     # IV only rises here, so its rank stays high
    assert high['trades']


def test_iv_scale_moves_premium_the_right_way():
    m = _market(np.full(400, 100.0), vix=20.0)
    idx = m['close'].index
    # A fixed strike (3% below), so only the premium moves with IV; a delta strike would also shift.
    legs = [{'type': 'put', 'side': 'short', 'qty': 1, 'strike': {'mode': 'pct', 'value': -3}}]

    def final(k):
        return ob.run('SPY', idx[230].date().isoformat(), idx[-1].date().isoformat(), legs, 'Short put',
                      alloc_pct=100, market=m, iv_scale=k)
    lo, mid, hi = (final(k)['metrics']['final_value'] for k in (0.9, 1.0, 1.1))
    assert lo < mid < hi                                       # a put seller in a flat market keeps more premium
    assert final(1.1)['model']['iv_scale'] == 1.1


def test_model_sensitivity_reruns_each_variant():
    m = _market(np.linspace(100, 115, 400), vix=19.0)
    p = ob.PRESETS['iron_condor']
    idx = m['close'].index
    kw = dict(ticker='SPY', start=idx[230].date().isoformat(), end=idx[-1].date().isoformat(), legs=p['legs'],
              name=p['name'], alloc_pct=p['alloc'], market=m)
    base = ob.run(**kw)
    s = orb.model_sensitivity(kw, base)
    assert [v['label'] for v in s['variants']] == ['IV 10% lower', 'IV 10% higher', 'No skew']
    assert s['cagr_low'] <= base['metrics']['cagr'] <= s['cagr_high']
    assert s['cagr_range'] == pytest.approx(s['cagr_high'] - s['cagr_low'], abs=0.011)
    no_skew = orb.model_sensitivity({**kw, 'skew': False}, base)
    assert [v['label'] for v in no_skew['variants']] == ['IV 10% lower', 'IV 10% higher']


@pytest.fixture
def client(monkeypatch):
    import app as app_module
    m = _market(100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.0003, 0.01, 520))), vix=18.0)
    monkeypatch.setattr(ob, 'load_market', lambda t, s, e: m)
    monkeypatch.setattr(app_module.limiter, 'enabled', False)
    idx = m['close'].index
    return app_module.app.test_client(), idx[230].date().isoformat(), idx[-1].date().isoformat()


def test_backtest_endpoint_takes_entry_rules_and_reports_sensitivity(client):
    c, start, end = client
    body = c.post('/api/options/backtest', json={'ticker': 'SPY', 'preset': 'short_put', 'start_date': start,
                                                 'end_date': end, 'entry': {'trend': 'above'}}).get_json()
    assert body['strategy']['entry'] == {'trend': 'above'}
    assert 'above its 200-day average' in body['strategy']['rules']
    assert len(body['sensitivity']['variants']) == 3


@pytest.mark.parametrize('entry', [{'vix_min': 40, 'vix_max': 10}, {'trend': 'up'}, {'ivr_min': -5}])
def test_backtest_endpoint_rejects_bad_entry_rules(client, entry):
    c, start, end = client
    r = c.post('/api/options/backtest', json={'ticker': 'SPY', 'preset': 'short_put', 'start_date': start,
                                              'end_date': end, 'entry': entry})
    assert r.status_code == 400


def test_compare_endpoint_applies_entry_rules(client):
    c, start, end = client
    body = c.post('/api/options/compare', json={'ticker': 'SPY', 'start_date': start, 'end_date': end,
                                                'entry': {'vix_min': 50}}).get_json()
    assert all(r['trades'] == 0 for r in body['results'])


def test_backtest_endpoint_takes_monthly_expiries(client):
    c, start, end = client
    body = c.post('/api/options/backtest', json={'ticker': 'SPY', 'preset': 'short_put', 'start_date': start,
                                                 'end_date': end, 'expiries': 'monthly'}).get_json()
    assert body['strategy']['expiries'] == 'monthly'
    bad = c.post('/api/options/backtest', json={'ticker': 'SPY', 'preset': 'short_put', 'start_date': start,
                                                'end_date': end, 'expiries': 'daily'})
    assert bad.status_code == 400
