"""
tools/options_quotes_check.py: model option prices against real archived quotes.
Fully offline: a synthetic archive priced at a known volatility.
"""

import math
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))

import options_bt as ob              # noqa: E402
import options_quotes_check as qc    # noqa: E402


def _context(iv=0.20):
    return {'S': 100.0, 'iv': iv, 'r': 0.04, 'q': 0.0, 'curve': ob.day_curve(iv, None), 'next_report': None}


def _entry(true_iv, day='2026-11-06', expiry='2026-12-04'):
    """Quotes made with flat volatility `true_iv`, a cent either side of the price."""
    T = 28 / 365
    rows = {}
    for kind, key in (('call', 'calls'), ('put', 'puts')):
        rows[key] = []
        for k in (85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0):
            p = ob.bs_price(kind, 100.0, k, T, 0.04, true_iv)
            rows[key].append([k, round(p - 0.01, 2), round(p + 0.01, 2), 10.0, 100.0])
    return {'spot': 100.0, 'expiries': {expiry: rows}}


def test_quote_filters():
    assert qc.quote_ok(1.0, 1.2) == pytest.approx(1.1)
    assert qc.quote_ok(0.0, 0.2) is None                 # no bid
    assert qc.quote_ok(1.0, 2.0) is None                 # spread over half the midpoint
    assert qc.quote_ok(0.01, 0.03) is None               # under five cents


def test_only_out_of_the_money_contracts_are_compared():
    rows = qc.compare_day('XYZ', '2026-11-06', _entry(0.20), _context())
    assert rows
    assert all((r['kind'] == 'call' and r['strike'] >= 100) or (r['kind'] == 'put' and r['strike'] <= 100)
               for r in rows)


def test_market_iv_is_recovered_and_the_error_is_the_model_minus_market():
    c = _context(0.20)
    rows = qc.compare_day('XYZ', '2026-11-06', _entry(0.25), c)
    atm = [r for r in rows if r['strike'] == 100.0]
    assert all(r['market_iv'] == pytest.approx(0.25, abs=0.003) for r in atm)
    for r in rows:
        assert r['iv_err'] == pytest.approx(r['model_iv'] - r['market_iv'])
        assert r['price_err'] == pytest.approx(r['model'] / r['mid'] - 1)
    s = qc.summarize(rows)
    assert s['n'] == len(rows) and s['iv_bias'] < 0          # a 20% model under 25% quotes reads low


def test_a_model_that_matches_the_quotes_lands_inside_the_spread():
    c = _context(0.22)
    c['curve'] = ob.day_curve(0.22, (1.0, 1.0, 1.0))
    T = 28 / 365
    entry = _entry(0.22)
    # Quote exactly what the model says (its term structure and skew), a cent either side.
    for kind, key in (('call', 'calls'), ('put', 'puts')):
        for row in entry['expiries']['2026-12-04'][key]:
            p, _ = ob.model_price(kind, c, row[0], T)
            row[1], row[2] = round(p - 0.01, 2), round(p + 0.01, 2)
    rows = qc.compare_day('XYZ', '2026-11-06', entry, c)
    s = qc.summarize(rows)
    assert s['inside_spread'] == 100.0
    assert abs(s['iv_bias']) < 0.2


def test_atm_point_picks_the_expiry_nearest_a_month():
    rows = qc.compare_day('XYZ', '2026-11-06', _entry(0.25, expiry='2026-12-04'), _context())
    rows += qc.compare_day('XYZ', '2026-11-06', _entry(0.25, expiry='2026-11-13'), _context())
    pt = qc.atm_point(rows)
    assert pt['expiry'] == '2026-12-04'
    assert math.isclose(pt['market'], 25.0, abs_tol=0.5)
