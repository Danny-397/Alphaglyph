"""
Stock splits in the options backtester: positions trade the actual price and are
adjusted on the ex-date as the exchange adjusts them. Fully offline.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import options_bt as ob            # noqa: E402


def _market(adj, split_at=None, ratio=4.0):
    """Split-adjusted closes `adj`; the actual price is `ratio` times higher before day `split_at`."""
    idx = pd.bdate_range('2022-01-03', periods=len(adj))
    adj = pd.Series(np.asarray(adj, float), index=idx)
    splits = pd.Series(dtype=float)
    raw = adj.copy()
    if split_at is not None:
        splits = pd.Series({idx[split_at]: ratio})
        raw = adj * ob.split_factor(idx, splits)
    return {'close': adj, 'spy': adj, 'vix': pd.Series(18.0, index=idx), 'rate': pd.Series(1.0, index=idx),
            'curve': pd.DataFrame({9: 0.95, 93: 1.08, 183: 1.15}, index=idx),
            'raw_close': raw, 'dividends': pd.Series(dtype=float), 'splits': splits}


def _run(m, preset='short_put', **kw):
    idx = m['close'].index
    p = ob.PRESETS[preset]
    return ob.run('XYZ', idx[25].date().isoformat(), idx[-1].date().isoformat(), p['legs'], name=p['name'],
                  alloc_pct=p['alloc'], market=m, spread_pct=0.0, commission=0.0, **kw)


def test_split_factor_multiplies_every_later_ratio():
    idx = pd.bdate_range('2020-01-01', periods=6)
    f = ob.split_factor(idx, pd.Series({idx[2]: 4.0, idx[4]: 10.0}))
    assert list(f) == [40.0, 40.0, 10.0, 10.0, 1.0, 1.0]


def test_split_words():
    assert ob.split_words(4.0) == '4-for-1 split'
    assert ob.split_words(1.5) == '3-for-2 split'
    assert ob.split_words(0.1) == '1-for-10 reverse split'


def test_tiingo_rows_give_actual_prices_and_both_actions():
    rows = [{'date': '2024-06-07T00:00:00.000Z', 'close': 1208.88, 'divCash': 0.0, 'splitFactor': 1.0},
            {'date': '2024-06-10T00:00:00.000Z', 'close': 121.79, 'divCash': 0.0, 'splitFactor': 10.0},
            {'date': '2024-06-11T00:00:00.000Z', 'close': 120.91, 'divCash': 0.01, 'splitFactor': 1.0}]
    raw, dv, sp = ob.tiingo_frame(rows)
    assert raw.iloc[0] == 1208.88
    assert list(sp) == [10.0] and sp.index[0] == pd.Timestamp('2024-06-10')
    assert list(dv) == [0.01]


def test_trades_the_actual_price_not_the_split_adjusted_one():
    m = _market(np.full(160, 50.0) * np.linspace(1, 1.05, 160), split_at=150)
    r = _run(m)
    assert r['trades'][0]['underlying_open'] == pytest.approx(4 * m['close'].iloc[25], abs=0.01)
    assert 'Splits on' in r['model']['splits']


def test_a_split_mid_position_is_not_a_loss():
    """The same history with and without a 4-for-1 split gives the same P&L, trade for trade."""
    n = 200
    adj = 50.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0004, 0.012, n)))
    plain = _market(adj * 4.0)                       # no split: always the pre-split scale
    split = _market(adj, split_at=120)               # actual price 4x higher, then splits
    a, b = _run(plain), _run(split)
    spanning = [t for t in b['trades'] if t['split']]
    assert len(spanning) == 1
    assert '4-for-1 split' in spanning[0]['split']
    pnl_a = [t['pnl'] for t in a['trades']]
    pnl_b = [t['pnl'] for t in b['trades']]
    assert len(pnl_a) == len(pnl_b)
    # Before the split the runs are identical; afterwards strikes round on a finer grid at a quarter of the price.
    k = next(i for i, t in enumerate(b['trades']) if t['split'])
    assert pnl_b[:k + 1] == pytest.approx(pnl_a[:k + 1], rel=1e-6, abs=0.05)
    assert b['metrics']['total_return'] == pytest.approx(a['metrics']['total_return'], abs=3.0)


def test_the_split_trade_keeps_its_books_balanced():
    n = 200
    adj = np.linspace(50, 60, n)
    r = _run(_market(adj, split_at=100), preset='covered_call')
    t = next(t for t in r['trades'] if t['split'])
    stock = next(leg for leg in t['legs'] if leg['type'] == 'stock')
    call = next(leg for leg in t['legs'] if leg['type'] == 'call')
    assert stock['qty'] == pytest.approx(4 * t['units'])        # contracts x4 after the split
    assert call['qty'] == pytest.approx(4 * t['units'])
    assert call['strike'] < 70                                   # post-split strike, not 4x the price
    vals = [e['value'] for e in r['equity']]
    jumps = np.abs(np.diff(vals)) / np.array(vals[:-1])
    assert jumps.max() < 0.05                                    # no fake jump on the ex-date


def test_without_actual_prices_the_run_says_splits_are_approximate():
    m = {**_market(np.linspace(50, 60, 120)), 'raw_close': None, 'dividends': None}
    r = _run(m)
    assert 'split-adjusted' in r['model']['splits']
