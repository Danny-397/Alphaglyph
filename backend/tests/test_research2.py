"""
Unit tests for the second research wave:

  * ledger.py    — append-only forward prediction record + lazy grading
  * cpcv.py      — combinatorial purged CV → Probability of Backtest Overfitting
  * costsweep.py — transaction-cost sensitivity sweep + break-even interpolation
  * backtest._regime_conditional — time-based per-regime performance split

Network boundaries (yfinance, the backtest engine) are monkeypatched; the maths
runs on synthetic data so the tests are fast and deterministic.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest


# ── ledger.py ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def _isolated_ledger(monkeypatch):
    """Point the ledger at a throwaway SQLite file and reset its schema flag."""
    import database as db
    import ledger
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    monkeypatch.setattr(db, '_USE_PG', False)
    monkeypatch.setattr(db, 'DB_PATH', path)
    monkeypatch.setattr(db, '_LOCAL_DB', path)
    ledger._schema_ready = False
    yield ledger
    ledger._schema_ready = False
    try:
        os.remove(path)
    except OSError:
        pass


def _prediction(ticker='AAA', direction='BULLISH', p_up=0.62, price=100.0, horizon=5):
    return {
        'ticker': ticker, 'available': True, 'price': price, 'horizon': horizon,
        'prediction': {'direction': direction, 'signal': 'BUY' if direction == 'BULLISH'
                       else 'SELL', 'p_up': p_up, 'confidence': 0.5},
    }


def test_ledger_logs_and_dedups(_isolated_ledger):
    led = _isolated_ledger
    assert led.log_prediction(_prediction()) is not None
    # Second call the same day for the same ticker is deduped.
    assert led.log_prediction(_prediction()) is None
    out = led.get_ledger(grade=False)
    assert out['available'] is True
    assert len([r for r in out['rows'] if r['ticker'] == 'AAA']) == 1
    assert out['n_pending'] == 1


def test_ledger_rejects_unavailable(_isolated_ledger):
    led = _isolated_ledger
    assert led.log_prediction({'ticker': 'X', 'available': False}) is None
    assert led.log_prediction({'ticker': 'X', 'available': True, 'price': 0,
                               'prediction': {'p_up': 0.6}}) is None


def test_ledger_grades_matured_call(monkeypatch, _isolated_ledger):
    led = _isolated_ledger
    # Log a bullish call, then backdate its due_date into the past so it matures.
    led.log_prediction(_prediction(price=100.0))
    conn = __import__('database').get_connection()
    conn.execute("UPDATE predictions SET due_date = '2020-01-06'")
    conn.commit(); conn.close()

    # Price rose 5% by the due date → a bullish call should grade correct.
    idx = pd.date_range('2020-01-06', periods=10, freq='B')
    df = pd.DataFrame({'Close': np.linspace(105, 110, 10),
                       'Open': 105, 'High': 111, 'Low': 104, 'Volume': 1e6}, index=idx)
    monkeypatch.setattr(led.feat, 'fetch_ohlcv', lambda t, period='6mo': df)

    n = led.grade_pending()
    assert n == 1
    out = led.get_ledger(grade=False)
    row = out['rows'][0]
    assert row['status'] == 'graded'
    assert row['correct'] == 1
    assert row['realized_pct'] > 0
    assert out['summary']['hit_rate'] == 100.0


def test_ledger_summary_excludes_neutral(monkeypatch, _isolated_ledger):
    led = _isolated_ledger
    led.log_prediction(_prediction(ticker='NEU', direction='NEUTRAL', p_up=0.5))
    conn = __import__('database').get_connection()
    conn.execute("UPDATE predictions SET due_date = '2020-01-06'")
    conn.commit(); conn.close()
    idx = pd.date_range('2020-01-06', periods=5, freq='B')
    df = pd.DataFrame({'Close': [101, 102, 103, 104, 105]}, index=idx)
    monkeypatch.setattr(led.feat, 'fetch_ohlcv', lambda t, period='6mo': df)
    led.grade_pending()
    out = led.get_ledger(grade=False)
    # Neutral call is graded (recorded) but not scored for hit-rate.
    assert out['summary']['n_scored'] == 0
    assert out['summary']['hit_rate'] is None


# ── cpcv.py ───────────────────────────────────────────────────────────────────

def _synthetic_prices(n=1600, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    idx = pd.date_range('2014-01-01', periods=n, freq='B')
    return pd.DataFrame({'Open': close, 'High': close * 1.01, 'Low': close * 0.99,
                         'Close': close, 'Volume': 1e6}, index=idx)


def test_cpcv_random_prices_show_high_pbo(monkeypatch):
    """On a pure random walk no strategy has real edge, so the in-sample winner
    should degrade badly out-of-sample — PBO should be substantial."""
    import cpcv
    cpcv._CACHE.clear()
    monkeypatch.setattr(cpcv.feat, 'fetch_ohlcv',
                        lambda t, period='5y': _synthetic_prices())
    out = cpcv.compute_cpcv('AAA', n_groups=8)
    assert out['available'] is True
    assert out['n_strategies'] > 10
    assert 0.0 <= out['pbo'] <= 1.0
    assert out['pbo'] > 0.25                    # noise → meaningful overfitting
    assert 0.0 <= out['prob_oos_loss'] <= 1.0
    assert len(out['logit_hist']) == 20
    assert len(out['scatter']) == out['n_combos']


def test_cpcv_insufficient_history(monkeypatch):
    import cpcv
    cpcv._CACHE.clear()
    monkeypatch.setattr(cpcv.feat, 'fetch_ohlcv',
                        lambda t, period='5y': _synthetic_prices(100))
    out = cpcv.compute_cpcv('AAA')
    assert out['available'] is False


def test_cpcv_forces_even_groups(monkeypatch):
    import cpcv
    cpcv._CACHE.clear()
    monkeypatch.setattr(cpcv.feat, 'fetch_ohlcv',
                        lambda t, period='5y': _synthetic_prices())
    out = cpcv.compute_cpcv('AAA', n_groups=7)
    assert out['n_groups'] % 2 == 0


# ── costsweep.py ──────────────────────────────────────────────────────────────

def test_costsweep_reports_decay_and_breakeven(monkeypatch):
    """Fake a backtest whose net return falls as costs rise and cross the
    benchmark — the sweep should recover a sensible break-even point."""
    import costsweep

    def fake_bt(strategy, tickers, start, end, cap, wf, risk,
                commission_pct=0.0, slippage_pct=0.0, custom_rules=None):
        bps = commission_pct * 10_000
        # Net return starts at 20%, loses 0.3%/bp; benchmark fixed at 10%.
        net = 20.0 - 0.3 * bps
        return {'metrics': {'total_return': round(net, 2), 'sharpe_ratio': 1.0,
                            'total_trades': 40, 'total_costs': bps * 5,
                            'benchmark_return': 10.0}}
    monkeypatch.setattr(costsweep.backtester, 'run_backtest', fake_bt)

    out = costsweep.run_cost_sweep('ma_crossover', ['AAA'], '2020-01-01', '2023-01-01')
    assert out['available'] is True
    assert out['benchmark_return'] == 10.0
    assert len(out['points']) == len(costsweep._COST_GRID_BPS)
    # Excess return hits zero at ~33 bp (net 10 == benchmark 10).
    assert out['breakeven_bench_bps'] is not None
    assert 30 <= out['breakeven_bench_bps'] <= 36
    # Net return hits zero at ~67 bp.
    assert 60 <= out['breakeven_zero_bps'] <= 70


def test_costsweep_no_edge_verdict(monkeypatch):
    import costsweep

    def fake_bt(strategy, tickers, start, end, cap, wf, risk,
                commission_pct=0.0, slippage_pct=0.0, custom_rules=None):
        return {'metrics': {'total_return': 5.0, 'sharpe_ratio': 0.2,
                            'total_trades': 10, 'total_costs': 0,
                            'benchmark_return': 12.0}}      # never beats benchmark
    monkeypatch.setattr(costsweep.backtester, 'run_backtest', fake_bt)
    out = costsweep.run_cost_sweep('rsi', ['AAA'], '2020-01-01', '2023-01-01')
    assert out['available'] is True
    assert "no edge" in out['verdict'].lower() or "doesn't beat" in out['verdict'].lower()


# ── backtest._regime_conditional ──────────────────────────────────────────────

def test_regime_conditional_splits_by_regime():
    import backtest as bt
    dates = pd.date_range('2021-01-01', periods=6, freq='B')
    port_hist = [{'date': d.strftime('%Y-%m-%d'), 'value': v}
                 for d, v in zip(dates, [100, 101, 102, 103, 102, 104])]
    spy_curve = [{'date': d.strftime('%Y-%m-%d'), 'value': v}
                 for d, v in zip(dates, [100, 100.5, 101, 101, 101, 101.5])]

    def regime_for(d):
        return 'TRENDING_UP' if d < pd.Timestamp('2021-01-06') else 'RANGING'

    out = bt._regime_conditional(port_hist, spy_curve, regime_for)
    assert set(out) == {'TRENDING_UP', 'RANGING'}
    assert sum(v['days'] for v in out.values()) == 5      # 6 points → 5 return steps
    for v in out.values():
        assert v['excess'] is not None
        assert 'sharpe' in v
