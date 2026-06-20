"""
Correctness tests for the backtest engine — the things that matter for a
quant project: no look-ahead bias, honest P&L accounting, and a correct
custom-rule evaluator.

Fully offline: market data and the (network-only) Fama-French download are
monkeypatched with deterministic synthetic data, so these are fast and stable.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import backtest as bt


def _synth(n: int = 420) -> pd.DataFrame:
    """A deterministic sine-plus-drift price path that produces real crossovers."""
    idx   = pd.date_range('2020-01-01', periods=n, freq='B')
    i     = np.arange(n)
    close = 100 + 20 * np.sin(i / 15.0) + i * 0.05
    return pd.DataFrame({
        'Open':   close,
        'High':   close * 1.01,
        'Low':    close * 0.99,
        'Close':  close,
        'Volume': 1_000_000 + (i % 10) * 100_000,
    }, index=idx)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    base = _synth()

    def fake_fetch(ticker, period='6mo', interval='1d', start=None, end=None):
        df = base.copy()
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    monkeypatch.setattr(bt.feat, 'fetch_ohlcv', fake_fetch)
    # Fama-French is the only network call left in run_backtest — stub it out.
    monkeypatch.setattr(bt.st, 'fama_french_decomposition', lambda ph: {'enabled': False})


# ── Custom rule evaluator ─────────────────────────────────────────────────────

class TestCustomEvaluator:
    def _df(self):
        return pd.DataFrame({
            'Close':  [10.0, 11, 12, 11, 10],
            'rsi14':  [20.0, 40, 80, 60, 25],
            'sma20':  [1.0,  2,  3,  2,  1],
            'sma50':  [2.0,  2,  2,  2,  2],
        })

    def test_lt_and_gt(self):
        df = self._df()
        lt = list(bt._condition(df, {'left': 'rsi14', 'op': 'lt', 'right': 30}))
        gt = list(bt._condition(df, {'left': 'rsi14', 'op': 'gt', 'right': 70}))
        assert lt == [True, False, False, False, True]
        assert gt == [False, False, True, False, False]

    def test_indicator_vs_indicator(self):
        df = self._df()
        # Close is always above the flat sma50 = 2.
        assert list(bt._condition(df, {'left': 'close', 'op': 'gt', 'right': 'sma50'})) == [True] * 5

    def test_cross_up_and_down(self):
        df = self._df()
        # sma20 = [1,2,3,2,1] crossing the constant 2.
        up = list(bt._condition(df, {'left': 'sma20', 'op': 'cross_up', 'right': 2}))
        dn = list(bt._condition(df, {'left': 'sma20', 'op': 'cross_dn', 'right': 2}))
        assert up == [False, False, True, False, False]
        assert dn == [False, False, False, False, True]

    def test_all_vs_any_logic(self):
        df = self._df()
        g_all = bt._eval_group(df, {'logic': 'all', 'conditions': [
            {'left': 'rsi14', 'op': 'lt', 'right': 30},
            {'left': 'close', 'op': 'gt', 'right': 5},
        ]})
        assert list(g_all) == [True, False, False, False, True]

        g_any = bt._eval_group(df, {'logic': 'any', 'conditions': [
            {'left': 'rsi14', 'op': 'lt', 'right': 30},
            {'left': 'rsi14', 'op': 'gt', 'right': 70},
        ]})
        assert list(g_any) == [True, False, True, False, True]

    def test_empty_group_is_all_false(self):
        df = self._df()
        assert not bt._eval_group(df, {'conditions': []}).any()
        assert not bt._eval_group(df, None).any()


# ── No look-ahead bias ────────────────────────────────────────────────────────

class TestNoLookahead:
    def test_past_trades_unchanged_when_future_truncated(self):
        """
        Truncating the FUTURE must not change PAST decisions. If it did, a signal
        was peeking ahead. We run the same backtest to two different end dates and
        require the trades up to the earlier date to be byte-for-byte identical.
        """
        common = dict(initial_capital=100_000, walk_forward=False,
                      risk_tolerance='moderate', commission_pct=0.001, slippage_pct=0.0005)
        full  = bt.run_backtest('ma_crossover', ['AAA'], '2020-06-01', '2021-06-01', **common)
        trunc = bt.run_backtest('ma_crossover', ['AAA'], '2020-06-01', '2021-01-01', **common)

        cutoff   = '2021-01-01'
        full_pre  = [t for t in full['trades']  if t['date'] <= cutoff]
        trunc_pre = [t for t in trunc['trades'] if t['date'] <= cutoff]

        assert trunc_pre, 'expected some trades before the cutoff'
        assert full_pre == trunc_pre


# ── P&L accounting ────────────────────────────────────────────────────────────

class TestPnLAccounting:
    def _run(self, tickers=('AAA',)):
        return bt.run_backtest('ma_crossover', list(tickers), '2020-06-01', '2021-06-01',
                               100_000, False, 'moderate', 0.001, 0.0005)

    def test_final_value_matches_equity_curve(self):
        r = self._run()
        assert abs(r['metrics']['final_value'] - r['equity_curve'][-1]['value']) < 0.01

    def test_total_return_math(self):
        r = self._run()
        m = r['metrics']
        expected = (m['final_value'] - m['initial_capital']) / m['initial_capital'] * 100
        assert abs(m['total_return'] - round(expected, 2)) < 0.05

    def test_costs_are_nonnegative_and_drag_returns(self):
        m = self._run()['metrics']
        assert m['total_costs'] >= 0
        # Gross (pre-cost) return is never worse than net (post-cost).
        assert m['gross_return'] >= m['total_return'] - 1e-6

    def test_win_and_loss_counts_reconcile(self):
        m = self._run(('AAA', 'BBB'))['metrics']
        assert m['winning_trades'] + m['losing_trades'] == m['total_trades']

    def test_custom_strategy_runs_end_to_end(self):
        rules = {'buy':  {'logic': 'all', 'conditions': [{'left': 'rsi14', 'op': 'lt', 'right': 35}]},
                 'sell': {'logic': 'any', 'conditions': [{'left': 'rsi14', 'op': 'gt', 'right': 70}]}}
        r = bt.run_backtest('custom', ['AAA'], '2020-06-01', '2021-06-01',
                            100_000, False, 'moderate', 0.001, 0.0005, custom_rules=rules)
        assert 'error' not in r
        assert r['metrics']['total_trades'] >= 0
