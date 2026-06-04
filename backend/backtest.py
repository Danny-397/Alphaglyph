"""
Backtesting engine with regime-aware metrics.

For each trade it records which market regime was active at that point in
time.  The results include a regime_breakdown section showing performance
statistics (win rate, P&L, trade count) broken down by regime.

In adaptive mode the strategy is selected per-date from the regime, exactly
mirroring what the live bot does.

Walk-forward mode splits the date range 70/30 so trades are only recorded
in the out-of-sample 30%.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import features as feat
import regime as reg
import risk

logger = logging.getLogger(__name__)


# ── Signal generation ──────────────────────────────────────────────────────────

def _add_signals(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Compute features and append +1 / -1 / 0 signal column."""
    df = feat.compute_features(df)
    df.dropna(inplace=True)

    if strategy == 'ma_crossover':
        buy  = (df['sma20'].shift(1) <= df['sma50'].shift(1)) & (df['sma20'] > df['sma50'])
        sell = (df['sma20'].shift(1) >= df['sma50'].shift(1)) & (df['sma20'] < df['sma50'])
    elif strategy == 'rsi':
        buy  = df['rsi14'] < 30
        sell = df['rsi14'] > 70
    elif strategy == 'macd':
        buy  = (
            (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) &
            (df['macd_line'] > df['macd_signal']) &
            (df['Volume'] > df['vol_ma20'])
        )
        sell = (
            (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)) &
            (df['macd_line'] < df['macd_signal'])
        )
    elif strategy == 'ml':
        buy  = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)
    else:
        buy  = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)

    df['signal'] = 0
    df.loc[buy,  'signal'] = 1
    df.loc[sell, 'signal'] = -1
    return df


# ── Regime breakdown helper ────────────────────────────────────────────────────

def _regime_breakdown(sell_trades: list[dict]) -> dict:
    """
    Aggregate performance statistics per regime.

    Returns a dict keyed by regime name, each containing:
      trade_count, win_rate, total_pnl, avg_pnl, best_trade, worst_trade
    """
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for t in sell_trades:
        r = t.get('regime') or 'UNKNOWN'
        if t['pnl'] is not None:
            buckets[r].append(t['pnl'])

    breakdown = {}
    for regime_name, pnls in buckets.items():
        wins = [p for p in pnls if p > 0]
        breakdown[regime_name] = {
            'label':       reg.REGIME_LABELS.get(regime_name, regime_name),
            'trade_count': len(pnls),
            'win_rate':    round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            'total_pnl':   round(sum(pnls), 2),
            'avg_pnl':     round(sum(pnls) / len(pnls), 2) if pnls else 0,
            'best_trade':  round(max(pnls), 2) if pnls else 0,
            'worst_trade': round(min(pnls), 2) if pnls else 0,
        }
    return breakdown


# ── Main engine ────────────────────────────────────────────────────────────────

def run_backtest(strategy: str, tickers: list[str],
                 start_date: str, end_date: str,
                 initial_capital: float = 100_000.0,
                 walk_forward: bool = False,
                 risk_tolerance: str = 'moderate') -> dict:
    """
    Run a daily-bar backtest.

    strategy       : 'adaptive' selects strategy per-date from regime detection.
    walk_forward   : only record trades in the final 30% of the date range.
    risk_tolerance : 'conservative' | 'moderate' | 'aggressive'
    """
    profile = risk.get_risk_profile(risk_tolerance)

    # ── Walk-forward split ─────────────────────────────────────────────────
    split_date = None
    if walk_forward:
        s = datetime.strptime(start_date, '%Y-%m-%d')
        e = datetime.strptime(end_date,   '%Y-%m-%d')
        split_date = (s + timedelta(days=int((e - s).days * 0.70))).strftime('%Y-%m-%d')

    # ── Pre-compute regime series from SPY (market proxy) ─────────────────
    # spy_raw is reused below for the benchmark — download once only.
    regime_series: pd.Series | None = None
    spy_raw = feat.fetch_ohlcv('SPY', start=start_date, end=end_date)
    if spy_raw is not None and len(spy_raw) >= 30:
        try:
            regime_series = reg.compute_regime_series(spy_raw)
        except Exception as exc:
            logger.warning('Could not compute regime series: %s', exc)

    def _regime_for_date(date) -> str:
        if regime_series is None:
            return 'RANGING'
        date_str = date.strftime('%Y-%m-%d')
        if date in regime_series.index:
            return str(regime_series.loc[date])
        # Nearest available
        past = regime_series[regime_series.index <= date]
        return str(past.iloc[-1]) if not past.empty else 'RANGING'

    # ── Download and signal-label data ─────────────────────────────────────
    # For adaptive strategy: pre-label each ticker with all three strategies
    is_adaptive = (strategy == 'adaptive')

    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        raw = feat.fetch_ohlcv(ticker, start=start_date, end=end_date)
        if raw is None or len(raw) < 30:
            logger.warning('Skipping %s — insufficient data', ticker)
            continue
        if is_adaptive:
            # Pre-compute signals for all three strategy variants
            tagged = feat.compute_features(raw.copy())
            tagged.dropna(inplace=True)
            # Add signal columns for each strategy
            for strat in ('ma_crossover', 'rsi', 'macd'):
                sf = _add_signals(raw.copy(), strat)
                tagged[f'signal_{strat}'] = sf['signal'].reindex(tagged.index).fillna(0)
            data[ticker] = tagged
        else:
            data[ticker] = _add_signals(raw, strategy)

    if not data:
        return {'error': 'No sufficient data for the selected tickers and date range.'}

    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

    # ── Simulation loop ────────────────────────────────────────────────────
    cash      = float(initial_capital)
    positions: dict[str, dict] = {}
    trades:    list[dict]      = []
    port_hist: list[dict]      = []

    for date in all_dates:
        recording = (split_date is None) or (date.strftime('%Y-%m-%d') >= split_date)

        # Determine active strategy for this date
        if is_adaptive:
            day_regime   = _regime_for_date(date)
            day_strategy = reg.get_regime_strategy(day_regime, risk_tolerance)
            signal_col   = f'signal_{day_strategy}' if day_strategy != 'hold' else None

            # Conservative profile: skip HIGH_VOLATILITY entirely
            if day_regime == 'HIGH_VOLATILITY' and not profile['trade_high_vol']:
                signal_col = None

            size_mult = (profile['vol_size_mult']
                         if day_regime == 'HIGH_VOLATILITY' else 1.0)
        else:
            day_regime   = _regime_for_date(date)
            day_strategy = strategy
            signal_col   = 'signal'
            size_mult    = (profile['vol_size_mult']
                            if day_regime == 'HIGH_VOLATILITY' else 1.0)

        port_val = cash
        for tkr, pos in positions.items():
            if tkr in data and date in data[tkr].index:
                port_val += pos['shares'] * float(data[tkr].loc[date, 'Close'])

        if recording:
            port_hist.append({'date': date.strftime('%Y-%m-%d'), 'value': round(port_val, 2)})

        for ticker, df in data.items():
            if date not in df.index:
                continue

            row   = df.loc[date]
            price = float(row['Close'])

            # Resolve signal for this date
            if signal_col and signal_col in df.columns:
                signal = int(row.get(signal_col, 0))
            elif not is_adaptive and 'signal' in df.columns:
                signal = int(row.get('signal', 0))
            else:
                signal = 0

            # ── Stop-loss / take-profit ────────────────────────────────────
            if ticker in positions:
                pos    = positions[ticker]
                entry  = pos['entry']
                reason = None
                if price <= entry * (1 - profile['stop_loss_pct']):
                    reason = 'stop_loss'
                elif price >= entry * (1 + profile['take_profit_pct']):
                    reason = 'take_profit'
                elif signal == -1:
                    reason = 'sell_signal'

                if reason:
                    shares  = pos['shares']
                    pnl     = (price - entry) * shares
                    pnl_pct = (price - entry) / entry * 100
                    cash   += shares * price
                    if recording:
                        trades.append({
                            'date':    date.strftime('%Y-%m-%d'),
                            'ticker':  ticker,
                            'action':  'SELL',
                            'price':   round(price,   2),
                            'shares':  shares,
                            'pnl':     round(pnl,     2),
                            'pnl_pct': round(pnl_pct, 2),
                            'reason':  reason,
                            'regime':  day_regime,
                            'strategy': day_strategy,
                        })
                    del positions[ticker]

            # ── Buy signal ─────────────────────────────────────────────────
            elif signal == 1 and price > 0 and recording and signal_col is not None:
                max_spend   = port_val * profile['max_position_pct']
                usable_cash = cash - port_val * profile['min_cash_reserve']
                if usable_cash > price:
                    spend  = min(max_spend * size_mult, usable_cash)
                    shares = int(spend / price)
                    if shares > 0:
                        cash -= shares * price
                        positions[ticker] = {'shares': shares, 'entry': price}
                        trades.append({
                            'date':    date.strftime('%Y-%m-%d'),
                            'ticker':  ticker,
                            'action':  'BUY',
                            'price':   round(price, 2),
                            'shares':  shares,
                            'pnl':     None,
                            'pnl_pct': None,
                            'reason':  'buy_signal',
                            'regime':  day_regime,
                            'strategy': day_strategy,
                        })

    # ── Performance metrics ────────────────────────────────────────────────
    final_value  = port_hist[-1]['value'] if port_hist else initial_capital
    total_return = (final_value - initial_capital) / initial_capital * 100

    sell_trades = [t for t in trades if t['action'] == 'SELL']
    wins        = [t for t in sell_trades if t['pnl'] and t['pnl'] > 0]
    losses      = [t for t in sell_trades if t['pnl'] and t['pnl'] <= 0]
    pnls        = [t['pnl'] for t in sell_trades if t['pnl'] is not None]

    values = [p['value'] for p in port_hist]
    max_dd = 0.0
    peak   = values[0] if values else initial_capital
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    sharpe = 0.0
    if len(values) > 2:
        arr  = np.array(values, dtype=float)
        rets = np.diff(arr) / arr[:-1]
        std  = rets.std()
        if std > 0:
            sharpe = round((rets.mean() - 0.04 / 252) / std * np.sqrt(252), 2)

    calmar = 0.0
    if max_dd > 0:
        years  = max(len(values) / 252, 1 / 252)
        ann_r  = (final_value / initial_capital) ** (1 / years) - 1
        calmar = round(ann_r / (max_dd / 100), 2)

    # SPY buy-and-hold benchmark — reuse the already-downloaded spy_raw
    spy_df           = spy_raw
    benchmark_return = 0.0
    spy_curve        = []
    if spy_df is not None and len(spy_df) > 1:
        if walk_forward and split_date:
            spy_df = spy_df[spy_df.index >= pd.Timestamp(split_date)]
        if len(spy_df) > 1:
            spy_start        = float(spy_df['Close'].iloc[0])
            spy_end          = float(spy_df['Close'].iloc[-1])
            benchmark_return = (spy_end - spy_start) / spy_start * 100
            spy_curve        = [
                {'date': d.strftime('%Y-%m-%d'),
                 'value': round(initial_capital * float(v) / spy_start, 2)}
                for d, v in spy_df['Close'].items()
            ]

    return {
        'metrics': {
            'total_return':     round(total_return, 2),
            'win_rate':         round(len(wins) / len(sell_trades) * 100, 1) if sell_trades else 0,
            'max_drawdown':     round(max_dd, 2),
            'sharpe_ratio':     sharpe,
            'calmar_ratio':     calmar,
            'total_trades':     len(sell_trades),
            'winning_trades':   len(wins),
            'losing_trades':    len(losses),
            'avg_win':          round(sum(t['pnl'] for t in wins)   / len(wins),   2) if wins   else 0,
            'avg_loss':         round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0,
            'best_trade':       round(max(pnls), 2) if pnls else 0,
            'worst_trade':      round(min(pnls), 2) if pnls else 0,
            'final_value':      round(final_value, 2),
            'initial_capital':  initial_capital,
            'benchmark_return': round(benchmark_return, 2),
        },
        'regime_breakdown': _regime_breakdown(sell_trades),
        'equity_curve': port_hist,
        'spy_curve':    spy_curve,
        'trades':       trades[-200:],
        'walk_forward': {'enabled': walk_forward, 'split_date': split_date},
    }
