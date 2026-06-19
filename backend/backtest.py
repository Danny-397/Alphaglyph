"""
Backtesting engine.

Three production features layered on top of the core simulation:

Transaction cost modelling
    Every buy and sell deducts a configurable commission + slippage cost.
    Results report gross return (pre-cost) alongside net return (post-cost).
    Default: 0.10% commission + 0.05% slippage per trade side.

Rolling Kelly Criterion sizing
    Position size for each buy is determined by the Kelly fraction computed
    from all closed trades PRIOR to that date (no look-ahead).
    Falls back to the profile's fixed max_position_pct until 10 trades exist.

Monte Carlo validation
    After the simulation, the daily return sequence is resampled 1,000 times
    to build a distribution of random paths.  The actual result's percentile
    rank tells you whether performance is statistically meaningful.

Regime-aware metrics
    Every trade is tagged with the market regime at execution time.
    Results include a performance breakdown by regime.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import features as feat
import monte_carlo as mc
import portfolio as portopt
import regime as reg
import risk
import stats as st

logger = logging.getLogger(__name__)


# ── Kelly helper (rolling, no look-ahead) ─────────────────────────────────────

def _kelly(wins: list, losses: list, half: bool = True) -> float | None:
    """
    Kelly fraction from running trade history.
    Returns None when fewer than 10 completed trades are available.
    """
    if len(wins) + len(losses) < 10 or not wins or not losses:
        return None
    p = len(wins) / (len(wins) + len(losses))
    b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))
    f = max(0.0, (b * p - (1 - p)) / b)
    return f * 0.5 if half else f


# ── Signal generation ──────────────────────────────────────────────────────────

def _add_signals(df: pd.DataFrame, strategy: str, ticker: str = '') -> pd.DataFrame:
    raw = df.copy()                  # ml branch needs unmodified OHLCV
    df = feat.compute_features(df)
    df.dropna(inplace=True)

    if strategy == 'ma_crossover':
        # Trend STANCE (not a one-shot cross): stay long while the fast average
        # is above the slow one. This rides the whole uptrend and lets the bot
        # re-enter if a stop knocks it out while the trend is still intact —
        # instead of selling on the first wobble and sitting in cash until a
        # brand-new golden cross that may never come during a sustained run.
        buy  = df['sma20'] > df['sma50']
        sell = df['sma20'] < df['sma50']
    elif strategy == 'rsi':
        # Mean reversion stays event-based: buy oversold, sell overbought.
        buy  = df['rsi14'] < 30
        sell = df['rsi14'] > 70
    elif strategy == 'macd':
        # Momentum STANCE: long while MACD is above its signal line.
        buy  = df['macd_line'] > df['macd_signal']
        sell = df['macd_line'] < df['macd_signal']
    elif strategy == 'ml':
        # Transformer signals: one batched ONNX call over every window in the
        # range.  Each day's window ends on that day — no look-ahead.  Without
        # a deployed model this stays all-zero (no trades) instead of failing.
        import ml_runtime
        ml_sigs = ml_runtime.backtest_signals(raw, ticker or 'SPY')
        if ml_sigs is not None:
            ml_sigs = ml_sigs.reindex(df.index).fillna(0).astype(int)
            buy  = ml_sigs == 1
            sell = ml_sigs == -1
        else:
            buy  = pd.Series(False, index=df.index)
            sell = pd.Series(False, index=df.index)
    elif strategy == 'hold':
        buy  = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)
    else:
        buy  = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)

    df['signal'] = 0
    df.loc[buy,  'signal'] = 1
    df.loc[sell, 'signal'] = -1
    return df


# ── Chart series (price + indicators + signal markers) ──────────────────────────

_CHART_INDICATORS = {
    'ma_crossover': ['sma20', 'sma50'],
    'rsi':          ['rsi14'],
    'macd':         ['macd_line', 'macd_signal', 'macd_hist'],
    'ml':           [],
}


def chart_series(strategy: str, ticker: str, period: str = '1y') -> dict | None:
    """
    Build the data the Stock Explorer draws: the close price, the strategy's
    own indicators, and the exact dates it would have signalled BUY/SELL — so a
    user can see precisely what the strategy "sees" on any stock. Reuses the same
    vectorised signal logic as the backtest, so the markers match real behaviour.
    """
    raw = feat.fetch_ohlcv(ticker, period=period)
    if raw is None or len(raw) < 30:
        return None

    df   = _add_signals(raw, strategy, ticker)
    cols = _CHART_INDICATORS.get(strategy, [])

    series = []
    for idx, row in df.iterrows():
        rec = {'date': idx.strftime('%Y-%m-%d'), 'close': round(float(row['Close']), 2)}
        for c in cols:
            if c in df.columns and pd.notna(row[c]):
                rec[c] = round(float(row[c]), 4)
        series.append(rec)

    signals = [
        {'date':   idx.strftime('%Y-%m-%d'),
         'action': 'BUY' if int(row['signal']) == 1 else 'SELL',
         'price':  round(float(row['Close']), 2)}
        for idx, row in df.iterrows() if int(row['signal']) != 0
    ]

    return {
        'ticker':     ticker,
        'strategy':   strategy,
        'indicators': cols,
        'series':     series,
        'signals':    signals,
    }


# ── Regime breakdown ───────────────────────────────────────────────────────────

def _regime_breakdown(sell_trades: list[dict]) -> dict:
    from collections import defaultdict
    buckets: dict[str, list] = defaultdict(list)
    for t in sell_trades:
        r = t.get('regime') or 'UNKNOWN'
        if t['pnl'] is not None:
            buckets[r].append(t['pnl'])

    out = {}
    for rname, pnls in buckets.items():
        wins = [p for p in pnls if p > 0]
        out[rname] = {
            'label':       reg.REGIME_LABELS.get(rname, rname),
            'trade_count': len(pnls),
            'win_rate':    round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            'total_pnl':   round(sum(pnls), 2),
            'avg_pnl':     round(sum(pnls) / len(pnls), 2) if pnls else 0,
            'best_trade':  round(max(pnls), 2) if pnls else 0,
            'worst_trade': round(min(pnls), 2) if pnls else 0,
        }
    return out


# ── Main engine ────────────────────────────────────────────────────────────────

def run_backtest(
    strategy:         str,
    tickers:          list[str],
    start_date:       str,
    end_date:         str,
    initial_capital:  float = 100_000.0,
    walk_forward:     bool  = False,
    risk_tolerance:   str   = 'moderate',
    commission_pct:   float = 0.001,
    slippage_pct:     float = 0.0005,
    use_markowitz:    bool  = False,
) -> dict:
    """
    Run a full backtest with transaction costs, rolling Kelly sizing,
    regime-aware metrics, and Monte Carlo validation.

    commission_pct  : fraction of trade value paid as commission each side
    slippage_pct    : fraction lost to bid-ask spread each side
    """
    profile    = risk.get_risk_profile(risk_tolerance)
    cost_pct   = commission_pct + slippage_pct   # total cost per trade side

    # ── Walk-forward split ─────────────────────────────────────────────────
    split_date = None
    if walk_forward:
        s = datetime.strptime(start_date, '%Y-%m-%d')
        e = datetime.strptime(end_date,   '%Y-%m-%d')
        split_date = (s + timedelta(days=int((e - s).days * 0.70))).strftime('%Y-%m-%d')

    # ── Markowitz weights — fitted on training window only ─────────────────
    # Using pre-split data avoids look-ahead bias: in walk-forward mode the
    # optimiser only sees data the strategy would have had at decision time.
    markowitz_weights: dict[str, float] = {}
    if use_markowitz:
        opt_end = split_date if (walk_forward and split_date) else end_date
        mz      = portopt.compute_efficient_frontier(tickers, start_date, opt_end)
        if 'max_sharpe' in mz:
            markowitz_weights = mz['max_sharpe']['weights']
            logger.info('Markowitz max-Sharpe weights: %s', markowitz_weights)
        else:
            logger.warning('Markowitz optimisation failed (%s) — using profile sizing',
                           mz.get('error'))

    # ── SPY: regime series + benchmark (single download) ──────────────────
    regime_series: pd.Series | None = None
    spy_raw = feat.fetch_ohlcv('SPY', start=start_date, end=end_date)
    if spy_raw is not None and len(spy_raw) >= 30:
        try:
            regime_series = reg.compute_regime_series(spy_raw)
        except Exception as exc:
            logger.warning('Could not compute regime series: %s', exc)

    def _regime_for(date) -> str:
        if regime_series is None:
            return 'RANGING'
        if date in regime_series.index:
            return str(regime_series.loc[date])
        past = regime_series[regime_series.index <= date]
        return str(past.iloc[-1]) if not past.empty else 'RANGING'

    # ── Data download + signal labelling ──────────────────────────────────
    is_adaptive = (strategy == 'adaptive')
    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        raw = feat.fetch_ohlcv(ticker, start=start_date, end=end_date)
        if raw is None or len(raw) < 30:
            logger.warning('Skipping %s — insufficient data', ticker)
            continue
        if is_adaptive:
            tagged = feat.compute_features(raw.copy())
            tagged.dropna(inplace=True)
            for strat in ('ma_crossover', 'rsi', 'macd'):
                sf = _add_signals(raw.copy(), strat)
                tagged[f'signal_{strat}'] = sf['signal'].reindex(tagged.index).fillna(0)
            data[ticker] = tagged
        else:
            data[ticker] = _add_signals(raw, strategy, ticker)

    if not data:
        return {'error': 'No sufficient data for the selected tickers and date range.'}

    all_dates = sorted(set().union(*[set(df.index) for df in data.values()]))

    # ── Simulation ─────────────────────────────────────────────────────────
    cash           = float(initial_capital)
    positions:     dict[str, dict] = {}
    trades:        list[dict]      = []
    port_hist:     list[dict]      = []
    total_costs    = 0.0

    # Rolling Kelly state (updated after every closed trade)
    kelly_wins:   list[float] = []
    kelly_losses: list[float] = []

    for date in all_dates:
        recording = (split_date is None) or (date.strftime('%Y-%m-%d') >= split_date)

        # Strategy and regime for today
        if is_adaptive:
            day_regime   = _regime_for(date)
            day_strategy = reg.get_regime_strategy(day_regime, risk_tolerance)
            signal_col   = f'signal_{day_strategy}' if day_strategy != 'hold' else None
            if day_regime == 'HIGH_VOLATILITY' and not profile['trade_high_vol']:
                signal_col = None
            size_mult = profile['vol_size_mult'] if day_regime == 'HIGH_VOLATILITY' else 1.0
        else:
            day_regime   = _regime_for(date)
            day_strategy = strategy
            signal_col   = 'signal'
            size_mult    = profile['vol_size_mult'] if day_regime == 'HIGH_VOLATILITY' else 1.0

        # Portfolio value for this day
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

            # Resolve signal
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

                pos['high'] = max(pos['high'], price)
                if price <= pos['high'] * (1 - profile['trail_pct']):
                    reason = 'stop_loss'
                elif price >= entry * (1 + profile['take_profit_pct']):
                    reason = 'take_profit'
                elif signal == -1:
                    reason = 'sell_signal'

                if reason:
                    shares = pos['shares']

                    # Transaction cost on sell side
                    sell_cost  = price * shares * cost_pct
                    proceeds   = price * shares - sell_cost
                    cash      += proceeds
                    total_costs += sell_cost

                    pnl     = proceeds - pos['cost_basis']
                    pnl_pct = pnl / pos['cost_basis'] * 100 if pos['cost_basis'] else 0

                    # Update rolling Kelly state
                    if pnl > 0:
                        kelly_wins.append(pnl)
                    else:
                        kelly_losses.append(abs(pnl))

                    if recording:
                        trades.append({
                            'date':     date.strftime('%Y-%m-%d'),
                            'ticker':   ticker,
                            'action':   'SELL',
                            'price':    round(price,   2),
                            'shares':   shares,
                            'pnl':      round(pnl,     2),
                            'pnl_pct':  round(pnl_pct, 2),
                            'reason':   reason,
                            'regime':   day_regime,
                            'strategy': day_strategy,
                            'cost':     round(sell_cost, 2),
                        })
                    del positions[ticker]

            # ── Buy signal ─────────────────────────────────────────────────
            elif signal == 1 and price > 0 and recording and signal_col is not None:
                # When Markowitz is active, cap each ticker's allocation at its
                # optimal weight — replaces the profile's fixed max_position_pct.
                if markowitz_weights:
                    mz_cap      = markowitz_weights.get(ticker, 1.0 / max(len(data), 1))
                    sizing_prof = {**profile, 'max_position_pct': mz_cap}
                else:
                    sizing_prof = profile
                kelly_f     = _kelly(kelly_wins, kelly_losses)
                base_shares = risk.calculate_position_size_kelly(
                    port_val, price, cash, kelly_f, sizing_prof)
                shares = max(int(base_shares * size_mult), 0)

                if shares > 0:
                    buy_cost    = price * shares * cost_pct
                    total_spend = price * shares + buy_cost
                    usable_cash = cash - port_val * profile['min_cash_reserve']

                    if usable_cash >= total_spend:
                        cash        -= total_spend
                        total_costs += buy_cost

                        positions[ticker] = {
                            'shares':     shares,
                            'entry':      price,
                            'high':       price,       # trailing stop high watermark
                            'cost_basis': total_spend,
                        }
                        trades.append({
                            'date':     date.strftime('%Y-%m-%d'),
                            'ticker':   ticker,
                            'action':   'BUY',
                            'price':    round(price, 2),
                            'shares':   shares,
                            'pnl':      None,
                            'pnl_pct':  None,
                            'reason':   'buy_signal',
                            'regime':   day_regime,
                            'strategy': day_strategy,
                            'cost':     round(buy_cost, 2),
                        })

    # ── Performance metrics ────────────────────────────────────────────────
    final_value   = port_hist[-1]['value'] if port_hist else initial_capital
    net_return    = (final_value - initial_capital) / initial_capital * 100
    gross_return  = net_return + (total_costs / initial_capital * 100)

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

    # Kelly fraction from full backtest results (reported as a metric)
    backtest_kelly = None
    if kelly_wins and kelly_losses:
        backtest_kelly = _kelly(kelly_wins, kelly_losses)

    # SPY benchmark (reuse already-downloaded spy_raw)
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

    # ── Monte Carlo validation ─────────────────────────────────────────────
    mc_result = mc.run_simulation(port_hist, initial_capital, sharpe)

    # ── Deflated Sharpe Ratio (corrects for multiple-testing bias) ─────────
    # n_strategies = 4 valid strategies + adaptive = 5 total tested
    port_rets = (np.diff([p['value'] for p in port_hist]) /
                 np.array([p['value'] for p in port_hist[:-1]]))
    dsr_result = (st.deflated_sharpe_ratio(port_rets, n_strategies=5)
                  if len(port_rets) >= 10 else None)

    # ── Fama-French 3-factor decomposition ────────────────────────────────
    ff3_result = st.fama_french_decomposition(port_hist)

    return {
        'metrics': {
            'total_return':     round(net_return,    2),
            'gross_return':     round(gross_return,  2),
            'total_costs':      round(total_costs,   2),
            'win_rate':         round(len(wins) / len(sell_trades) * 100, 1) if sell_trades else 0,
            'max_drawdown':     round(max_dd, 2),
            'sharpe_ratio':     sharpe,
            'calmar_ratio':     calmar,
            'kelly_fraction':   round(backtest_kelly * 100, 2) if backtest_kelly else None,
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
        'monte_carlo':        mc_result,
        'deflated_sharpe':    dsr_result,
        'fama_french':        ff3_result,
        'markowitz_weights':  markowitz_weights if use_markowitz else None,
        'regime_breakdown':   _regime_breakdown(sell_trades),
        'equity_curve':     port_hist,
        'spy_curve':        spy_curve,
        'trades':           trades[-200:],
        'walk_forward':     {'enabled': walk_forward, 'split_date': split_date},
    }
