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

# Dip-weighted sizing bounds: a buy at the 52-week low gets up to RANGE_MAX× the
# normal size; at the 52-week high it gets RANGE_MIN×. Opt-in (range_sizing).
RANGE_MAX = 2.5
RANGE_MIN = 0.3

# ── "Dip Buyer" strategy parameters ───────────────────────────────────────────
# A patient value strategy: buy in tranches as a stock falls toward its 52-week
# low, average down on further drops, and sell when it recovers toward the high —
# always keeping cash in reserve so it can buy the next company that sells off.
DIP_BUY_THRESH  = 0.30   # buy while in the bottom 30% of the 52-week range
DIP_SELL_THRESH = 0.80   # sell once it recovers into the top 20%
DIP_TRANCHE_PCT = 0.10   # each tranche ≈ 10% of the portfolio (before the dip multiplier)
DIP_ADD_DROP    = 0.08   # average down after a further 8% drop below avg cost
DIP_MAX_POS_PCT = 0.30   # never let one name exceed 30% of the portfolio
DIP_RESERVE     = 0.10   # always keep ≥10% cash as dry powder for new dips


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

# ── Custom rule evaluator (user-built strategies) ─────────────────────────────
# A safe, whitelist-only evaluator — no eval/exec. Each condition compares an
# indicator to a number or another indicator; conditions combine with all/any.

def _indicator_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == 'range52':                       # position in the 52-week range, 0–1
        lo = df['Close'].rolling(252, min_periods=40).min()
        hi = df['Close'].rolling(252, min_periods=40).max()
        return ((df['Close'] - lo) / (hi - lo)).clip(0, 1)
    if name == 'close':
        return df['Close']
    if name == 'volume':
        return df['Volume'] if 'Volume' in df.columns else pd.Series(0.0, index=df.index)
    return df[name] if name in df.columns else pd.Series(0.0, index=df.index)


def _condition(df: pd.DataFrame, c: dict) -> pd.Series:
    left  = _indicator_series(df, c.get('left'))
    rv    = c.get('right')
    if isinstance(rv, str) and rv:
        right = _indicator_series(df, rv)
        right_prev = right.shift(1)
    else:
        try:
            right = float(rv)
        except (TypeError, ValueError):
            right = 0.0
        right_prev = right
    op = c.get('op')
    if op == 'lt':
        return left < right
    if op == 'gt':
        return left > right
    if op == 'cross_up':
        return (left.shift(1) <= right_prev) & (left > right)
    if op == 'cross_dn':
        return (left.shift(1) >= right_prev) & (left < right)
    return pd.Series(False, index=df.index)


def _eval_group(df: pd.DataFrame, group: dict | None) -> pd.Series:
    conds = (group or {}).get('conditions') or []
    valid = [_condition(df, c) for c in conds if c.get('left') and c.get('op')]
    if not valid:
        return pd.Series(False, index=df.index)
    out = valid[0]
    for s in valid[1:]:
        out = (out & s) if (group or {}).get('logic') == 'all' else (out | s)
    return out.fillna(False)


def _add_signals(df: pd.DataFrame, strategy: str, ticker: str = '',
                 custom_rules: dict | None = None) -> pd.DataFrame:
    raw = df.copy()                  # ml branch needs unmodified OHLCV
    df = feat.compute_features(df)
    df.dropna(inplace=True)

    if strategy == 'ma_crossover':
        # Trend STANCE with a hysteresis band: go long when the 20-day average is
        # at least 1% above the 50-day, and don't exit until it falls 1% below.
        # The dead band stops the position flip-flopping (and bleeding costs) when
        # the two averages hug each other in a choppy market.
        band = 0.01
        buy  = df['sma20'] > df['sma50'] * (1 + band)
        sell = df['sma20'] < df['sma50'] * (1 - band)
    elif strategy == 'rsi':
        # Buy oversold DIPS within an uptrend (price above its 50-day average) and
        # exit when overbought or the uptrend breaks. The trend filter is what
        # keeps mean reversion from catching falling knives in a downtrend — the
        # classic failure mode of buying every oversold reading.
        buy  = (df['rsi14'] < 40) & (df['sma20'] > df['sma50'])
        sell = (df['rsi14'] > 70) | (df['sma20'] < df['sma50'])
    elif strategy == 'macd':
        # Momentum STANCE: long while the MACD line is above zero (a confirmed
        # uptrend). Using the zero line instead of the line/signal cross trades far
        # less — the signal-line cross oscillates constantly and churns the
        # position; the zero-line cross only flips when the trend genuinely turns.
        buy  = df['macd_line'] > 0
        sell = df['macd_line'] < 0
    elif strategy == 'dip_buyer':
        # Value: buy near the 52-week low, sell when recovered toward the high.
        low52  = df['Close'].rolling(252, min_periods=40).min()
        high52 = df['Close'].rolling(252, min_periods=40).max()
        pct    = ((df['Close'] - low52) / (high52 - low52)).clip(0, 1)
        df['low52'], df['high52'] = low52, high52
        buy  = pct < DIP_BUY_THRESH
        sell = pct > DIP_SELL_THRESH
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
    elif strategy == 'custom':
        # User-built rules: buy/sell when their indicator conditions are met.
        buy  = _eval_group(df, (custom_rules or {}).get('buy'))
        sell = _eval_group(df, (custom_rules or {}).get('sell'))
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
    range_sizing:     bool  = False,
    cash_in_market:   bool  = False,
    custom_rules:     dict | None = None,
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
            data[ticker] = _add_signals(raw, strategy, ticker, custom_rules)

    if not data:
        return {'error': 'No sufficient data for the selected tickers and date range.'}

    # 52-week range, for optional dip-weighted sizing. Trailing rolling window
    # (≤ today, no look-ahead): low/high of the prior ~year of closes.
    if range_sizing:
        for df in data.values():
            df['low52']  = df['Close'].rolling(252, min_periods=40).min()
            df['high52'] = df['Close'].rolling(252, min_periods=40).max()

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

    def _dip_tranche(pv, px, csh, row_):
        """Shares for one Dip-Buyer tranche: ~DIP_TRANCHE_PCT of the portfolio,
        scaled up the deeper the price sits in its 52-week range, and bounded so
        the cash reserve is preserved (dry powder for the next dip)."""
        base = pv * DIP_TRANCHE_PCT
        lo, hi = row_.get('low52'), row_.get('high52')
        if pd.notna(lo) and pd.notna(hi) and hi > lo:
            pc = min(max((px - float(lo)) / (float(hi) - float(lo)), 0.0), 1.0)
            base *= RANGE_MAX - (RANGE_MAX - RANGE_MIN) * pc
        spend = min(base, csh - pv * DIP_RESERVE)
        return max(int(spend / px), 0) if px > 0 and spend > 0 else 0

    # Reserve-in-market: when enabled, idle cash earns the S&P 500's return instead
    # of 0% — the strategy's dry powder is parked in VOO/SPY (modelled with the
    # already-downloaded SPY series) and effectively rotated into names as they dip
    # and back out as they recover. Most impactful for the patient Dip Buyer, which
    # otherwise sits on a large cash pile during a rising market.
    spy_close_map = ({d: float(v) for d, v in spy_raw['Close'].items()}
                     if cash_in_market and spy_raw is not None else {})
    prev_spy_close = None

    for date in all_dates:
        recording = (split_date is None) or (date.strftime('%Y-%m-%d') >= split_date)

        # Grow idle cash at the S&P 500's daily return (reserve held in VOO/SPY).
        today_spy = spy_close_map.get(date)
        if today_spy:
            if cash_in_market and prev_spy_close:
                cash *= today_spy / prev_spy_close
            prev_spy_close = today_spy

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

            is_dip = (day_strategy == 'dip_buyer')

            # ── Manage an open position ────────────────────────────────────
            if ticker in positions:
                pos = positions[ticker]
                pos['high'] = max(pos['high'], price)

                if is_dip:
                    # Dip Buyer: no trailing stop / take-profit. Sell only once it
                    # has recovered toward its 52-week high; otherwise average down
                    # on a further drop, while keeping the cash reserve intact.
                    if signal == -1:
                        sh        = pos['shares']
                        sell_cost = price * sh * cost_pct
                        proceeds  = price * sh - sell_cost
                        cash += proceeds
                        total_costs += sell_cost
                        pnl     = proceeds - pos['cost_basis']
                        pnl_pct = pnl / pos['cost_basis'] * 100 if pos['cost_basis'] else 0
                        (kelly_wins if pnl > 0 else kelly_losses).append(abs(pnl))
                        if recording:
                            trades.append({
                                'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'SELL',
                                'price': round(price, 2), 'shares': sh, 'pnl': round(pnl, 2),
                                'pnl_pct': round(pnl_pct, 2), 'reason': 'recovered',
                                'regime': day_regime, 'strategy': day_strategy, 'cost': round(sell_cost, 2)})
                        del positions[ticker]
                    elif recording and price <= pos['entry'] * (1 - DIP_ADD_DROP):
                        add  = _dip_tranche(port_val, price, cash, row)
                        room = int(max(DIP_MAX_POS_PCT * port_val - pos['cost_basis'], 0) / price) if price else 0
                        add  = min(add, room)
                        spend = price * add * (1 + cost_pct)
                        if add > 0 and cash - port_val * DIP_RESERVE >= spend:
                            buy_cost = price * add * cost_pct
                            cash -= spend
                            total_costs += buy_cost
                            tot = pos['shares'] + add
                            pos['cost_basis'] += spend
                            pos['entry']  = pos['cost_basis'] / tot   # average cost (incl. fees)
                            pos['shares'] = tot
                            trades.append({
                                'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'BUY',
                                'price': round(price, 2), 'shares': add, 'pnl': None, 'pnl_pct': None,
                                'reason': 'scale_in', 'regime': day_regime, 'strategy': day_strategy,
                                'cost': round(buy_cost, 2)})
                else:
                    entry  = pos['entry']
                    reason = None
                    if price <= pos['high'] * (1 - profile['trail_pct']):
                        reason = 'stop_loss'
                    elif price >= entry * (1 + profile['take_profit_pct']):
                        reason = 'take_profit'
                    elif signal == -1:
                        reason = 'sell_signal'

                    if reason:
                        shares     = pos['shares']
                        sell_cost  = price * shares * cost_pct
                        proceeds   = price * shares - sell_cost
                        cash      += proceeds
                        total_costs += sell_cost
                        pnl     = proceeds - pos['cost_basis']
                        pnl_pct = pnl / pos['cost_basis'] * 100 if pos['cost_basis'] else 0
                        if pnl > 0:
                            kelly_wins.append(pnl)
                        else:
                            kelly_losses.append(abs(pnl))
                        if recording:
                            trades.append({
                                'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'SELL',
                                'price': round(price, 2), 'shares': shares, 'pnl': round(pnl, 2),
                                'pnl_pct': round(pnl_pct, 2), 'reason': reason, 'regime': day_regime,
                                'strategy': day_strategy, 'cost': round(sell_cost, 2)})
                        del positions[ticker]

            # ── Open a new position on a buy signal ────────────────────────
            elif signal == 1 and price > 0 and recording and signal_col is not None:
                if is_dip:
                    shares = _dip_tranche(port_val, price, cash, row)
                    cap    = int(DIP_MAX_POS_PCT * port_val / price) if price else 0
                    shares = min(shares, cap)
                    spend  = price * shares * (1 + cost_pct)
                    if shares > 0 and cash - port_val * DIP_RESERVE >= spend:
                        buy_cost = price * shares * cost_pct
                        cash -= spend
                        total_costs += buy_cost
                        positions[ticker] = {'shares': shares, 'entry': spend / shares,
                                             'high': price, 'cost_basis': spend}
                        trades.append({
                            'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'BUY',
                            'price': round(price, 2), 'shares': shares, 'pnl': None, 'pnl_pct': None,
                            'reason': 'dip_buy', 'regime': day_regime, 'strategy': day_strategy,
                            'cost': round(buy_cost, 2)})
                else:
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

                    # Optional dip-weighted sizing: bigger near the 52-week low.
                    rmult = 1.0
                    if range_sizing and 'low52' in df.columns:
                        lo, hi = row.get('low52'), row.get('high52')
                        if pd.notna(lo) and pd.notna(hi) and hi > lo:
                            pct   = min(max((price - float(lo)) / (float(hi) - float(lo)), 0.0), 1.0)
                            rmult = RANGE_MAX - (RANGE_MAX - RANGE_MIN) * pct

                    shares = max(int(base_shares * size_mult * rmult), 0)
                    if shares > 0:
                        buy_cost    = price * shares * cost_pct
                        total_spend = price * shares + buy_cost
                        usable_cash = cash - port_val * profile['min_cash_reserve']
                        if usable_cash >= total_spend:
                            cash        -= total_spend
                            total_costs += buy_cost
                            positions[ticker] = {'shares': shares, 'entry': price, 'high': price,
                                                 'cost_basis': total_spend}
                            trades.append({
                                'date': date.strftime('%Y-%m-%d'), 'ticker': ticker, 'action': 'BUY',
                                'price': round(price, 2), 'shares': shares, 'pnl': None, 'pnl_pct': None,
                                'reason': 'buy_signal', 'regime': day_regime, 'strategy': day_strategy,
                                'cost': round(buy_cost, 2)})

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

    # ── Monte Carlo outcome distribution (fan chart — NOT a skill test) ────
    mc_result = mc.run_simulation(port_hist, initial_capital, sharpe)

    # Strategy daily returns (reused by the skill test and the DSR below)
    port_rets = (np.diff([p['value'] for p in port_hist]) /
                 np.array([p['value'] for p in port_hist[:-1]]))

    # ── Skill test: does the timing beat random market timing? ─────────────
    # Uses the SPY benchmark as the tradable proxy for the null model.
    skill_result = {'enabled': False}
    if spy_curve and len(spy_curve) > 1:
        spy_vals  = np.array([p['value'] for p in spy_curve], dtype=float)
        spy_rets  = np.diff(spy_vals) / spy_vals[:-1]
        skill_result = mc.random_timing_test(sharpe, spy_rets, port_rets)

    # ── Deflated Sharpe Ratio (corrects for multiple-testing bias) ─────────
    # n_strategies = 4 valid strategies + adaptive = 5 total tested
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
        'skill_test':         skill_result,
        'deflated_sharpe':    dsr_result,
        'fama_french':        ff3_result,
        'markowitz_weights':  markowitz_weights if use_markowitz else None,
        'regime_breakdown':   _regime_breakdown(sell_trades),
        'equity_curve':     port_hist,
        'spy_curve':        spy_curve,
        'trades':           trades[-200:],
        'walk_forward':     {'enabled': walk_forward, 'split_date': split_date},
    }
