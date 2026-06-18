"""
Core trading bot.

Every 5-minute cycle during market hours:
  1. Detects the current market regime from SPY (broad market proxy)
  2. Selects the optimal strategy for that regime + the user's risk tolerance
  3. Checks risk gates (market open, daily trade limit, cash reserve)
  4. Enforces trailing stop-loss / take-profit on every open position
  5. Generates signals for every watchlist ticker using the selected strategy
  6. Executes BUY / SELL orders via the internal paper-trading simulator
  7. Persists a portfolio snapshot to SQLite

Orders are filled at the real last close price (yfinance).
No external brokerage account or API key is required.
"""

import logging
import os
import threading
from datetime import datetime

import database
import features
import regime as reg
import risk
import simulator
import strategies

logger        = logging.getLogger(__name__)
_bot_thread   = None
_stop_event   = threading.Event()
_activity_log = []
_position_highs: dict[str, float] = {}  # ticker → highest price seen since entry

_last_regime: reg.RegimeResult | None = None

# How often a trading cycle should run, in seconds. Persisted as last_cycle_at
# in bot_state so timing survives process restarts (Render free-tier spin-downs,
# gunicorn worker recycles). Override with BOT_CYCLE_SECONDS for faster demos.
_CYCLE_SECONDS = max(30, int(os.getenv('BOT_CYCLE_SECONDS', '300')))

# A non-reentrant lock that guarantees only ONE trading cycle runs at a time,
# no matter how many requests trigger a tick concurrently (gunicorn runs one
# worker with several threads). Acquired by maybe_run_cycle() and released by
# the cycle thread when it finishes.
_cycle_lock = threading.Lock()


def _autostart_enabled() -> bool:
    return os.getenv('BOT_AUTOSTART', 'true').strip().lower() in ('1', 'true', 'yes', 'on')


# ── Activity log ──────────────────────────────────────────────────────────────

def _log(msg: str):
    ts    = datetime.utcnow().strftime('%H:%M:%S UTC')
    entry = f'[{ts}] {msg}'
    _activity_log.insert(0, entry)
    if len(_activity_log) > 200:
        _activity_log.pop()
    logger.info(msg)


def get_activity_log() -> list[str]:
    return _activity_log[:50]


def get_last_regime() -> dict | None:
    if _last_regime is None:
        return None
    return {
        'regime':      _last_regime.regime,
        'label':       _last_regime.label,
        'description': _last_regime.description,
        'strategy':    _last_regime.strategy,
        'adx':         _last_regime.adx,
        'plus_di':     _last_regime.plus_di,
        'minus_di':    _last_regime.minus_di,
        'bb_width':    _last_regime.bb_width,
        'vol_30d':     _last_regime.vol_30d,
    }


# ── Regime detection ──────────────────────────────────────────────────────────

def _detect_market_regime() -> reg.RegimeResult | None:
    global _last_regime
    try:
        spy_df = features.fetch_ohlcv('SPY', period='6mo')
        if spy_df is not None and len(spy_df) >= 60:
            result = reg.detect_regime(spy_df)
            _last_regime = result
            return result
    except Exception as exc:
        logger.error('Regime detection error: %s', exc)
    return _last_regime


# ── Portfolio summary (read from simulator) ───────────────────────────────────

def get_portfolio_summary() -> dict:
    try:
        account  = simulator.get_account()
        all_pos  = simulator.get_all_positions()

        port_val = account.portfolio_value
        cash     = account.cash
        equity   = account.equity

        state    = database.get_bot_state()
        risk_tol = state.get('risk_tolerance', 'moderate') if state else 'moderate'
        profile  = risk.get_risk_profile(risk_tol)

        positions = []
        for pos in all_pos:
            entry   = pos.avg_entry_price
            current = pos.current_price
            pnl     = pos.unrealized_pl
            pnl_pct = pos.unrealized_plpc
            positions.append({
                'ticker':        pos.symbol,
                'shares':        round(pos.qty,     4),
                'entry_price':   round(entry,        2),
                'current_price': round(current,      2),
                'pnl':           round(pnl,          2),
                'pnl_pct':       round(pnl_pct * 100, 2),
                'stop_loss':     risk.calculate_stop_loss(
                                     entry, profile,
                                     high_price=_position_highs.get(pos.symbol, entry)),
                'take_profit':   risk.calculate_take_profit(entry, profile),
            })

        initial_val  = state['initial_value'] if state else 100_000
        total_return = (port_val - initial_val) / initial_val * 100 if initial_val else 0

        return {
            'portfolio_value':  round(port_val,     2),
            'cash':             round(cash,          2),
            'equity':           round(equity,        2),
            'total_return':     round(total_return,  2),
            'positions':        positions,
            'active_positions': len(positions),
        }

    except Exception as exc:
        logger.error('portfolio fetch error: %s', exc)
        return {'error': str(exc), 'positions': [], 'active_positions': 0}


# ── Order execution ───────────────────────────────────────────────────────────

def _buy(ticker, shares, price, strategy_name, regime_name, profile):
    try:
        order = simulator.submit_buy(ticker, shares, price)
        risk.increment_trade_count()
        _position_highs[ticker] = price
        database.log_trade(ticker, 'BUY', shares, price, strategy_name,
                           order_id=order.id, regime=regime_name)
        _log(f'BUY  {shares:>4} {ticker:<5} @ ${price:>9.2f}  '
             f'[{strategy_name}] [{regime_name}]')
        return True
    except Exception as exc:
        _log(f'BUY ERROR {ticker}: {exc}')
        return False


def _sell(ticker, shares, price, entry, strategy_name, regime_name, reason, profile):
    try:
        order   = simulator.submit_sell(ticker, shares, price)
        risk.increment_trade_count()
        _position_highs.pop(ticker, None)
        pnl     = (price - entry) * shares
        pnl_pct = (price - entry) / entry * 100
        database.log_trade(ticker, 'SELL', shares, price, strategy_name,
                           order_id=order.id,
                           entry_price=entry,
                           pnl=round(pnl,     2),
                           pnl_pct=round(pnl_pct, 2),
                           regime=regime_name)
        _log(f'SELL {shares:>4} {ticker:<5} @ ${price:>9.2f}  '
             f'PnL ${pnl:>+9.2f} ({pnl_pct:>+.1f}%)  [{reason}]')
        return True
    except Exception as exc:
        _log(f'SELL ERROR {ticker}: {exc}')
        return False


# ── Trading cycle ─────────────────────────────────────────────────────────────

def _trading_cycle(configured_strategy: str):
    try:
        account  = simulator.get_account()
        port_val = account.portfolio_value
        cash     = account.cash

        state    = database.get_bot_state()
        risk_tol = state.get('risk_tolerance', 'moderate') if state else 'moderate'
        profile  = risk.get_risk_profile(risk_tol)

        regime_result = _detect_market_regime()
        regime_name   = regime_result.regime if regime_result else 'RANGING'

        if configured_strategy == 'adaptive':
            active_strategy = reg.get_regime_strategy(regime_name, risk_tol)
            _log(f'Regime: {regime_name} | Risk: {risk_tol} → Strategy: {active_strategy}')
        else:
            active_strategy = configured_strategy

        if regime_name == 'HIGH_VOLATILITY' and not profile['trade_high_vol']:
            _log(f'Skipping cycle — HIGH_VOLATILITY regime, {risk_tol} profile avoids it')
            database.log_portfolio_snapshot(
                port_val, cash, account.equity, active_strategy)
            return

        size_mult = profile['vol_size_mult'] if regime_name == 'HIGH_VOLATILITY' else 1.0

        ok, reason = risk.can_trade(port_val, cash, profile)
        if not ok:
            _log(f'Skipping cycle — {reason}')
            database.log_portfolio_snapshot(
                port_val, cash, account.equity, active_strategy)
            return

        # ── Trailing stop / take-profit sweep ─────────────────────────────
        positions = {p.symbol: p for p in simulator.get_all_positions()}
        for ticker, pos in list(positions.items()):
            current = pos.current_price
            entry   = pos.avg_entry_price
            _position_highs[ticker] = max(_position_highs.get(ticker, entry), current)
            trigger = risk.check_stop_take(current, entry, profile,
                                            high_since_entry=_position_highs[ticker])
            if trigger:
                _sell(ticker, pos.qty, current, entry,
                      active_strategy, regime_name, trigger, profile)

        # Refresh after any sells
        positions = {p.symbol: p for p in simulator.get_all_positions()}
        account   = simulator.get_account()
        port_val  = account.portfolio_value
        cash      = account.cash

        # ── Signal-driven trading ─────────────────────────────────────────
        for ticker in strategies.WATCHLIST:
            if _stop_event.is_set():
                break

            signal, price = strategies.get_signal(active_strategy, ticker)
            if signal is None or price is None:
                continue

            if signal == 'BUY' and ticker not in positions:
                kelly       = database.compute_kelly_fraction(active_strategy)
                base_shares = risk.calculate_position_size_kelly(
                    port_val, price, cash, kelly, profile)
                shares = max(int(base_shares * size_mult), 0)
                if shares > 0:
                    ok, _ = risk.can_trade(port_val, cash, profile)
                    if ok:
                        _buy(ticker, shares, price,
                             active_strategy, regime_name, profile)
                        cash -= shares * price

            elif signal == 'SELL' and ticker in positions:
                pos   = positions[ticker]
                entry = pos.avg_entry_price
                curr  = pos.current_price
                _sell(ticker, pos.qty, curr, entry,
                      active_strategy, regime_name, 'signal', profile)

        account = simulator.get_account()
        database.log_portfolio_snapshot(
            account.portfolio_value,
            account.cash,
            account.equity,
            active_strategy,
        )

    except Exception as exc:
        _log(f'Cycle error: {exc}')
        logger.exception('Unhandled error in trading cycle')


# ── Tick-driven engine ─────────────────────────────────────────────────────────
#
# WHY THIS DESIGN (and not a long-lived background thread):
# The previous version ran the bot inside a daemon thread and treated that
# thread's liveness as "is the bot running". On free-tier hosting that is fatal:
# Render spins the whole process down after ~15 min idle and gunicorn recycles
# workers — either event silently kills the thread, so the bot "stopped every
# single time". Worse, on ephemeral SQLite the persisted state was wiped on
# restart, so the thread never even auto-resumed.
#
# The fix: the bot is "running" when bot_state.is_running is set in the database
# — that's the single source of truth. Trading cycles are DUE on a fixed cadence
# (last_cycle_at persisted in the DB) and are driven by ANY incoming request
# (the dashboard polls /api/status every 10 s, and a keep-warm ping hits
# /health). So the bot makes progress whenever the service is awake, with no
# dependence on a thread surviving. A lightweight heartbeat thread still runs
# when the process is continuously alive (local/dev), but nothing depends on it.


def _cycle_due(last_iso: str | None) -> bool:
    if not last_iso:
        return True
    try:
        last = datetime.fromisoformat(last_iso)
    except (ValueError, TypeError):
        return True
    return (datetime.utcnow() - last).total_seconds() >= _CYCLE_SECONDS


def seconds_until_next_cycle() -> int:
    state = database.get_bot_state()
    if not (state and state.get('is_running')):
        return 0
    last = state.get('last_cycle_at')
    if not last:
        return 0
    try:
        elapsed = (datetime.utcnow() - datetime.fromisoformat(last)).total_seconds()
    except (ValueError, TypeError):
        return 0
    return max(0, int(_CYCLE_SECONDS - elapsed))


def _run_cycle_locked(strategy: str):
    """Run one cycle, then release the lock. Always runs in its own thread so the
    request that triggered the tick returns immediately (a cycle can take many
    seconds fetching market data)."""
    try:
        _trading_cycle(strategy)
    except Exception as exc:
        logger.exception('Trading cycle crashed (will retry next cycle): %s', exc)
    finally:
        _cycle_lock.release()


def maybe_run_cycle() -> bool:
    """
    Run a trading cycle if one is due. Safe and cheap to call from any request
    handler — this is what keeps the bot alive on free-tier hosting.

    Returns True if a cycle was kicked off. Non-blocking: the cycle executes in
    a background thread, and the persisted last_cycle_at slot is claimed BEFORE
    starting so concurrent requests (or a redundant heartbeat tick) can never
    double-trade.
    """
    state = database.get_bot_state()
    if not (state and state.get('is_running')):
        return False
    if not _cycle_due(state.get('last_cycle_at')):
        return False
    # Only one cycle at a time. If we can't grab the lock, a cycle is already
    # running — skip silently.
    if not _cycle_lock.acquire(blocking=False):
        return False
    try:
        # Re-read under the lock and claim the slot before doing any work.
        state = database.get_bot_state()
        if not (state and state.get('is_running')) or not _cycle_due(state.get('last_cycle_at')):
            _cycle_lock.release()
            return False
        database.update_bot_state(last_cycle_at=datetime.utcnow().isoformat())
        strategy = state.get('strategy', 'adaptive')
        threading.Thread(target=_run_cycle_locked, args=(strategy,),
                         daemon=True, name='alphaglyph-cycle').start()
        return True
    except Exception as exc:
        # Make sure we never leak the lock on an unexpected failure.
        logger.exception('maybe_run_cycle failed: %s', exc)
        try:
            _cycle_lock.release()
        except RuntimeError:
            pass
        return False


def _heartbeat_loop():
    """Best-effort local driver: when the process stays alive (local/dev, or a
    paid always-on host), this nudges the engine so cycles still run without any
    inbound traffic. On free tiers it simply dies with the process — harmless,
    because request-driven ticks take over."""
    _log('Bot engine started')
    while not _stop_event.is_set():
        try:
            maybe_run_cycle()
        except Exception as exc:
            logger.exception('Heartbeat error (continuing): %s', exc)
        _stop_event.wait(min(60, _CYCLE_SECONDS))


def _ensure_heartbeat():
    global _bot_thread
    if _bot_thread and _bot_thread.is_alive():
        return
    _stop_event.clear()
    _bot_thread = threading.Thread(target=_heartbeat_loop, daemon=True, name='alphaglyph')
    _bot_thread.start()


# ── Public controls ───────────────────────────────────────────────────────────

def start_bot() -> tuple[bool, str]:
    account = simulator.get_account()
    database.update_bot_state(
        is_running=True,
        started_at=datetime.utcnow().isoformat(),
        initial_value=account.portfolio_value,
        last_cycle_at='',                 # empty → first cycle runs immediately
    )
    _ensure_heartbeat()
    _log('Bot started')
    return True, 'Bot started'


def stop_bot() -> tuple[bool, str]:
    _stop_event.set()
    database.update_bot_state(is_running=False)
    _log('Bot stopped')
    return True, 'Stop signal sent'


def resume_if_running() -> bool:
    """
    Make sure the heartbeat thread exists if the persisted state says the bot is
    running. Called at app import and from the /health watchdog. The bot trades
    via request-driven ticks regardless — this just restarts the local heartbeat
    after a restart so a continuously-awake process keeps cycling on its own.
    """
    state = database.get_bot_state()
    if not (state and state.get('is_running')):
        return False
    if _bot_thread and _bot_thread.is_alive():
        return False
    _ensure_heartbeat()
    _log('Bot engine resumed after process restart')
    return True


def ensure_running_default() -> bool:
    """
    Keep the public demo bot ON by default so it is never sitting idle.

    Only auto-starts a *fresh* bot (never started, or a wiped ephemeral DB where
    started_at was reset) — an explicit owner stop on a persistent (Postgres) DB
    is honoured because started_at remains set. Disable with BOT_AUTOSTART=false.
    """
    if not _autostart_enabled():
        return False
    state = database.get_bot_state()
    if state and not state.get('is_running') and not state.get('started_at'):
        start_bot()
        _log('Bot auto-started (BOT_AUTOSTART) — demo bot is on by default')
        return True
    resume_if_running()
    return False


def is_running() -> bool:
    """The bot is running when the persisted state says so — independent of any
    thread's liveness. Request-driven ticks keep it trading even when the
    heartbeat thread has been killed by a spin-down or worker recycle."""
    state = database.get_bot_state()
    return bool(state and state.get('is_running'))


def engine_thread_alive() -> bool:
    """Diagnostic: whether the in-process heartbeat thread is currently alive."""
    return _bot_thread is not None and _bot_thread.is_alive()
