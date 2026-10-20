"""
Internal paper-trading simulator — replaces the Alpaca TradingClient.

Orders are filled at the real last close price (from yfinance).
Cash and open positions are persisted in SQLite alongside trades and
portfolio snapshots so all history survives restarts.

SimAccount and SimPosition expose the same attribute names as the
corresponding alpaca-py objects, so bot.py needs only minimal changes.

Tables
------
sim_account   — one row: current cash balance
sim_positions — one row per open position (ticker, shares, entry_price)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

import database as db
import features as feat

logger = logging.getLogger(__name__)


# ── Value objects (mirror alpaca-py interface) ────────────────────────────────

@dataclass
class SimAccount:
    portfolio_value: float
    cash:            float
    equity:          float


@dataclass
class SimPosition:
    symbol:          str
    qty:             float
    avg_entry_price: float
    current_price:   float
    unrealized_pl:   float
    unrealized_plpc: float


@dataclass
class SimOrder:
    id: str


# ── Price helper ──────────────────────────────────────────────────────────────

def _latest_price(ticker: str) -> float | None:
    """Return the most recent close price for ticker via yfinance."""
    df = feat.fetch_ohlcv(ticker, period='5d')
    if df is None or df.empty:
        return None
    return float(df['Close'].iloc[-1])


# ── Database layer ────────────────────────────────────────────────────────────

def init_simulator() -> None:
    """
    Create simulator tables and seed the cash balance.

    Idempotent — safe to call on every app startup.
    Cash is seeded from bot_state.initial_value only on the very first
    call (when the sim_account row does not yet exist).
    """
    conn = db.get_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS sim_positions (
            ticker      TEXT PRIMARY KEY,
            shares      REAL NOT NULL,
            entry_price REAL NOT NULL,
            opened_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sim_account (
            id   INTEGER PRIMARY KEY,
            cash REAL    NOT NULL DEFAULT 100000
        );
    ''')

    existing = conn.execute('SELECT id FROM sim_account WHERE id = 1').fetchone()
    if not existing:
        state   = conn.execute('SELECT initial_value FROM bot_state WHERE id = 1').fetchone()
        capital = float(state['initial_value']) if state else 100_000.0
        conn.execute('INSERT INTO sim_account (id, cash) VALUES (1, ?)', (capital,))

    conn.commit()
    conn.close()
    logger.info('Paper-trading simulator initialised')


def reset(initial_capital: float = 100_000.0) -> None:
    """
    Wipe all open positions and reset cash to initial_capital.
    Call this when starting a brand-new paper-trading session.
    """
    conn = db.get_connection()
    conn.execute('DELETE FROM sim_positions')
    conn.execute('UPDATE sim_account SET cash = ? WHERE id = 1', (initial_capital,))
    conn.commit()
    conn.close()
    logger.info('Simulator reset — cash=%.2f, positions cleared', initial_capital)


def _cash() -> float:
    conn = db.get_connection()
    row  = conn.execute('SELECT cash FROM sim_account WHERE id = 1').fetchone()
    conn.close()
    return float(row['cash']) if row else 100_000.0


def _set_cash(value: float) -> None:
    conn = db.get_connection()
    conn.execute('UPDATE sim_account SET cash = ? WHERE id = 1', (value,))
    conn.commit()
    conn.close()


def _positions_raw() -> list[dict]:
    conn = db.get_connection()
    rows = conn.execute('SELECT * FROM sim_positions').fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Public API ────────────────────────────────────────────────────────────────

def get_account() -> SimAccount:
    """
    Return current portfolio state.

    Equity = sum of (current_price × shares) for all open positions.
    portfolio_value = cash + equity.
    """
    cash      = _cash()
    positions = _positions_raw()

    equity = 0.0
    for pos in positions:
        price   = _latest_price(pos['ticker']) or pos['entry_price']
        equity += pos['shares'] * price

    return SimAccount(
        portfolio_value = cash + equity,
        cash            = cash,
        equity          = equity,
    )


def get_all_positions() -> list[SimPosition]:
    """Return open positions enriched with live prices and unrealised P&L."""
    result = []
    for row in _positions_raw():
        price   = _latest_price(row['ticker']) or row['entry_price']
        entry   = row['entry_price']
        qty     = row['shares']
        unpl    = (price - entry) * qty
        unplpc  = (price - entry) / entry if entry else 0.0
        result.append(SimPosition(
            symbol          = row['ticker'],
            qty             = qty,
            avg_entry_price = entry,
            current_price   = price,
            unrealized_pl   = unpl,
            unrealized_plpc = unplpc,
        ))
    return result


def submit_buy(ticker: str, shares: float, price: float) -> SimOrder:
    """
    Fill a buy order at the given price.

    Deducts cost from cash and records the position.
    Raises ValueError if cash is insufficient.
    """
    cost = shares * price
    cash = _cash()

    if cost > cash + 0.01:   # small tolerance for float rounding
        raise ValueError(
            f'Insufficient cash for {ticker}: need ${cost:.2f}, have ${cash:.2f}'
        )

    conn = db.get_connection()
    existing = conn.execute(
        'SELECT shares, entry_price FROM sim_positions WHERE ticker = ?', (ticker,)
    ).fetchone()

    if existing:
        # Average in to existing position
        total  = existing['shares'] + shares
        avg    = (existing['shares'] * existing['entry_price'] + shares * price) / total
        conn.execute(
            'UPDATE sim_positions SET shares = ?, entry_price = ? WHERE ticker = ?',
            (total, avg, ticker),
        )
    else:
        conn.execute(
            'INSERT INTO sim_positions (ticker, shares, entry_price, opened_at) '
            'VALUES (?, ?, ?, ?)',
            (ticker, shares, price, datetime.now(timezone.utc).replace(tzinfo=None).isoformat()),
        )

    conn.execute('UPDATE sim_account SET cash = ? WHERE id = 1', (cash - cost,))
    conn.commit()
    conn.close()

    logger.debug('SIM BUY  %s x%.0f @ %.2f  cash=%.2f', ticker, shares, price, cash - cost)
    return SimOrder(id=str(uuid.uuid4()))


def submit_sell(ticker: str, shares: float, price: float) -> SimOrder:
    """
    Fill a sell order at the given price.

    Adds proceeds to cash and removes the position.
    Raises ValueError if no position exists.
    """
    conn = db.get_connection()
    existing = conn.execute(
        'SELECT shares FROM sim_positions WHERE ticker = ?', (ticker,)
    ).fetchone()

    if not existing:
        conn.close()
        raise ValueError(f'No open position in {ticker}')

    proceeds = shares * price
    cash     = _cash()

    conn.execute('DELETE FROM sim_positions WHERE ticker = ?', (ticker,))
    conn.execute('UPDATE sim_account SET cash = ? WHERE id = 1', (cash + proceeds,))
    conn.commit()
    conn.close()

    logger.debug('SIM SELL %s x%.0f @ %.2f  cash=%.2f', ticker, shares, price, cash + proceeds)
    return SimOrder(id=str(uuid.uuid4()))
