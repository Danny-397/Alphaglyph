import os
import logging
import sqlite3
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────
# Production: set DATABASE_URL to a Postgres connection string (e.g. a free
# Neon / Supabase database) so the bot's track record and trade history SURVIVE
# redeploys. Local development and the test suite leave it unset and transparently
# use SQLite — identical behaviour, no Postgres required.
DATABASE_URL = os.getenv('DATABASE_URL', '').strip()
_USE_PG = DATABASE_URL.startswith(('postgres://', 'postgresql://'))

if _USE_PG:
    import psycopg2
    import psycopg2.extras

# SQLite fallback path (used only when DATABASE_URL is unset).
_LOCAL_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alphaglyph.db')
DB_PATH = os.getenv('DATABASE_PATH', '').strip() or _LOCAL_DB


def _resolve_db_path():
    """Ensure the SQLite DB_PATH parent dir exists; fall back to the local path."""
    global DB_PATH
    parent = os.path.dirname(DB_PATH)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            logger.warning('Cannot create %s — using local SQLite path %s.', parent, _LOCAL_DB)
            DB_PATH = _LOCAL_DB
    return DB_PATH


class _DB:
    """
    One connection API across SQLite (local/tests) and Postgres (production).

    The rest of this module keeps writing plain SQL with '?' placeholders and
    `conn.execute(...).fetchall()` / `.commit()` / `.close()`; this wrapper makes
    that work on both backends ('?' is rewritten to '%s' for Postgres, and rows
    come back dict-accessible either way).
    """

    def __init__(self):
        if _USE_PG:
            kwargs = {} if 'sslmode' in DATABASE_URL else {'sslmode': 'require'}
            self.conn = psycopg2.connect(DATABASE_URL, **kwargs)
        else:
            self.conn = sqlite3.connect(_resolve_db_path())
            self.conn.row_factory = sqlite3.Row
            # WAL: lets API reads and the bot's writes coexist without locking.
            self.conn.execute('PRAGMA journal_mode=WAL')

    def execute(self, sql, params=()):
        if _USE_PG:
            cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql.replace('?', '%s'), params)
            return cur
        return self.conn.execute(sql, params)

    def executescript(self, script):
        if _USE_PG:
            with self.conn.cursor() as cur:
                cur.execute(script)
        else:
            self.conn.executescript(script)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


def get_connection():
    return _DB()


def init_db():
    db   = get_connection()
    pk   = 'SERIAL PRIMARY KEY' if _USE_PG else 'INTEGER PRIMARY KEY AUTOINCREMENT'
    real = 'DOUBLE PRECISION'  if _USE_PG else 'REAL'

    db.executescript(f'''
        CREATE TABLE IF NOT EXISTS trades (
            id          {pk},
            timestamp   TEXT    NOT NULL,
            ticker      TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            shares      {real}  NOT NULL,
            price       {real}  NOT NULL,
            strategy    TEXT    NOT NULL,
            order_id    TEXT,
            entry_price {real},
            pnl         {real},
            pnl_pct     {real},
            regime      TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              {pk},
            timestamp       TEXT   NOT NULL,
            portfolio_value {real} NOT NULL,
            cash            {real} NOT NULL,
            equity          {real} NOT NULL,
            strategy        TEXT   NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bot_state (
            id             INTEGER PRIMARY KEY,
            is_running     INTEGER NOT NULL DEFAULT 0,
            strategy       TEXT    NOT NULL DEFAULT 'adaptive',
            started_at     TEXT,
            initial_value  {real}  DEFAULT 100000,
            risk_tolerance TEXT    NOT NULL DEFAULT 'moderate',
            last_cycle_at  TEXT
        );
    ''')

    # Seed the single bot_state row (id=1) if it doesn't exist yet.
    if _USE_PG:
        db.execute('INSERT INTO bot_state (id, is_running, strategy, initial_value, '
                   'risk_tolerance) VALUES (1, 0, ?, 100000, ?) ON CONFLICT (id) DO NOTHING',
                   ('adaptive', 'moderate'))
    else:
        db.execute('INSERT OR IGNORE INTO bot_state (id, is_running, strategy, '
                   'initial_value, risk_tolerance) VALUES (1, 0, ?, 100000, ?)',
                   ('adaptive', 'moderate'))
    db.commit()

    _migrate(db)   # back-fill columns on an older existing database
    db.close()


def _migrate(db):
    """Add columns that may be missing on an older existing database."""
    ine = 'IF NOT EXISTS ' if _USE_PG else ''
    migrations = [
        f'ALTER TABLE trades    ADD COLUMN {ine}regime TEXT',
        f"ALTER TABLE bot_state ADD COLUMN {ine}risk_tolerance TEXT NOT NULL DEFAULT 'moderate'",
        f'ALTER TABLE bot_state ADD COLUMN {ine}last_cycle_at TEXT',
    ]
    for sql in migrations:
        try:
            db.execute(sql)
            db.commit()
        except Exception:
            db.rollback()   # column already exists (SQLite has no IF NOT EXISTS)


def log_trade(ticker, action, shares, price, strategy,
              order_id=None, entry_price=None, pnl=None, pnl_pct=None,
              regime=None):
    conn = get_connection()
    conn.execute(
        '''INSERT INTO trades
           (timestamp, ticker, action, shares, price, strategy,
            order_id, entry_price, pnl, pnl_pct, regime)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (datetime.utcnow().isoformat(), ticker, action, shares, price,
         strategy, order_id, entry_price, pnl, pnl_pct, regime)
    )
    conn.commit()
    conn.close()


def log_portfolio_snapshot(portfolio_value, cash, equity, strategy):
    conn = get_connection()
    conn.execute(
        'INSERT INTO portfolio_snapshots (timestamp, portfolio_value, cash, equity, strategy) VALUES (?, ?, ?, ?, ?)',
        (datetime.utcnow().isoformat(), portfolio_value, cash, equity, strategy)
    )
    conn.commit()
    conn.close()


def get_trades(limit=50, strategy=None):
    conn = get_connection()
    if strategy:
        rows = conn.execute(
            'SELECT * FROM trades WHERE strategy = ? ORDER BY timestamp DESC LIMIT ?',
            (strategy, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_portfolio_history(strategy=None, limit=500):
    conn = get_connection()
    if strategy:
        rows = conn.execute(
            'SELECT * FROM portfolio_snapshots WHERE strategy = ? ORDER BY timestamp ASC LIMIT ?',
            (strategy, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM portfolio_snapshots ORDER BY timestamp ASC LIMIT ?', (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_bot_state():
    conn = get_connection()
    row = conn.execute('SELECT * FROM bot_state WHERE id = 1').fetchone()
    conn.close()
    return dict(row) if row else None


def update_bot_state(is_running=None, strategy=None, started_at=None,
                     initial_value=None, risk_tolerance=None, last_cycle_at=None):
    conn = get_connection()
    if is_running is not None:
        conn.execute('UPDATE bot_state SET is_running = ? WHERE id = 1', (1 if is_running else 0,))
    if strategy is not None:
        conn.execute('UPDATE bot_state SET strategy = ? WHERE id = 1', (strategy,))
    if started_at is not None:
        conn.execute('UPDATE bot_state SET started_at = ? WHERE id = 1', (started_at,))
    if initial_value is not None:
        conn.execute('UPDATE bot_state SET initial_value = ? WHERE id = 1', (initial_value,))
    if risk_tolerance is not None:
        conn.execute('UPDATE bot_state SET risk_tolerance = ? WHERE id = 1', (risk_tolerance,))
    # last_cycle_at uses '' (not None) as the "run a cycle now" sentinel, so an
    # empty string must still be written — hence the explicit `is not None` check.
    if last_cycle_at is not None:
        conn.execute('UPDATE bot_state SET last_cycle_at = ? WHERE id = 1', (last_cycle_at,))
    conn.commit()
    conn.close()


def get_performance_metrics(strategy=None):
    conn = get_connection()
    if strategy and strategy != 'adaptive':
        rows = conn.execute(
            "SELECT * FROM trades WHERE action = 'SELL' AND strategy = ?", (strategy,)
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM trades WHERE action = 'SELL'").fetchall()
    conn.close()

    if not rows:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
            'avg_win': 0, 'avg_loss': 0, 'best_trade': 0, 'worst_trade': 0
        }

    pnls   = [r['pnl'] for r in rows if r['pnl'] is not None]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    return {
        'total_trades': len(rows),
        'win_rate':     round(len(wins) / len(rows) * 100, 1) if rows else 0,
        'total_pnl':    round(sum(pnls), 2),
        'avg_win':      round(sum(wins)   / len(wins),   2) if wins   else 0,
        'avg_loss':     round(sum(losses) / len(losses), 2) if losses else 0,
        'best_trade':   round(max(pnls), 2) if pnls else 0,
        'worst_trade':  round(min(pnls), 2) if pnls else 0,
    }


def compute_kelly_fraction(strategy: str = None, min_trades: int = 10,
                           half_kelly: bool = True) -> float | None:
    """
    Compute the Kelly Criterion position-sizing fraction from closed trade history.

    Kelly formula:  f* = (b·p − q) / b
      b = average win / average |loss|   (odds ratio)
      p = win rate,  q = 1 − p

    Half-Kelly (default) scales by 0.5 to reduce variance in practice.

    Returns None when fewer than min_trades closed trades are available —
    the caller should fall back to the profile's fixed max_position_pct.
    """
    conn = get_connection()
    if strategy and strategy not in ('adaptive', 'ml'):
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE action='SELL' AND strategy=? AND pnl IS NOT NULL",
            (strategy,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT pnl FROM trades WHERE action='SELL' AND pnl IS NOT NULL"
        ).fetchall()
    conn.close()

    if len(rows) < min_trades:
        return None

    pnls   = [r['pnl'] for r in rows]
    wins   = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    if not wins or not losses:
        return None

    p = len(wins) / len(pnls)
    q = 1.0 - p
    b = (sum(wins) / len(wins)) / (sum(losses) / len(losses))  # odds ratio

    kelly = (b * p - q) / b
    kelly = max(0.0, kelly)

    if half_kelly:
        kelly *= 0.5

    return round(kelly, 4)


def get_live_metrics():
    """Computes Sharpe ratio and max drawdown from live portfolio snapshot history."""
    conn = get_connection()
    rows = conn.execute(
        'SELECT portfolio_value FROM portfolio_snapshots ORDER BY timestamp ASC'
    ).fetchall()
    conn.close()

    if len(rows) < 2:
        return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

    values = [r['portfolio_value'] for r in rows]

    max_dd = 0.0
    peak   = values[0]
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    arr    = np.array(values, dtype=float)
    rets   = np.diff(arr) / arr[:-1]
    std    = float(rets.std())
    sharpe = 0.0
    if std > 0:
        sharpe = round((float(rets.mean()) - 0.04 / 252) / std * np.sqrt(252), 2)

    return {'sharpe_ratio': sharpe, 'max_drawdown': round(max_dd, 2)}
