import sqlite3
import os
from datetime import datetime
import numpy as np

# Allow DATABASE_PATH env var so Render's persistent disk can be used.
# On Render: set DATABASE_PATH=/data/tradebot.db and mount a disk at /data/.
# Without this, the DB lives on Render's ephemeral filesystem and is wiped on every deploy.
DB_PATH = os.getenv(
    'DATABASE_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tradebot.db')
)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # WAL mode: allows concurrent reads while the bot thread writes snapshots.
    # Without this, Flask API reads and the bot write thread can collide and
    # produce "database is locked" errors.
    conn.execute('PRAGMA journal_mode=WAL')
    return conn


def init_db():
    conn = get_connection()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            ticker      TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            shares      REAL    NOT NULL,
            price       REAL    NOT NULL,
            strategy    TEXT    NOT NULL,
            order_id    TEXT,
            entry_price REAL,
            pnl         REAL,
            pnl_pct     REAL,
            regime      TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            portfolio_value REAL    NOT NULL,
            cash            REAL    NOT NULL,
            equity          REAL    NOT NULL,
            strategy        TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS bot_state (
            id             INTEGER PRIMARY KEY,
            is_running     INTEGER NOT NULL DEFAULT 0,
            strategy       TEXT    NOT NULL DEFAULT 'adaptive',
            started_at     TEXT,
            initial_value  REAL    DEFAULT 100000,
            risk_tolerance TEXT    NOT NULL DEFAULT 'moderate'
        );
    ''')
    conn.execute(
        '''INSERT OR IGNORE INTO bot_state
           (id, is_running, strategy, initial_value, risk_tolerance)
           VALUES (1, 0, "adaptive", 100000, "moderate")'''
    )
    conn.commit()

    # Safe migration: add columns that may not exist in older DBs
    _migrate(conn)
    conn.close()


def _migrate(conn):
    """Add new columns to existing tables without dropping data."""
    migrations = [
        ('trades',    'ALTER TABLE trades    ADD COLUMN regime TEXT'),
        ('bot_state', 'ALTER TABLE bot_state ADD COLUMN risk_tolerance TEXT NOT NULL DEFAULT "moderate"'),
    ]
    for _table, sql in migrations:
        try:
            conn.execute(sql)
            conn.commit()
        except Exception:
            pass  # column already exists


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
                     initial_value=None, risk_tolerance=None):
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
