import os
import logging
import sqlite3

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────
# Production: set DATABASE_URL to a Postgres connection string (e.g. a free
# Neon / Supabase database) so the Predictor's forward ledger (ledger.py) and the
# price cache (features.py) SURVIVE redeploys. Local development and the test suite leave it unset and transparently
# use SQLite — identical behaviour, no Postgres required.
DATABASE_URL = os.getenv('DATABASE_URL', '').strip()
_USE_PG = DATABASE_URL.startswith(('postgres://', 'postgresql://'))

if _USE_PG:
    import psycopg2
    import psycopg2.extras



def is_postgres() -> bool:
    """True when DATABASE_URL points at Postgres (data survives redeploys)."""
    return _USE_PG


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
    # trades, portfolio_snapshots and bot_state belonged to the retired paper-trading
    # bot. Nothing reads or writes them now; they are kept so existing rows survive.
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
