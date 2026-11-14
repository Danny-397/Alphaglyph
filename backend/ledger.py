"""
Live prediction ledger — a time-stamped, append-only forward track record.

Every time the Predictor is asked about a ticker, the directional call it made
is written here *at that moment* (price, p_up, direction, horizon). Once the
horizon has elapsed the row is graded against what actually happened. Nothing is
ever edited or deleted after the fact, so the resulting hit-rate / Brier score is
a genuine out-of-sample record that cannot be retro-fitted in the model's favour.

This is the honest counterpart to calibration.py: calibration measures the model
on *historical* out-of-sample data we already had; the ledger measures it on the
*future*, one real day at a time.

Storage
-------
Uses the same SQLite/Postgres layer as the rest of the app (database.py). With no
DATABASE_URL the table lives in the local SQLite file — it persists across
requests within a deploy but is wiped on a Render redeploy. Set DATABASE_URL to a
free Postgres instance and the ledger survives redeploys, which is what you want
for a track record that accrues over months.

Grading is *lazy*: get_ledger() first matures any pending rows whose horizon has
elapsed, so a plain GET keeps the record current with no cron required (though an
external ping to /api/predict/ledger still helps it stay fresh).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pandas as pd

import database as db
import features as feat
import track_record

logger = logging.getLogger(__name__)

# Don't log the same ticker more than once per UTC day: repeated Predictor loads
# (and its 30-min cache) would otherwise flood the ledger with duplicates.
_DEDUP_PER_DAY = True

_schema_ready = False


def _ensure_schema() -> None:
    """Create the predictions table if it doesn't exist (idempotent).

    Self-contained so the stateless API doesn't depend on the bot's init_db().
    """
    global _schema_ready
    if _schema_ready:
        return
    conn = db.get_connection()
    try:
        pk   = 'SERIAL PRIMARY KEY' if db._USE_PG else 'INTEGER PRIMARY KEY AUTOINCREMENT'
        real = 'DOUBLE PRECISION'  if db._USE_PG else 'REAL'
        conn.executescript(f'''
            CREATE TABLE IF NOT EXISTS predictions (
                id             {pk},
                created_at     TEXT   NOT NULL,
                created_date   TEXT   NOT NULL,
                ticker         TEXT   NOT NULL,
                horizon        INTEGER NOT NULL,
                direction      TEXT   NOT NULL,
                signal         TEXT,
                p_up           {real} NOT NULL,
                confidence     {real},
                price_at_pred  {real} NOT NULL,
                due_date       TEXT   NOT NULL,
                status         TEXT   NOT NULL DEFAULT 'pending',
                graded_at      TEXT,
                price_at_grade {real},
                realized_pct   {real},
                realized_up    INTEGER,
                correct        INTEGER
            );
        ''')
        conn.commit()
        _schema_ready = True
    except Exception as exc:
        conn.rollback()
        logger.warning('ledger: could not ensure schema: %s', exc)
    finally:
        conn.close()


def _due_date(created: datetime, horizon: int) -> pd.Timestamp:
    """The trading day the call matures on: created + `horizon` business days."""
    return track_record.due_date(created.date(), horizon)


# ── Logging ──────────────────────────────────────────────────────────────────

def log_prediction(prediction: dict) -> int | None:
    """Append one Predictor result to the ledger.

    Returns the new row id, or None when the call is not loggable (unavailable,
    no price) or a row for this ticker already exists today.
    """
    if not prediction or not prediction.get('available'):
        return None
    price = prediction.get('price')
    pred  = prediction.get('prediction') or {}
    try:
        ok = math.isfinite(float(price)) and float(price) > 0 and math.isfinite(float(pred['p_up']))
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:                                  # NaN passes `not price` and `price <= 0`
        return None

    _ensure_schema()
    now       = datetime.now(timezone.utc)
    today_str = now.date().isoformat()
    ticker    = prediction['ticker']
    horizon   = int(prediction.get('horizon') or 5)

    conn = db.get_connection()
    try:
        if _DEDUP_PER_DAY:
            existing = conn.execute(
                'SELECT id FROM predictions WHERE ticker = ? AND created_date = ?',
                (ticker, today_str)).fetchone()
            if existing:
                return None

        cur = conn.execute(
            '''INSERT INTO predictions
               (created_at, created_date, ticker, horizon, direction, signal,
                p_up, confidence, price_at_pred, due_date, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')''',
            (now.isoformat(), today_str, ticker, horizon,
             pred.get('direction', 'NEUTRAL'), pred.get('signal'),
             float(pred['p_up']), pred.get('confidence'),
             float(price), _due_date(now, horizon).date().isoformat()))
        conn.commit()
        # lastrowid is not portable across backends; fine to return best-effort.
        return getattr(cur, 'lastrowid', None)
    except Exception as exc:
        conn.rollback()
        logger.warning('ledger: log failed for %s: %s', ticker, exc)
        return None
    finally:
        conn.close()


# ── Grading ──────────────────────────────────────────────────────────────────

def _close_on_or_after(ticker: str, due: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """First available close on or after `due`. None if the market hasn't
    printed a bar there yet (call not matured) or data is unavailable."""
    df = feat.fetch_ohlcv(ticker, period='6mo')
    if df is None or df.empty:
        return None
    fwd = df[df.index >= due]
    if fwd.empty:
        return None
    row = fwd.iloc[0]
    return fwd.index[0], float(row['Close'])


def grade_pending(max_rows: int = 100) -> int:
    """Mature every pending call whose horizon has elapsed. Returns count graded."""
    _ensure_schema()
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    conn  = db.get_connection()
    graded = 0
    try:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE status = 'pending' AND due_date <= ? "
            'ORDER BY due_date ASC LIMIT ?',
            (today.date().isoformat(), max_rows)).fetchall()
        rows = [dict(r) for r in rows]
    except Exception as exc:
        conn.close()
        logger.warning('ledger: could not read pending rows: %s', exc)
        return 0

    for r in rows:
        try:
            due    = pd.Timestamp(r['due_date'])
            landed = _close_on_or_after(r['ticker'], due)
            if landed is None:
                continue                       # not matured / no data yet
            _, close = landed
            out = track_record.grade(float(r['price_at_pred']), close, r['direction'])
            conn.execute(
                "UPDATE predictions SET status='graded', graded_at=?, price_at_grade=?, "
                'realized_pct=?, realized_up=?, correct=? WHERE id=?',
                (datetime.now(timezone.utc).isoformat(), round(close, 4),
                 out['realized_pct'], out['realized_up'], out['correct'], r['id']))
            graded += 1
        except Exception as exc:
            logger.warning('ledger: grading row %s failed: %s', r.get('id'), exc)
    if graded:
        conn.commit()
    conn.close()
    return graded


# ── Read + summary ───────────────────────────────────────────────────────────

def _summary(graded: list[dict]) -> dict:
    """Track-record stats over the graded, directional (non-neutral) calls."""
    return track_record.summary(graded)


def get_ledger(limit: int = 50, grade: bool = True) -> dict:
    """Return the recent ledger rows plus a track-record summary.

    Grades any matured pending rows first (lazy), so a plain GET keeps the record
    up to date without a scheduler.
    """
    _ensure_schema()
    if grade:
        try:
            grade_pending()
        except Exception as exc:
            logger.warning('ledger: lazy grade failed: %s', exc)

    conn = db.get_connection()
    try:
        recent = [dict(r) for r in conn.execute(
            'SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?',
            (int(limit),)).fetchall()]
        graded = [dict(r) for r in conn.execute(
            "SELECT * FROM predictions WHERE status = 'graded'").fetchall()]
        pending_n = conn.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE status = 'pending'").fetchone()
        pending_n = dict(pending_n)['n'] if pending_n else 0
    except Exception as exc:
        conn.close()
        return {'available': False, 'reason': str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return {
        'available':   True,
        'n_total':     len(recent) if len(recent) < limit else None,
        'n_pending':   pending_n,
        'rows':        recent,
        'summary':     _summary(graded),
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'note': ('An append-only forward record: each call is logged at prediction '
                 'time and graded once its horizon elapses. No row is ever edited '
                 'after grading.'),
    }
