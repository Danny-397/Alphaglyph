"""
A forward track record kept in git: log today's calls, grade matured ones.

The live ledger (backend/ledger.py) lives in the backend's database. On
Render's free tier, without DATABASE_URL, that is a SQLite file on an
ephemeral disk, wiped on every redeploy, and it only records predictions a
visitor happened to ask for. This builds a record that does accrue:

  - every trading day, the live Predictor is asked about a fixed basket, and
    each call is appended to frontend/data/forward_ledger.csv;
  - once a call's horizon has passed, its row is graded against the first
    close on or after the due date, using the same rules as the live ledger
    (backend/track_record.py);
  - the file is committed by .github/workflows/forward-ledger.yml.

The commit history is the audit trail: each call is in a commit dated before
the outcome existed, and a graded row is filled in once and never edited.

    python tools/forward_ledger.py            # log (if in the window) and grade
    python tools/forward_ledger.py --grade    # grade only
    python tools/forward_ledger.py --force    # log now, ignoring the time window
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'backend'))

import track_record  # noqa: E402

API = os.getenv('ALPHAGLYPH_API', 'https://trading-bot-jy87.onrender.com').rstrip('/')
DATA_DIR = os.path.join(ROOT, 'frontend', 'data')
CSV_PATH = os.path.join(DATA_DIR, 'forward_ledger.csv')
JSON_PATH = os.path.join(DATA_DIR, 'forward_ledger.json')

BASKET = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'JPM']
NY = ZoneInfo('America/New_York')
# From shortly before the close to the end of the trading day. GitHub starts
# scheduled runs late, by three hours on 2026-11-06, so a window that closed
# at 16:15 silently recorded nothing on a normal day. A call made after the
# close uses that day's closing price and is still made days before its
# 5-day outcome exists, so it is as forward as one made at 15:40.
WINDOW = (time(15, 15), time(23, 59, 59))


def can_log(ny_now: datetime) -> bool:
    """Whether a run at this New York time may log today's calls."""
    return WINDOW[0] <= ny_now.time() <= WINDOW[1]


FIELDS = ['created_at', 'created_date', 'ticker', 'horizon', 'direction', 'signal', 'p_up',
          'confidence', 'price_at_pred', 'due_date', 'model_version', 'status',
          'graded_at', 'price_at_grade', 'realized_pct', 'realized_up', 'correct']


# ── File I/O ────────────────────────────────────────────────────────────────

def read_rows(path: str = CSV_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def write_rows(rows: list[dict], path: str = CSV_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator='\n')
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, '') for k in FIELDS})
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


# ── Logging ─────────────────────────────────────────────────────────────────

def row_from_prediction(pred: dict, now: datetime, model_version: str) -> dict | None:
    """A ledger row from one /api/predict response, or None if it is not loggable."""
    if not pred or not pred.get('available'):
        return None
    call = pred.get('prediction') or {}
    price = pred.get('price')
    if not price or price <= 0 or 'p_up' not in call:
        return None
    today = now.astimezone(NY).date()
    horizon = int(pred.get('horizon') or 5)
    return {
        'created_at': now.astimezone(timezone.utc).isoformat(timespec='seconds'),
        'created_date': today.isoformat(),
        'ticker': pred['ticker'],
        'horizon': horizon,
        'direction': call.get('direction', 'NEUTRAL'),
        'signal': call.get('signal', ''),
        'p_up': round(float(call['p_up']), 4),
        'confidence': call.get('confidence', ''),
        'price_at_pred': round(float(price), 4),
        'due_date': track_record.due_date(today, horizon).date().isoformat(),
        'model_version': model_version,
        'status': 'pending',
    }


def log_calls(rows: list[dict], predict, now: datetime, model_version: str,
              basket=BASKET) -> list[str]:
    """Append today's call for each basket ticker not already logged today."""
    today = now.astimezone(NY).date().isoformat()
    done = {r['ticker'] for r in rows if r['created_date'] == today}
    added = []
    for ticker in basket:
        if ticker in done:
            continue
        try:
            row = row_from_prediction(predict(ticker), now, model_version)
        except Exception as exc:                  # one bad ticker never stops the run
            print(f'  {ticker}: prediction failed ({exc})')
            continue
        if row:
            rows.append(row)
            added.append(ticker)
    return added


# ── Grading ─────────────────────────────────────────────────────────────────

def grade_rows(rows: list[dict], close_on_or_after, today: date, now: datetime) -> list[str]:
    """Grade every pending row whose due date has passed. Graded rows are never touched."""
    graded = []
    closes: dict = {}
    for r in rows:
        if r['status'] != 'pending' or r['due_date'] > today.isoformat():
            continue
        key = (r['ticker'], r['due_date'])
        if key not in closes:
            try:
                closes[key] = close_on_or_after(r['ticker'], pd.Timestamp(r['due_date']))
            except Exception as exc:
                print(f'  {r["ticker"]}: no close for {r["due_date"]} ({exc})')
                closes[key] = None
        landed = closes[key]
        if landed is None:
            continue                               # not matured yet, or no data today
        out = track_record.grade(float(r['price_at_pred']), landed, r['direction'])
        r.update({
            'status': 'graded',
            'graded_at': now.astimezone(timezone.utc).isoformat(timespec='seconds'),
            'price_at_grade': round(landed, 4),
            'realized_pct': out['realized_pct'],
            'realized_up': out['realized_up'],
            'correct': '' if out['correct'] is None else out['correct'],
        })
        graded.append(f'{r["ticker"]} {r["created_date"]}')
    return graded


def _typed(r: dict) -> dict:
    def num(v):
        return None if v in ('', None) else float(v)
    out = dict(r)
    out['p_up'] = num(r['p_up'])
    out['price_at_pred'] = num(r['price_at_pred'])
    out['horizon'] = int(r['horizon'])
    for k in ('price_at_grade', 'realized_pct'):
        out[k] = num(r.get(k))
    for k in ('realized_up', 'correct'):
        out[k] = None if r.get(k) in ('', None) else int(float(r[k]))
    return out


def summary_json(rows: list[dict], now: datetime, recent: int = 50) -> dict:
    """Same shape as /api/predict/ledger, so the frontend renders either one."""
    typed = [_typed(r) for r in rows]
    graded = [r for r in typed if r['status'] == 'graded']
    newest = sorted(typed, key=lambda r: (r['created_date'], r['ticker']), reverse=True)
    first = min((r['created_date'] for r in typed), default=None)
    return {
        'available': True,
        'source': 'git',
        'generated_at': now.astimezone(timezone.utc).isoformat(timespec='seconds'),
        'since': first,
        'basket': BASKET,
        'n_total': len(typed),
        'n_pending': sum(r['status'] == 'pending' for r in typed),
        'summary': track_record.summary(graded),
        'rows': newest[:recent],
        'note': ('Logged automatically every trading day near the close for a fixed basket, and '
                 'committed to GitHub before the outcome exists. A row is graded once, against the '
                 'first close on or after its due date, and never edited afterwards. Full history: '
                 'frontend/data/forward_ledger.csv in the repository.'),
    }


# ── Network ─────────────────────────────────────────────────────────────────

def fetch_prediction(ticker: str) -> dict:
    resp = requests.get(f'{API}/api/predict', params={'ticker': ticker}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def fetch_model_version() -> str:
    try:
        info = requests.get(f'{API}/api/ml/info', timeout=60).json()
        return str(info.get('version') or 'unknown')
    except Exception:
        return 'unknown'


def fetch_close_on_or_after(ticker: str, due: pd.Timestamp) -> float | None:
    """First daily close on or after `due`: yfinance, then Stooq as a fallback."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=due.date().isoformat(), interval='1d', auto_adjust=False)
        if df is not None and not df.empty:
            df.index = df.index.tz_localize(None)
            fwd = df[df.index >= due]
            if not fwd.empty:
                return float(fwd['Close'].iloc[0])
            return None
    except Exception as exc:
        print(f'  yfinance failed for {ticker}: {exc}')
    url = f'https://stooq.com/q/d/l/?s={ticker.lower().replace("-", ".")}.us&i=d'
    df = pd.read_csv(io.StringIO(requests.get(url, timeout=30).text), parse_dates=['Date'])
    fwd = df[df['Date'] >= due]
    return float(fwd['Close'].iloc[0]) if not fwd.empty else None


def is_trading_day(day: date) -> bool:
    """True when SPY has a daily bar for `day` (so weekends and holidays are skipped)."""
    try:
        import yfinance as yf
        df = yf.Ticker('SPY').history(period='5d', interval='1d')
        return any(ts.date() == day for ts in df.index)
    except Exception:
        return day.weekday() < 5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grade', action='store_true', help='grade only; do not log new calls')
    ap.add_argument('--force', action='store_true', help='log even outside the time window')
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    ny_now = now.astimezone(NY)
    rows = read_rows()

    if not args.grade:
        if not (can_log(ny_now) or args.force):
            print(f'{ny_now:%H:%M} ET is outside the logging window; grading only.')
        elif not is_trading_day(ny_now.date()):
            print(f'{ny_now.date()} is not a trading day; grading only.')
        else:
            added = log_calls(rows, fetch_prediction, now, fetch_model_version())
            print(f'logged {len(added)} call(s): {" ".join(added) or "none new"}')

    graded = grade_rows(rows, fetch_close_on_or_after, ny_now.date(), now)
    print(f'graded {len(graded)} call(s)' + (f': {", ".join(graded)}' if graded else ''))

    write_rows(rows)
    with open(JSON_PATH, 'w', encoding='utf-8') as fh:
        json.dump(summary_json(rows, now), fh, indent=1)
        fh.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
