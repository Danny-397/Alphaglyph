"""
Earnings & fundamentals signal for the Predictor.

A best-effort, graceful-degradation module. It pulls a ticker's recent earnings
history from yfinance and distils it into a single directional tilt built around
**post-earnings-announcement drift (PEAD)** — the long-documented tendency of a
stock to keep drifting in the direction of its latest earnings surprise for weeks
after the report (Ball & Brown 1968; Bernard & Thomas 1989). A beat nudges the
signal bullish, a miss bearish, and the nudge decays as the surprise ages out
over roughly a quarter.

On top of the drift it reads a second, slower signal: year-over-year EPS growth,
computed from the earnings history itself (this quarter's reported EPS vs the
same quarter a year ago) so it needs no extra provider call.

Honest limitations, surfaced to the caller and the UI:
  * Free earnings data (yfinance) is thin and occasionally rate-limited from a
    datacenter IP. When it is missing the caller gets {'available': False} and
    the Predictor simply drops the earnings channel — the same "degrade, don't
    crash" contract the ML modality blocks use.
  * PEAD is a statistical tendency across many names, not a promise for one.
  * We deliberately do NOT try to predict the earnings *gap* itself (near-random);
    we read the *drift* after a report, which is the part with documented edge.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# PEAD fades over ~one quarter of trading — a surprise this old carries no drift.
_DRIFT_DECAY_DAYS = 60
# A report within this many days is treated as an imminent binary event: it
# *raises uncertainty* rather than adding directional conviction.
_EVENT_RISK_DAYS = 7

# Short in-process cache — earnings data changes at most once a quarter, so
# there's no reason to re-hit yfinance on every visitor/poll.
_CACHE: dict = {}
_CACHE_TTL = 6 * 3600  # 6 hours


def _find_col(cols, *needles):
    """Locate a column by case-insensitive substring (yfinance renames over versions)."""
    for c in cols:
        lc = str(c).lower()
        if all(n in lc for n in needles):
            return c
    return None


def _fetch_earnings_frame(ticker: str) -> pd.DataFrame | None:
    """
    Raw earnings-date table from yfinance, tz-naive and sorted newest-first.

    Columns are normalised to: eps_est, eps_actual, surprise_pct (any may be NaN;
    future rows have NaN eps_actual). Returns None when unavailable.
    """
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=16)
    except Exception as exc:
        logger.warning('earnings: yfinance failed for %s: %s', ticker, exc)
        return None
    if raw is None or raw.empty:
        return None

    df = raw.copy()
    # Index is a tz-aware "Earnings Date" — make it tz-naive for clean day math.
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except (TypeError, ValueError):
        df.index = pd.to_datetime(df.index).tz_convert(None)

    est  = _find_col(df.columns, 'eps', 'estimate')
    act  = _find_col(df.columns, 'reported') or _find_col(df.columns, 'eps', 'actual')
    surp = _find_col(df.columns, 'surprise')

    out = pd.DataFrame(index=df.index)
    out['eps_est']      = pd.to_numeric(df[est],  errors='coerce') if est  else np.nan
    out['eps_actual']   = pd.to_numeric(df[act],  errors='coerce') if act  else np.nan
    out['surprise_pct'] = pd.to_numeric(df[surp], errors='coerce') if surp else np.nan
    return out.sort_index(ascending=False)


def earnings_signal(ticker: str) -> dict:
    """
    Directional earnings tilt for one ticker.

    Returns a dict with (when available):
        available        — bool
        score            — directional tilt in [-1, +1] (bullish positive)
        confidence       — [0, 1], how much weight this signal deserves
        surprise_pct     — last reported EPS surprise, %
        days_since       — trading-agnostic calendar days since last report
        eps_yoy_pct      — YoY growth of reported EPS, % (or None)
        next_days        — days until the next scheduled report (or None)
        event_risk       — bool, a report lands within _EVENT_RISK_DAYS
        factors          — list of plain-English strings for the UI
    Always returns {'available': False, ...} instead of raising.
    """
    ticker = ticker.upper().strip()
    hit = _CACHE.get(ticker)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    frame = _fetch_earnings_frame(ticker)
    if frame is None:
        result = {'available': False, 'reason': 'no earnings data', 'factors': []}
        _CACHE[ticker] = (time.time(), result)
        return result

    now = pd.Timestamp.now().normalize()
    reported = frame[frame['eps_actual'].notna()]
    future   = frame[(frame['eps_actual'].isna()) & (frame.index > now)]

    if reported.empty:
        result = {'available': False, 'reason': 'no reported earnings', 'factors': []}
        _CACHE[ticker] = (time.time(), result)
        return result

    last_date   = reported.index[0]
    last        = reported.iloc[0]
    days_since  = int((now - last_date.normalize()).days)
    surprise    = last['surprise_pct']
    # Fall back to computing the surprise if the provider left the column blank.
    if pd.isna(surprise) and pd.notna(last['eps_est']) and last['eps_est'] != 0:
        surprise = (last['eps_actual'] - last['eps_est']) / abs(last['eps_est']) * 100.0
    surprise = float(surprise) if pd.notna(surprise) else 0.0

    # ── PEAD drift component ──────────────────────────────────────────────────
    # Direction = sign of the surprise; magnitude saturates (an ±8% surprise is
    # already a strong signal), then decays linearly as the surprise ages out.
    drift_raw   = math.tanh(surprise / 8.0)
    decay       = max(0.0, 1.0 - days_since / _DRIFT_DECAY_DAYS)
    drift_score = drift_raw * decay

    # ── YoY EPS growth component ──────────────────────────────────────────────
    # Compare the latest reported EPS to the same quarter a year ago (4 reports
    # back). Guards against a zero / negative base, where a "% growth" is noise.
    eps_yoy = None
    if len(reported) >= 5:
        eps_now  = reported.iloc[0]['eps_actual']
        eps_prev = reported.iloc[4]['eps_actual']
        if pd.notna(eps_now) and pd.notna(eps_prev) and eps_prev > 0:
            eps_yoy = float((eps_now - eps_prev) / abs(eps_prev) * 100.0)
    growth_score = math.tanh(eps_yoy / 40.0) if eps_yoy is not None else None

    # ── Blend drift + growth into one earnings score ──────────────────────────
    if growth_score is not None:
        score = 0.65 * drift_score + 0.35 * growth_score
    else:
        score = drift_score

    # ── Next report / event risk ──────────────────────────────────────────────
    next_days, event_risk = None, False
    if not future.empty:
        next_date = future.index.min()
        next_days = int((next_date.normalize() - now).days)
        event_risk = 0 <= next_days <= _EVENT_RISK_DAYS

    # ── Confidence ────────────────────────────────────────────────────────────
    # Fresh, large surprises deserve weight; stale ones don't. An imminent report
    # is a coin-flip binary event, so it *cuts* confidence rather than adding to it.
    confidence = 0.25 + 0.45 * decay + 0.30 * min(1.0, abs(surprise) / 10.0)
    if growth_score is not None:
        confidence = min(1.0, confidence + 0.05)
    if event_risk:
        confidence *= 0.6
    confidence = float(max(0.0, min(1.0, confidence)))

    # ── Plain-English factors ─────────────────────────────────────────────────
    factors = []
    if surprise >= 0.5:
        factors.append(f"Beat EPS estimates by {surprise:.1f}% on "
                       f"{last_date.date():%b %d, %Y} ({days_since}d ago).")
    elif surprise <= -0.5:
        factors.append(f"Missed EPS estimates by {abs(surprise):.1f}% on "
                       f"{last_date.date():%b %d, %Y} ({days_since}d ago).")
    else:
        factors.append(f"Reported roughly in line with estimates on "
                       f"{last_date.date():%b %d, %Y} ({days_since}d ago).")

    if decay > 0 and abs(surprise) >= 0.5:
        direction = 'higher' if surprise > 0 else 'lower'
        factors.append(f"Post-earnings drift (PEAD) still leans {direction} "
                       f"({decay*100:.0f}% of its ~1-quarter window remaining).")
    elif abs(surprise) >= 0.5:
        factors.append("That surprise is now stale (>1 quarter old) — PEAD spent.")

    if eps_yoy is not None:
        factors.append(f"Reported EPS is {'up' if eps_yoy >= 0 else 'down'} "
                       f"{abs(eps_yoy):.0f}% year-over-year.")

    if event_risk:
        factors.append(f"⚠ Next report in {next_days}d — a binary event; "
                       f"treat any lean as low-confidence into the print.")
    elif next_days is not None:
        factors.append(f"Next report in ~{next_days}d.")

    result = {
        'available':    True,
        'score':        float(max(-1.0, min(1.0, score))),
        'confidence':   confidence,
        'surprise_pct': round(surprise, 2),
        'days_since':   days_since,
        'eps_yoy_pct':  round(eps_yoy, 1) if eps_yoy is not None else None,
        'next_days':    next_days,
        'event_risk':   event_risk,
        'factors':      factors,
    }
    _CACHE[ticker] = (time.time(), result)
    return result
