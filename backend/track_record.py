"""
Grading rules for a forward track record, shared by the live ledger (ledger.py)
and the committed daily record (tools/forward_ledger.py).

Pure functions with no database or network access, so both records grade a
call the same way and the rules can be tested on fixed numbers:

  - a call made on day d with horizon h matures on d + h business days;
  - it is graded against the first close on or after that day;
  - BULLISH is right when the price rose, BEARISH when it fell; NEUTRAL calls
    are recorded but never scored;
  - the summary scores only directional calls, and compares the probabilities
    with the base rate (Brier skill), so "always say up in a rising market"
    earns no credit.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def due_date(created: date, horizon: int) -> pd.Timestamp:
    """The trading day the call matures on: created + `horizon` business days."""
    return (pd.Timestamp(created) + pd.tseries.offsets.BDay(max(1, int(horizon)))).normalize()


def grade(entry: float, close: float, direction: str) -> dict:
    """Outcome of one call: realized move, whether it rose, and whether the call was right."""
    realized = (close - entry) / entry * 100.0 if entry else 0.0
    up = 1 if realized > 0 else 0
    if direction == 'BULLISH':
        correct = up
    elif direction == 'BEARISH':
        correct = 1 - up
    else:
        correct = None
    return {'realized_pct': round(realized, 4), 'realized_up': up, 'correct': correct}


def summary(graded: list[dict]) -> dict:
    """Track-record stats over the graded, directional (non-neutral) calls."""
    scored = [g for g in graded if g.get('correct') is not None]
    if not scored:
        return {'n_graded': len(graded), 'n_scored': 0, 'hit_rate': None, 'brier': None,
                'brier_skill': None, 'avg_realized_pct': None,
                'avg_when_bullish': None, 'avg_when_bearish': None}

    correct = np.array([g['correct'] for g in scored], dtype=float)
    p_up = np.array([g['p_up'] for g in scored], dtype=float)
    up = np.array([g['realized_up'] for g in scored], dtype=float)
    realized = np.array([g['realized_pct'] for g in scored], dtype=float)

    base = float(up.mean())                                    # unconditional up-rate
    brier = float(np.mean((p_up - up) ** 2))
    brier_base = float(np.mean((base - up) ** 2))
    bss = (1.0 - brier / brier_base) if brier_base > 1e-9 else None

    bull = realized[[g['direction'] == 'BULLISH' for g in scored]]
    bear = realized[[g['direction'] == 'BEARISH' for g in scored]]

    return {
        'n_graded':         len(graded),
        'n_scored':         len(scored),
        'hit_rate':         round(float(correct.mean()) * 100, 1),
        'base_up_rate':     round(base * 100, 1),
        'brier':            round(brier, 4),
        'brier_skill':      round(bss, 4) if bss is not None else None,
        'avg_realized_pct': round(float(realized.mean()), 3),
        'avg_when_bullish': round(float(bull.mean()), 3) if bull.size else None,
        'avg_when_bearish': round(float(bear.mean()), 3) if bear.size else None,
    }
