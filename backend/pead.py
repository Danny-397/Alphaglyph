"""
Post-Earnings-Announcement Drift (PEAD) event study.

PEAD is one of the most durable anomalies in finance: after a company reports,
its stock tends to keep drifting in the direction of the earnings *surprise* for
weeks — good news drifts up, bad news drifts down (Ball & Brown 1968; Bernard &
Thomas 1989). This module tests whether that drift is actually present in a
basket of names, the honest way: as a pooled event study, not a cherry-picked
backtest.

Method:
  1. Collect every earnings event we have (date + EPS surprise%) across the
     basket, from yfinance.
  2. For each event, measure the *market-adjusted* return (stock minus SPY) on
     each of the next `window` trading days, starting the day AFTER the report —
     so we capture the drift, not the announcement-day jump itself.
  3. Cumulate each event into a CAR (cumulative abnormal return) path, then sort
     events into terciles by surprise and average the paths within each tercile.
  4. Report the beat-minus-miss spread at the horizon with a two-sample t-test —
     the drift's effect size and whether it clears the noise.

Honest limitations, surfaced in the payload and UI:
  * Free earnings data is thin; the pooled sample is a few hundred events at
    best, so wide error and an unstable t-stat are expected — the study is a
    demonstration of the method, not a tradable signal.
  * Announcement timestamps are coarse (we drift from the next full session);
    market-adjustment uses a beta-1 market model, not a full factor model.
  * Survivorship: the basket is today's liquid names, which tilts the sample.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from scipy import stats as sstats

import earnings as earn
import features as feat

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM',
                    'JNJ', 'XOM', 'PG', 'HD', 'BAC', 'DIS', 'INTC', 'WMT']

_WINDOW = 20            # trading days of post-announcement drift to track
_MIN_EVENTS = 12        # below this the study isn't worth reporting

_CACHE: dict = {}
_CACHE_TTL = 12 * 3600


def _event_paths(price: pd.DataFrame, mkt_ret: pd.Series,
                 events: list[tuple[pd.Timestamp, float]], window: int):
    """(surprise, CAR-path) for each event with a full post-event window."""
    ar = price['Close'].pct_change().sub(mkt_ret.reindex(price.index)).dropna()
    out = []
    for ed, surp in events:
        pos = ar.index.searchsorted(pd.Timestamp(ed).normalize(), side='right')
        if pos <= 0 or pos + window > len(ar):
            continue
        seg = ar.iloc[pos:pos + window].values
        if np.isnan(seg).any():
            continue
        out.append((surp, np.cumsum(seg)))
    return out


def compute_pead(tickers=None, window: int = _WINDOW) -> dict:
    """Pooled PEAD event study across the basket. Never raises."""
    tickers = tickers or DEFAULT_UNIVERSE
    tickers = [t.upper().strip() for t in tickers][:25]
    cache_key = (tuple(tickers), window)
    hit = _CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    # Market benchmark once (SPY market-model adjustment).
    spy = feat.fetch_ohlcv('SPY', period='5y')
    if spy is None or spy.empty:
        return {'available': False, 'reason': 'benchmark (SPY) unavailable'}
    mkt_ret = spy['Close'].pct_change()

    all_paths, used = [], []
    for t in tickers:
        frame = earn._fetch_earnings_frame(t)
        if frame is None:
            continue
        reported = frame[frame['eps_actual'].notna() & frame['surprise_pct'].notna()]
        if reported.empty:
            continue
        price = feat.fetch_ohlcv(t, period='5y')
        if price is None or price.empty:
            continue
        events = [(d, float(s)) for d, s in reported['surprise_pct'].items()]
        paths = _event_paths(price, mkt_ret, events, window)
        if paths:
            all_paths.extend(paths)
            used.append(t)

    if len(all_paths) < _MIN_EVENTS:
        return {'available': False,
                'reason': f'only {len(all_paths)} events — need ≥{_MIN_EVENTS}',
                'n_events': len(all_paths)}

    surprises = np.array([p[0] for p in all_paths])
    car_mat   = np.vstack([p[1] for p in all_paths])   # [n_events, window]
    terminal  = car_mat[:, -1]

    # ── Terciles by surprise ──────────────────────────────────────────────────
    lo_q, hi_q = np.quantile(surprises, [1 / 3, 2 / 3])
    groups = {
        'miss':   surprises <= lo_q,
        'inline': (surprises > lo_q) & (surprises < hi_q),
        'beat':   surprises >= hi_q,
    }
    labels = {'beat': 'Beat (top ⅓ surprise)', 'inline': 'In line (mid ⅓)',
              'miss': 'Miss (bottom ⅓ surprise)'}

    curves = []
    for key in ('beat', 'inline', 'miss'):
        mask = groups[key]
        if mask.sum() == 0:
            continue
        mean_path = car_mat[mask].mean(axis=0) * 100          # percent
        curves.append({
            'key':          key,
            'label':        labels[key],
            'n':            int(mask.sum()),
            'avg_surprise': round(float(surprises[mask].mean()), 2),
            'car':          [round(float(v), 3) for v in mean_path],
            'terminal_car': round(float(mean_path[-1]), 3),
        })

    # ── Beat-minus-miss drift with a two-sample t-test ────────────────────────
    beat_term = terminal[groups['beat']] * 100
    miss_term = terminal[groups['miss']] * 100
    spread = float(beat_term.mean() - miss_term.mean())
    if len(beat_term) >= 3 and len(miss_term) >= 3:
        tstat, pval = sstats.ttest_ind(beat_term, miss_term, equal_var=False)
        tstat, pval = float(tstat), float(pval)
    else:
        tstat, pval = None, None

    # Correlation of surprise → terminal drift (does more surprise = more drift?).
    if np.std(surprises) > 0:
        corr = float(np.corrcoef(surprises, terminal)[0, 1])
    else:
        corr = None

    result = {
        'available':    True,
        'window':       window,
        'n_events':     len(all_paths),
        'n_tickers':    len(used),
        'tickers':      used,
        'curves':       curves,
        'spread': {
            'beat_minus_miss_pct': round(spread, 3),
            't_stat':              round(tstat, 3) if tstat is not None else None,
            'p_value':             round(pval, 4) if pval is not None else None,
            'significant':         bool(pval is not None and pval < 0.05 and spread > 0),
            'surprise_drift_corr': round(corr, 3) if corr is not None else None,
        },
        'note': (
            'Pooled, market-adjusted CAR by earnings-surprise tercile. A positive '
            'beat-minus-miss spread that clears its t-test is PEAD showing up. With '
            'only a few hundred free-data events, treat this as a demonstration of '
            'the method — the drift is real in the literature but the sample here is '
            'small, the timing coarse, and the basket survivorship-biased.'
        ),
    }
    _CACHE[cache_key] = (time.time(), result)
    return result
