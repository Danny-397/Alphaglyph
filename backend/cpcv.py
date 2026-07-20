"""
Combinatorial Purged Cross-Validation (CPCV) + Probability of Backtest
Overfitting (PBO).

The single most important question about any backtest is: *did I overfit?* When
you search a grid of strategy variants and keep the best one, its in-sample
Sharpe is an upward-biased estimate — you selected on noise. PBO quantifies
exactly how bad that bias is.

Method (Bailey, Borwein, López de Prado & Zhu, 2014 — "The Probability of
Backtest Overfitting"), via Combinatorially-Symmetric Cross-Validation (CSCV):

  1. Build a matrix M (T periods × N strategy variants) of per-period returns.
  2. Split the T rows into S contiguous groups (S even). Purge/embargo a few
     rows at every group boundary so serial correlation can't leak the IS edge
     straight into the adjacent OOS block.
  3. For every way of choosing S/2 groups as in-sample (the rest out-of-sample):
       - pick the strategy with the best IS Sharpe (n*),
       - find n*'s *rank* among all strategies OOS,
       - ω = rank / (N+1);  logit λ = ln(ω / (1−ω)).
  4. PBO = fraction of splits where λ ≤ 0 — i.e. the IS champion lands in the
     bottom half OOS. A PBO near 0.5 means selection told you nothing.

Also reported: the IS-vs-OOS performance-degradation regression (how much of the
in-sample Sharpe survives), and the probability the selected strategy actually
loses money OOS.

The strategy grid is a genuine trend/mean-reversion hyperparameter sweep on a
real price series — the exact situation PBO was designed to police.
"""

from __future__ import annotations

import logging
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import features as feat

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_MIN_ROWS     = 400      # need enough history to split into S groups with data
_CACHE: dict  = {}
_CACHE_TTL    = 12 * 3600


# ── Strategy grid → returns matrix ─────────────────────────────────────────────

def _strategy_matrix(prices: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Build a (T × N) matrix of daily strategy returns from a price series.

    N variants span two honest, well-known families whose parameters people
    routinely tune (and over-tune):
      * trend-following  — long when fast SMA > slow SMA, else flat
      * mean-reversion   — long when RSI < buy-threshold, exit when RSI > 55
    Positions are shifted one day (decide on close t, hold over t→t+1) so there
    is no look-ahead.
    """
    close = prices.astype(float)
    ret   = close.pct_change().fillna(0.0).values
    cols: list[np.ndarray] = []
    names: list[str] = []

    # Trend-following grid.
    fasts = [5, 10, 15, 20, 30, 40]
    slows = [50, 80, 100, 150, 200]
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            fast = close.rolling(f).mean()
            slow = close.rolling(s).mean()
            pos  = (fast > slow).astype(float).shift(1).fillna(0.0).values
            cols.append(pos * ret)
            names.append(f'MA {f}/{s}')

    # Mean-reversion grid (RSI).
    for period in (7, 14, 21):
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = (100 - 100 / (1 + rs)).fillna(50.0)
        for buy in (25, 30, 35, 40):
            # Stateful long/flat: enter under `buy`, exit over 55.
            long = np.zeros(len(close))
            holding = False
            rvals = rsi.values
            for i in range(len(close)):
                if not holding and rvals[i] < buy:
                    holding = True
                elif holding and rvals[i] > 55:
                    holding = False
                long[i] = 1.0 if holding else 0.0
            pos = pd.Series(long, index=close.index).shift(1).fillna(0.0).values
            cols.append(pos * ret)
            names.append(f'RSI{period}<{buy}')

    M = np.column_stack(cols)
    return M, names


def _sharpe(mat: np.ndarray) -> np.ndarray:
    """Annualised Sharpe per column (columns with ~zero variance → -inf so they
    are never selected as the IS champion)."""
    mu  = mat.mean(axis=0)
    sd  = mat.std(axis=0, ddof=1)
    out = np.full(mat.shape[1], -np.inf)
    ok  = sd > 1e-12
    out[ok] = mu[ok] / sd[ok] * np.sqrt(_TRADING_DAYS)
    return out


# ── CSCV / PBO core ────────────────────────────────────────────────────────────

def compute_cpcv(ticker: str, n_groups: int = 10, embargo: int = 5,
                 period: str = '5y') -> dict:
    """Run CSCV on a real hyperparameter grid for `ticker` and return PBO plus the
    supporting diagnostics the frontend visualises."""
    ticker = (ticker or 'SPY').upper().strip()
    key = (ticker, n_groups, embargo, period)
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    if n_groups % 2 != 0:
        n_groups += 1
    n_groups = max(6, min(n_groups, 12))       # C(12,6)=924 combos is the ceiling

    df = feat.fetch_ohlcv(ticker, period=period)
    if df is None or len(df) < _MIN_ROWS:
        return {'available': False,
                'reason': f'Need ≥ {_MIN_ROWS} trading days of history for {ticker}.'}

    M, names = _strategy_matrix(df['Close'])
    # Drop the indicator warm-up rows (leading zeros from the longest window).
    warm = 205
    M = M[warm:]
    T, N = M.shape
    if T < n_groups * 20:
        return {'available': False, 'reason': 'Not enough usable rows after warm-up.'}

    # Contiguous groups of near-equal size.
    bounds = np.linspace(0, T, n_groups + 1).astype(int)
    groups = [(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    def _rows(group_ids: tuple[int, ...], *, embargoed: bool) -> np.ndarray:
        """Concatenated row-indices for the given groups. When `embargoed`, trim
        `embargo` rows at each end of every block (the IS side) so the adjacent
        OOS block can't borrow serially-correlated edge across the seam."""
        idx = []
        for g in group_ids:
            lo, hi = groups[g]
            if embargoed and (hi - lo) > 2 * embargo:
                lo, hi = lo + embargo, hi - embargo
            idx.append(np.arange(lo, hi))
        return np.concatenate(idx) if idx else np.array([], dtype=int)

    all_ids  = list(range(n_groups))
    half     = n_groups // 2
    combos   = list(combinations(all_ids, half))

    logits, is_best, oos_best = [], [], []
    for is_ids in combos:
        oos_ids = tuple(g for g in all_ids if g not in is_ids)
        is_rows  = _rows(is_ids,  embargoed=True)
        oos_rows = _rows(oos_ids, embargoed=False)
        if len(is_rows) < N or len(oos_rows) < 5:
            continue

        is_sr  = _sharpe(M[is_rows])
        oos_sr = _sharpe(M[oos_rows])
        n_star = int(np.argmax(is_sr))

        # Fractional rank of the IS champion among all strategies OOS.
        ranks = rankdata(oos_sr)                       # 1 = worst, N = best
        omega = ranks[n_star] / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(omega / (1 - omega))))
        is_best.append(float(is_sr[n_star]))
        oos_best.append(float(oos_sr[n_star]))

    if not logits:
        return {'available': False, 'reason': 'No valid CSCV splits produced.'}

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0))

    # A strategy that simply stays flat over an OOS block has zero variance and
    # gets a -inf Sharpe; that's a degenerate "no-trade" outcome, not a real
    # performance figure, so drop those before the regression / medians / scatter.
    is_arr, oos_arr = np.array(is_best), np.array(oos_best)
    finite = np.isfinite(is_arr) & np.isfinite(oos_arr)
    x, y = is_arr[finite], oos_arr[finite]

    # Performance degradation: OOS_best ~ a + b·IS_best. b << 1 means the
    # in-sample edge mostly evaporates out of sample.
    if x.size >= 2 and x.std() > 1e-9:
        b, a = np.polyfit(x, y, 1)
    else:
        b, a = 0.0, (float(y.mean()) if y.size else 0.0)

    # OOS-loss rate over the non-degenerate splits only.
    prob_oos_loss = float(np.mean(y < 0)) if y.size else 0.0

    # Logit histogram for the reliability plot.
    lo, hi = float(logits.min()), float(logits.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    counts, edges = np.histogram(logits, bins=20, range=(lo, hi))
    hist = [{'x': round((edges[i] + edges[i + 1]) / 2, 3), 'count': int(counts[i])}
            for i in range(len(counts))]

    # A compact diagram of the first few splits for the CPCV grid visual.
    diagram = []
    for is_ids in combos[:8]:
        diagram.append(['IS' if g in is_ids else 'OOS' for g in all_ids])

    result = {
        'available':      True,
        'ticker':         ticker,
        'n_strategies':   N,
        'n_groups':       n_groups,
        'n_combos':       len(logits),
        'n_rows':         int(T),
        'embargo':        embargo,
        'pbo':            round(pbo, 4),
        'prob_oos_loss':  round(prob_oos_loss, 4),
        'degradation': {
            'slope':          round(float(b), 4),
            'intercept':      round(float(a), 4),
            'median_is_sr':   round(float(np.median(x)), 3) if x.size else None,
            'median_oos_sr':  round(float(np.median(y)), 3) if y.size else None,
        },
        'logit_hist':     hist,
        'scatter':        [{'is': round(float(xi), 3), 'oos': round(float(yi), 3)}
                           for xi, yi in zip(x, y)],
        'splits_diagram': diagram,
        'verdict':        _verdict(pbo, float(np.median(x)) if x.size else 0.0,
                                   float(np.median(y)) if y.size else 0.0,
                                   prob_oos_loss),
        'generated_at':   pd.Timestamp.utcnow().isoformat(),
    }
    _CACHE[key] = (time.time(), result)
    return result


def _verdict(pbo: float, median_is: float, median_oos: float, prob_loss: float) -> str:
    if pbo >= 0.5:
        head = (f'PBO = {pbo*100:.0f}%. The in-sample winner lands in the bottom '
                'half out-of-sample more often than not — selection here is '
                'essentially picking noise.')
    elif pbo >= 0.25:
        head = (f'PBO = {pbo*100:.0f}%. Real but material overfitting risk: the '
                'best-looking variant frequently disappoints out-of-sample.')
    else:
        head = (f'PBO = {pbo*100:.0f}%. Low overfitting probability — the '
                'in-sample ranking tends to hold up out-of-sample.')
    # How much of the champion's in-sample Sharpe survives, at the median split.
    retention = (median_oos / median_is * 100) if median_is > 1e-6 else 0.0
    retention = max(0.0, min(retention, 100.0))
    tail = (f' The chosen strategy carries a median in-sample Sharpe of '
            f'{median_is:.2f} but only {median_oos:.2f} out-of-sample '
            f'(~{retention:.0f}% survives), and loses money OOS '
            f'{prob_loss*100:.0f}% of the time.')
    return head + tail
