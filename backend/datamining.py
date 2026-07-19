"""
The Data-Mining Lab — a live demonstration of how backtesting fools you.

This is the story in AlphaGlyph's README, made interactive: search over enough
strategies and the *best* one will look brilliant purely by luck. The lab makes
that undeniable.

It generates N random long/flat "timing" strategies on a real ticker — each goes
long on a random ~half of the days, so none has any genuine edge — backtests all
of them, and keeps the single best Sharpe. Then it shows the two verdicts side by
side:

    Naive PSR (pretends the winner was the only thing you tried)
        → often says "significant". This is the trap.

    Deflated Sharpe Ratio (knows you tried N and corrects for it, López de Prado
    2014) → almost always says "not significant".

The gap between those two numbers is the whole lesson: a great backtested Sharpe
is meaningless without knowing how many were tried to find it. The expected
maximum Sharpe of N pure-noise strategies (the DSR benchmark) is drawn on the
histogram so the user can see the winner sitting right where chance predicts.
"""

from __future__ import annotations

import logging

import numpy as np

import features as feat
import stats

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_EXPOSURE = 0.5          # each random strategy is long ~half the days
_MIN_N, _MAX_N = 20, 2000


def run_sweep(ticker: str, n_strategies: int = 200, seed: int | None = None) -> dict:
    """
    Sweep n random timing strategies on `ticker`, keep the best in-sample Sharpe,
    and contrast its naive PSR with its Deflated Sharpe. Never raises.
    """
    ticker = (ticker or '').upper().strip()
    n = int(np.clip(n_strategies, _MIN_N, _MAX_N))

    df = feat.get_feature_df(ticker, period='5y')
    if df is None or len(df) < 250:
        return {'available': False, 'reason': f'not enough history for {ticker}'}

    rets = df['Close'].pct_change().dropna().values
    T = len(rets)
    rng = np.random.default_rng(seed)

    # N random long/flat masks in one shot: [N, T] of {0,1}, ~half long.
    masks = (rng.random((n, T)) < _EXPOSURE).astype(np.float64)
    strat_rets = masks * rets                                   # long or flat each day

    mu    = strat_rets.mean(axis=1)
    sigma = strat_rets.std(axis=1, ddof=1)
    good  = sigma > 1e-12
    sharpes = np.full(n, np.nan)
    sharpes[good] = mu[good] / sigma[good] * np.sqrt(_TRADING_DAYS)
    sharpes_valid = sharpes[np.isfinite(sharpes)]
    if sharpes_valid.size == 0:
        return {'available': False, 'reason': 'degenerate sweep'}

    best_i = int(np.nanargmax(sharpes))
    best_sharpe = float(sharpes[best_i])
    best_ret = strat_rets[best_i]

    # Buy-and-hold Sharpe for a reference marker.
    bh_sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(_TRADING_DAYS)) if rets.std() > 0 else 0.0

    naive_psr = stats.probabilistic_sharpe_ratio(best_ret, sr_benchmark_annual=0.0)
    dsr = stats.deflated_sharpe_ratio(best_ret, n_strategies=n)

    # ── Histogram of the N Sharpes ────────────────────────────────────────────
    lo, hi = float(np.min(sharpes_valid)), float(np.max(sharpes_valid))
    n_bins = 24
    counts, edges = np.histogram(sharpes_valid, bins=n_bins, range=(lo, hi))
    hist = [{'x': round(float((edges[i] + edges[i + 1]) / 2), 3), 'count': int(counts[i])}
            for i in range(n_bins)]

    naive_sig = bool(np.isfinite(naive_psr) and naive_psr > 0.95)
    dsr_sig = bool(dsr.get('is_significant'))

    return {
        'available':      True,
        'ticker':         ticker,
        'n_strategies':   n,
        'n_days':         T,
        'best_sharpe':    round(best_sharpe, 3),
        'median_sharpe':  round(float(np.median(sharpes_valid)), 3),
        'buy_hold_sharpe': round(bh_sharpe, 3),
        'expected_max_sharpe': dsr.get('sr_benchmark'),        # DSR's SR* benchmark
        'naive_psr':      round(float(naive_psr), 4) if np.isfinite(naive_psr) else None,
        'naive_significant': naive_sig,
        'deflated_sharpe':   dsr.get('dsr'),
        'deflated_significant': dsr_sig,
        'histogram':      hist,
        'verdict': (
            f"Searching {n} strategies of pure noise, the luckiest scored a Sharpe of "
            f"{best_sharpe:.2f}. Naive statistics call that "
            f"{'significant — the trap' if naive_sig else 'borderline'}; the Deflated "
            f"Sharpe, which knows {n} were tried, says "
            f"{'it is real' if dsr_sig else 'it is just the best of many coin flips'}."
        ),
        'note': (
            'Every strategy here is random by construction — none has any edge. The '
            'best Sharpe is therefore 100% luck, yet it can look impressive. That is '
            'why AlphaGlyph reports the Deflated Sharpe on real backtests: a number is '
            'only as meaningful as the number of tries it took to find it.'
        ),
    }
