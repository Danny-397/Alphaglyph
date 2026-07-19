"""
Markowitz Mean-Variance Portfolio Optimization.

Solves two convex quadratic programs using SLSQP:

    Max-Sharpe (tangency portfolio)
        max   (w^T μ - r_f) / sqrt(w^T Σ w)
        s.t.  sum(w) = 1,  w_i >= 0

    Min-Variance
        min   w^T Σ w
        s.t.  sum(w) = 1,  w_i >= 0

The efficient frontier is traced by sweeping target returns between the
minimum-variance return and the maximum achievable return, solving the
min-variance problem at each target level.

All annualisation uses 252 trading days.
Public entry point: compute_efficient_frontier()
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import features as feat

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.04   # annualised — matches Sharpe calc in backtest.py
TRADING_DAYS   = 252


# ── Math primitives ───────────────────────────────────────────────────────────

def _stats(weights: np.ndarray, mu: np.ndarray,
           cov: np.ndarray) -> tuple[float, float, float]:
    """Return (annualised_return, annualised_vol, sharpe) for a weight vector."""
    ret    = float(weights @ mu * TRADING_DAYS)
    var    = float(weights @ cov @ weights * TRADING_DAYS)
    vol    = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


def _minimize(objective, mu: np.ndarray, cov: np.ndarray,
              extra_constraints: list | None = None):
    """Run SLSQP with long-only + fully-invested constraints."""
    n           = len(mu)
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    return minimize(
        objective, np.ones(n) / n,
        args=(mu, cov),
        method='SLSQP',
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-10},
    )


# ── Data layer ────────────────────────────────────────────────────────────────

def build_return_matrix(tickers: list[str],
                        start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV for each ticker and return a aligned DataFrame of daily
    returns (columns = tickers, rows = dates, no NaN rows).
    Returns None when fewer than 2 tickers or fewer than 30 common dates.
    """
    series = {}
    for ticker in tickers:
        raw = feat.fetch_ohlcv(ticker, start=start, end=end)
        if raw is not None and len(raw) > 30:
            series[ticker] = raw['Close'].pct_change().dropna()
    if len(series) < 2:
        return None
    df = pd.DataFrame(series).dropna()
    return df if len(df) >= 30 else None


# ── Main entry point ──────────────────────────────────────────────────────────

def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) shrinkage of the sample covariance toward a scaled
    identity — "Honey, I shrunk the sample covariance matrix".

    The sample covariance is a noisy estimate: with N assets it has N(N+1)/2
    free numbers estimated from limited data, so its extreme eigenvalues are
    biased (too large and too small). That noise is exactly what a Markowitz
    optimizer exploits, over-concentrating in whatever pair happened to look
    least correlated. Ledoit-Wolf pulls the estimate toward a well-conditioned
    target F = μI (μ = average variance) by the analytically optimal amount δ:

        Σ_shrunk = δ·μI + (1 − δ)·S

    δ ∈ [0, 1] is estimated from the data (no free knob): more noise → more
    shrinkage. Returns (shrunk_cov_daily, δ).
    """
    Y = returns.values
    Y = Y - Y.mean(axis=0)
    T, p = Y.shape
    S = (Y.T @ Y) / T                                   # sample cov (1/T)
    m = np.trace(S) / p                                 # ⟨S, I⟩ = mean variance

    # d² = ‖S − mI‖²  (normalised Frobenius, ⟨A,B⟩ = tr(AB')/p)
    d2 = float(np.sum((S - m * np.eye(p)) ** 2) / p)
    if d2 <= 0:
        return S, 0.0

    # b̄² = (1/T²) Σ_k ‖x_k x_kᵀ − S‖², expanded to a vectorised form.
    sq = np.sum(Y ** 2, axis=1)                         # ‖x_k‖² per observation
    sum_x4  = float(np.sum(sq ** 2))                    # Σ ‖x_k‖⁴
    sum_xSx = float(np.einsum('ti,ij,tj->', Y, S, Y))  # Σ x_kᵀ S x_k
    S_fro2  = float(np.sum(S ** 2))
    b2bar = (sum_x4 - 2 * sum_xSx + T * S_fro2) / (T ** 2 * p)
    b2 = min(b2bar, d2)                                 # bounded by d²

    delta = float(max(0.0, min(1.0, b2 / d2)))
    shrunk = delta * m * np.eye(p) + (1 - delta) * S
    return shrunk, delta


def compute_efficient_frontier(
    tickers: list[str],
    start_date: str,
    end_date: str,
    n_points: int = 60,
    shrink: bool = False,
) -> dict:
    """
    Compute the efficient frontier for the given tickers and date range.

    Returns a dict with:
        tickers            — tickers actually used (some may be dropped for insufficient data)
        max_sharpe         — tangency portfolio: weights, return, vol, Sharpe
        min_variance       — global minimum variance portfolio
        efficient_frontier — list of {return, volatility, sharpe} along the frontier
        individual_assets  — standalone return/vol/Sharpe for each ticker
        covariance_matrix  — annualised covariance {tickers, data}

    On failure returns {'error': '<reason>'}.
    """
    returns = build_return_matrix(tickers, start_date, end_date)
    if returns is None:
        return {'error': 'Need at least 2 tickers with 30+ days of overlapping data.'}

    used = list(returns.columns)
    n    = len(used)
    mu   = returns.mean().values
    if shrink:
        cov, shrink_delta = ledoit_wolf_cov(returns)
    else:
        cov, shrink_delta = returns.cov().values, None

    # ── Max-Sharpe (tangency) portfolio ───────────────────────────────────────
    ms     = _minimize(lambda w, m, c: -_stats(w, m, c)[2], mu, cov)
    ms_w   = ms.x
    ms_ret, ms_vol, ms_sr = _stats(ms_w, mu, cov)

    # ── Global minimum-variance portfolio ─────────────────────────────────────
    mv     = _minimize(lambda w, m, c: _stats(w, m, c)[1], mu, cov)
    mv_w   = mv.x
    mv_ret, mv_vol, mv_sr = _stats(mv_w, mu, cov)

    # ── Efficient frontier: sweep target returns ──────────────────────────────
    r_min    = mv_ret
    r_max    = float(mu.max() * TRADING_DAYS)
    frontier = []
    for target in np.linspace(r_min, r_max, n_points):
        extra = [{'type': 'eq',
                  'fun': lambda w, t=target: float(w @ mu * TRADING_DAYS) - t}]
        res = _minimize(lambda w, m, c: _stats(w, m, c)[1], mu, cov, extra)
        if res.success:
            r, v, s = _stats(res.x, mu, cov)
            frontier.append({
                'return':     round(r * 100, 2),
                'volatility': round(v * 100, 2),
                'sharpe':     round(s, 3),
            })

    # ── Individual asset stats (for scatter overlay on the frontier chart) ────
    individual = {}
    for i, ticker in enumerate(used):
        w_i    = np.zeros(n)
        w_i[i] = 1.0
        r, v, s = _stats(w_i, mu, cov)
        individual[ticker] = {
            'expected_return': round(r * 100, 2),
            'volatility':      round(v * 100, 2),
            'sharpe_ratio':    round(s, 3),
        }

    # ── Annualised covariance matrix ──────────────────────────────────────────
    ann_cov = (cov * TRADING_DAYS).tolist()

    # Concentration (Herfindahl) of the max-Sharpe weights — shrinkage should
    # visibly reduce this, the whole practical point of the estimator.
    hhi = float(np.sum(ms_w ** 2))

    return {
        'tickers': used,
        'shrinkage': None if shrink_delta is None else {
            'method': 'ledoit_wolf',
            'delta':  round(shrink_delta, 4),
            'hhi':    round(hhi, 4),
        },
        'max_sharpe': {
            'weights':         {t: round(float(w), 4) for t, w in zip(used, ms_w)},
            'expected_return': round(ms_ret * 100, 2),
            'volatility':      round(ms_vol * 100, 2),
            'sharpe_ratio':    round(ms_sr,  3),
        },
        'min_variance': {
            'weights':         {t: round(float(w), 4) for t, w in zip(used, mv_w)},
            'expected_return': round(mv_ret * 100, 2),
            'volatility':      round(mv_vol * 100, 2),
            'sharpe_ratio':    round(mv_sr,  3),
        },
        'efficient_frontier': frontier,
        'individual_assets':  individual,
        'covariance_matrix': {
            'tickers': used,
            'data':    [[round(ann_cov[i][j], 6) for j in range(n)]
                        for i in range(n)],
        },
    }
