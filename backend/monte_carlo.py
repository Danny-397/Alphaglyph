"""
Monte Carlo simulation for backtest result validation.

Takes the daily return sequence from a completed backtest, resamples it
1,000 times, and asks: where does the actual result sit in the distribution
of random paths?

If your strategy's Sharpe of 1.4 ranks in the 92nd percentile of 1,000
random shuffles, that's statistically meaningful.  If it ranks in the 53rd
percentile, the strategy is essentially indistinguishable from random.

Bootstrap method
----------------
The default is a **stationary block bootstrap** (Politis & Romano, 1994),
not the naive i.i.d. resample.  A plain i.i.d. bootstrap draws each day
independently, which destroys the serial correlation real return series have
(volatility clusters, short-term momentum/mean-reversion) and therefore
*understates* the variance of the paths — flattering the strategy's
percentile rank.  The stationary bootstrap instead stitches together blocks
of consecutive days whose lengths are geometrically distributed (expected
length ≈ n^(1/3)), so each resampled path preserves the local autocorrelation
structure while the series as a whole stays stationary.  Pass method='iid'
to recover the classic independent resample for comparison.

Returns
-------
A dict containing:
  actual_return_pct     — the strategy's actual total return
  actual_percentile     — percentile rank vs simulated paths  (0–100)
  sharpe_percentile     — percentile rank of actual Sharpe vs simulated
  return_distribution   — {p5, p25, p50, p75, p95} of simulated final returns
  sharpe_distribution   — {p5, p25, p50, p75, p95} of simulated Sharpe ratios
  fan_chart             — {dates, p5, p25, p50, p75, p95} sampled equity bands

When the ML course is done
--------------------------
Nothing in this module needs to change.  Run it on ML strategy backtests
exactly as you do for rule-based ones — the percentile rank tells you
whether the ML results are statistically meaningful or just overfit noise.
"""

from __future__ import annotations

import numpy as np


def _default_block_len(n: int) -> int:
    """Expected block length for the stationary bootstrap.

    The n^(1/3) rule is the standard order for the optimal block length of a
    stationary series; clamped to [2, n] so short backtests still get real
    blocks without the length exceeding the sample.
    """
    return int(min(n, max(2, round(n ** (1.0 / 3.0)))))


def _stationary_bootstrap(returns: np.ndarray, n_simulations: int,
                          avg_block_len: int) -> np.ndarray:
    """Stationary block bootstrap (Politis & Romano, 1994).

    Returns an (n_simulations, n) array of resampled daily returns in which
    each path is built from wrap-around blocks of consecutive days whose
    lengths are geometrically distributed with mean ``avg_block_len`` — so the
    local autocorrelation of the original series is preserved.
    """
    n = len(returns)
    p = 1.0 / avg_block_len                 # per-step probability of a new block
    idx = np.empty((n_simulations, n), dtype=np.intp)
    idx[:, 0] = np.random.randint(0, n, size=n_simulations)
    new_block   = np.random.random((n_simulations, n)) < p
    fresh_start = np.random.randint(0, n, size=(n_simulations, n))
    for t in range(1, n):
        # Continue the current block (next day, wrapping) unless a new block starts.
        cont = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(new_block[:, t], fresh_start[:, t], cont)
    return returns[idx]


def run_simulation(
    port_hist: list[dict],
    initial_capital: float,
    actual_sharpe: float,
    n_simulations: int = 1000,
    method: str = 'stationary',
    avg_block_len: int | None = None,
) -> dict:
    """
    Bootstrap resample the daily return sequence n_simulations times.

    Parameters
    ----------
    port_hist       : list of {date, value} dicts from the backtest
    initial_capital : starting cash used in the backtest
    actual_sharpe   : Sharpe ratio the strategy actually achieved
    n_simulations   : number of random paths (default 1000)
    method          : 'stationary' (default, block bootstrap — preserves
                      autocorrelation) or 'iid' (classic independent resample)
    avg_block_len   : expected block length for the stationary bootstrap;
                      defaults to ~n^(1/3) (ignored when method='iid')

    Returns an 'enabled: False' dict if there are fewer than 5 data points.
    """
    if len(port_hist) < 5:
        return {'enabled': False}

    values = np.array([p['value'] for p in port_hist], dtype=float)
    dates  = [p['date'] for p in port_hist]

    # Daily returns (one fewer point than equity curve)
    returns = np.diff(values) / values[:-1]
    n       = len(returns)

    actual_final  = float(values[-1])
    actual_return = (actual_final / initial_capital - 1) * 100

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    # No fixed seed — slight variation across runs is expected and correct.
    # Each run gives slightly different percentile estimates, which demonstrates
    # that the conclusion is stable, not an artifact of a particular sample.
    # Stationary block bootstrap by default so paths keep the serial correlation
    # (volatility clustering, momentum) that an i.i.d. resample would erase.
    block_len = avg_block_len or _default_block_len(n)
    if method == 'iid':
        sim_returns = np.random.choice(returns, size=(n_simulations, n), replace=True)
    else:
        method = 'stationary'
        sim_returns = _stationary_bootstrap(returns, n_simulations, block_len)

    # Equity curves: shape (n_simulations, n)
    # Each row is one simulated path starting from initial_capital
    equity_matrix = initial_capital * np.cumprod(1 + sim_returns, axis=1)
    final_values  = equity_matrix[:, -1]
    final_returns = (final_values / initial_capital - 1) * 100

    # ── Simulated Sharpe ratios ────────────────────────────────────────────────
    stds        = sim_returns.std(axis=1)
    means       = sim_returns.mean(axis=1)
    rf_daily    = 0.04 / 252
    # A resampled path can have zero volatility (e.g. every draw was the same
    # return), which would make the Sharpe undefined. Divide only where std > 0 —
    # np.where alone isn't enough because it still evaluates (and warns on) the
    # 1/0 in the discarded branch before masking it.
    sim_sharpes = np.zeros_like(stds)
    np.divide((means - rf_daily) * np.sqrt(252), stds,
              out=sim_sharpes, where=stds > 0)

    # ── Percentile ranks of the actual results ─────────────────────────────────
    actual_pct = float(np.mean(final_values <= actual_final) * 100)
    sharpe_pct = float(np.mean(sim_sharpes <= actual_sharpe) * 100)

    # ── Fan chart bands ────────────────────────────────────────────────────────
    # Sample ~60 time points to keep the JSON payload small
    step      = max(1, n // 60)
    idx       = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    sampled   = equity_matrix[:, idx]
    # equity_matrix[:, i] = portfolio after the i-th daily return
    # that corresponds to port_hist[i+1], so shift dates by 1
    fan_dates = [dates[i + 1] for i in idx]

    def _band(pct):
        return [round(float(v), 2) for v in np.percentile(sampled, pct, axis=0)]

    return {
        'enabled':             True,
        'n_simulations':       n_simulations,
        'bootstrap_method':    method,
        'avg_block_len':       block_len if method == 'stationary' else 1,
        'is_skill_test':       False,   # this is an outcome-spread view, not a skill test
        'actual_return_pct':   round(actual_return, 2),
        'actual_percentile':   round(actual_pct,    1),
        'sharpe_percentile':   round(sharpe_pct,    1),
        'return_distribution': {
            'p5':  round(float(np.percentile(final_returns,  5)), 2),
            'p25': round(float(np.percentile(final_returns, 25)), 2),
            'p50': round(float(np.percentile(final_returns, 50)), 2),
            'p75': round(float(np.percentile(final_returns, 75)), 2),
            'p95': round(float(np.percentile(final_returns, 95)), 2),
        },
        'sharpe_distribution': {
            'p5':  round(float(np.percentile(sim_sharpes,  5)), 2),
            'p25': round(float(np.percentile(sim_sharpes, 25)), 2),
            'p50': round(float(np.percentile(sim_sharpes, 50)), 2),
            'p75': round(float(np.percentile(sim_sharpes, 75)), 2),
            'p95': round(float(np.percentile(sim_sharpes, 95)), 2),
        },
        'fan_chart': {
            'dates': fan_dates,
            'p5':    _band(5),
            'p25':   _band(25),
            'p50':   _band(50),
            'p75':   _band(75),
            'p95':   _band(95),
        },
    }


# ── Skill test: random-timing permutation ──────────────────────────────────────
# Why this exists (and why the bootstrap above is NOT a skill test):
# The bootstrap resamples the strategy's *own* daily returns, so the actual
# compound return and Sharpe sit at ~the 50th percentile of the resampled
# distribution *by construction* (the resampled mean equals the sample mean).
# That makes the fan chart a good picture of the *range of outcomes*, but it
# cannot tell you whether the strategy has genuine skill — its percentile is
# ~50 for almost any strategy. To actually ask "did the timing beat random?"
# you need a null with real structure, which is what this test provides.


def random_timing_test(
    strategy_sharpe: float,
    benchmark_returns,
    strategy_returns,
    n_simulations: int = 1000,
) -> dict:
    """
    Permutation test: does the strategy's Sharpe beat *random market timing*?

    The null model is a "monkey" trader that, on a random subset of days, is
    long the market benchmark (SPY) and is otherwise in cash — sized so it is
    invested on the same fraction of days as the real strategy (its exposure).
    We build n_simulations such random-timing paths, compute each one's Sharpe
    the same way the backtest does, and report the percentile rank of the
    strategy's Sharpe in that distribution.

        skill_percentile ≈ 50  →  no better than randomly timing the market
        skill_percentile ≥ 95  →  timing beat ~95% of random schedules (p ≲ 0.05)

    Honest limitation: the null trades the benchmark, not the strategy's exact
    universe, so it blends "market-timing skill" with "asset selection". It is
    a genuine, non-degenerate null (unlike the self-resample) — not a perfect
    attribution. Returns {'enabled': False, ...} when inputs are too short.
    """
    bm = np.asarray(benchmark_returns, dtype=float)
    sr = np.asarray(strategy_returns, dtype=float)
    n  = len(bm)
    if n < 20 or not np.isfinite(strategy_sharpe):
        return {'enabled': False, 'reason': 'Need 20+ benchmark days for a skill test.'}

    # Exposure proxy: fraction of days the strategy was actually invested.
    # A fully-in-cash day has an exactly-zero portfolio return; any open
    # position makes the day's return non-zero. Clamp away from 0 and 1.
    if len(sr):
        exposure = float(np.mean(np.abs(sr) > 1e-9))
    else:
        exposure = 0.5
    exposure = min(max(exposure, 1.0 / n), 1.0)
    k = max(1, min(n, int(round(exposure * n))))

    rf_daily = 0.04 / 252

    # Vectorised: each row picks k distinct in-market days via argsort of noise.
    order   = np.random.random((n_simulations, n)).argsort(axis=1)
    in_mkt  = order < k                                  # exactly k True per row
    sim     = np.where(in_mkt, bm[None, :], 0.0)         # cash days earn 0

    means = sim.mean(axis=1)
    stds  = sim.std(axis=1)
    sharpes = np.zeros(n_simulations)
    np.divide((means - rf_daily) * np.sqrt(252), stds, out=sharpes, where=stds > 0)

    pct = float(np.mean(sharpes <= strategy_sharpe) * 100)

    return {
        'enabled':            True,
        'skill_percentile':   round(pct, 1),
        'strategy_sharpe':    round(float(strategy_sharpe), 3),
        'exposure_pct':       round(exposure * 100, 1),
        'n_simulations':      n_simulations,
        'null_sharpe_median': round(float(np.percentile(sharpes, 50)), 3),
        'null_sharpe_p95':    round(float(np.percentile(sharpes, 95)), 3),
        'is_significant':     bool(pct >= 95.0),
    }
