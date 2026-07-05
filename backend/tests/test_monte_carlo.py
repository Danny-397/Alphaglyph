"""Tests for the Monte Carlo bootstrap resampler (monte_carlo.run_simulation)."""

import warnings

import numpy as np

import monte_carlo


def _equity_curve(returns, initial=100_000.0):
    """Turn a daily-return list into the {date, value} port_hist the sim expects."""
    values = [initial]
    for r in returns:
        values.append(values[-1] * (1 + r))
    return [{'date': f'2023-01-{i + 1:02d}', 'value': v} for i, v in enumerate(values)]


def test_returns_bands_and_percentiles():
    rng = np.random.default_rng(0)
    port_hist = _equity_curve(rng.normal(0.001, 0.02, 40))
    out = monte_carlo.run_simulation(port_hist, 100_000.0, actual_sharpe=1.0,
                                     n_simulations=500)
    assert out['enabled'] is True
    # Percentile ranks are valid percentages
    assert 0.0 <= out['sharpe_percentile'] <= 100.0
    assert 0.0 <= out['actual_percentile'] <= 100.0


def test_too_few_points_disabled():
    out = monte_carlo.run_simulation(_equity_curve([0.01, 0.02]), 100_000.0, 1.0)
    assert out['enabled'] is False


def test_zero_volatility_path_is_warning_free_and_finite():
    """A constant-return curve makes some bootstrap paths have zero volatility.

    The Sharpe computation must not raise a divide-by-zero warning, and every
    simulated Sharpe must stay finite (the undefined ones are defined to be 0).
    """
    # Every daily return identical → many resampled paths have std == 0.
    port_hist = _equity_curve([0.01] * 30)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any RuntimeWarning becomes a failure
        out = monte_carlo.run_simulation(port_hist, 100_000.0, actual_sharpe=0.0,
                                         n_simulations=200)
    assert out['enabled'] is True
    assert np.isfinite(out['sharpe_percentile'])
    assert np.isfinite(out['actual_percentile'])
    # Undefined (zero-vol) Sharpes are defined to be 0 — the distribution stays finite.
    assert all(np.isfinite(v) for v in out['sharpe_distribution'].values())
