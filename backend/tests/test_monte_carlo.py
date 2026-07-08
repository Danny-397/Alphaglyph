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


def test_stationary_bootstrap_is_the_default():
    """The default resample is the stationary block bootstrap, and it reports
    the block length it used."""
    rng = np.random.default_rng(1)
    port_hist = _equity_curve(rng.normal(0.001, 0.02, 90))
    out = monte_carlo.run_simulation(port_hist, 100_000.0, actual_sharpe=1.0,
                                     n_simulations=300)
    assert out['bootstrap_method'] == 'stationary'
    assert out['avg_block_len'] >= 2


def test_iid_method_selectable_and_labelled():
    rng = np.random.default_rng(2)
    port_hist = _equity_curve(rng.normal(0.001, 0.02, 40))
    out = monte_carlo.run_simulation(port_hist, 100_000.0, actual_sharpe=1.0,
                                     n_simulations=300, method='iid')
    assert out['bootstrap_method'] == 'iid'
    assert out['avg_block_len'] == 1
    assert 0.0 <= out['sharpe_percentile'] <= 100.0


def test_block_bootstrap_preserves_autocorrelation():
    """A strongly autocorrelated series should keep more of its lag-1
    autocorrelation under the block bootstrap than under the i.i.d. resample."""
    rng = np.random.default_rng(3)
    # AR(1) return series with high persistence.
    n = 400
    eps = rng.normal(0, 0.01, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.6 * r[i - 1] + eps[i]
    port_hist = _equity_curve(r.tolist())

    def _mean_abs_lag1_autocorr(sim):
        a = sim[:, :-1] - sim.mean(axis=1, keepdims=True)
        b = sim[:, 1:] - sim.mean(axis=1, keepdims=True)
        num = (a * b).sum(axis=1)
        var = (a * a).sum(axis=1)
        return float(np.mean(np.abs(num / np.where(var > 0, var, np.nan))))

    ret = np.diff([p['value'] for p in port_hist]) / \
        np.array([p['value'] for p in port_hist])[:-1]
    np.random.seed(0)
    block = monte_carlo._stationary_bootstrap(ret, 400, monte_carlo._default_block_len(len(ret)))
    iid = np.random.choice(ret, size=(400, len(ret)), replace=True)
    assert _mean_abs_lag1_autocorr(block) > _mean_abs_lag1_autocorr(iid)


def test_skill_test_ranks_high_sharpe_high_and_low_sharpe_low():
    """The random-timing permutation test must place an implausibly good Sharpe
    near the top of the null distribution and a poor Sharpe near the bottom."""
    rng = np.random.default_rng(7)
    bench = rng.normal(0.0004, 0.012, 250)          # ~1yr of market days
    strat_rets = rng.normal(0.0004, 0.012, 250)     # exposure proxy ~100%
    np.random.seed(0)
    good = monte_carlo.random_timing_test(5.0, bench, strat_rets, n_simulations=500)
    np.random.seed(0)
    bad = monte_carlo.random_timing_test(-5.0, bench, strat_rets, n_simulations=500)
    assert good['enabled'] and bad['enabled']
    assert 0.0 <= good['skill_percentile'] <= 100.0
    assert good['skill_percentile'] > bad['skill_percentile']
    assert good['skill_percentile'] >= 95.0 and good['is_significant'] is True
    assert bad['skill_percentile'] <= 5.0


def test_skill_test_percentile_moves_with_the_sharpe():
    """The percentile must actually track the Sharpe (unlike the self-resample
    bootstrap, which pins the actual at ~50 regardless). Straddle the null's
    own median so the two probes land on opposite sides of it."""
    rng = np.random.default_rng(8)
    bench = rng.normal(0.0003, 0.011, 300)
    strat_rets = rng.normal(0.0003, 0.011, 300)
    np.random.seed(1)
    base = monte_carlo.random_timing_test(0.0, bench, strat_rets, n_simulations=500)
    med = base['null_sharpe_median']
    np.random.seed(1)
    low = monte_carlo.random_timing_test(med - 0.5, bench, strat_rets, n_simulations=500)
    np.random.seed(1)
    high = monte_carlo.random_timing_test(med + 0.5, bench, strat_rets, n_simulations=500)
    assert high['skill_percentile'] > low['skill_percentile']
    assert 0.0 <= base['exposure_pct'] <= 100.0


def test_skill_test_disabled_on_short_series():
    out = monte_carlo.random_timing_test(1.0, [0.01] * 5, [0.01] * 5)
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
