"""
Do the statistics mean what they claim? Checked by simulation.

The other stats tests check formulas on single examples. These check the
property each statistic exists for, on data where the right answer is known:

  - PSR at the 95% level should call pure noise "significant" about 5% of the
    time, including with fat tails, since its correction for skew and kurtosis
    is the point of it.
  - Picking the best of N noise strategies is where a plain Sharpe test fails.
    PSR on the winner should be fooled most of the time; DSR should not.
  - PBO should sit near 0.5 when every strategy is noise (selection carries no
    information) and near 0 when one strategy has a real edge.

Every simulation is seeded, so the numbers are fixed; the bounds leave room
for the sampling error of the simulation itself, not for a broken statistic.
Measured when written: PSR 5.05% (normal) and 5.1% (Student-t, 4 df); best of
20: PSR 63%, DSR 0.5%; PBO 0.49 on noise and 0.07 with a planted edge.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import cpcv
import stats as st

T = 504          # two years of daily returns


def _psr_false_positive_rate(draw, n=2000):
    psr = np.array([st.probabilistic_sharpe_ratio(draw()) for _ in range(n)])
    return float(np.mean(psr > 0.95))


def test_psr_is_calibrated_on_normal_noise():
    rng = np.random.default_rng(7)
    rate = _psr_false_positive_rate(lambda: rng.normal(0, 0.01, T))
    assert 0.035 <= rate <= 0.065, rate


def test_psr_stays_calibrated_with_fat_tails():
    rng = np.random.default_rng(8)
    rate = _psr_false_positive_rate(lambda: rng.standard_t(4, T) * 0.007)
    assert 0.035 <= rate <= 0.065, rate


def test_dsr_is_not_fooled_by_the_best_of_many_noise_strategies():
    rng = np.random.default_rng(9)
    n_strategies, psr_hits, dsr_hits, trials = 20, 0, 0, 1000
    for _ in range(trials):
        R = rng.normal(0, 0.01, (T, n_strategies))
        winner = R[:, np.argmax(R.mean(0) / R.std(0, ddof=1))]
        out = st.deflated_sharpe_ratio(winner, n_strategies=n_strategies)
        psr_hits += out['psr'] > 0.95
        dsr_hits += out['dsr'] > 0.95
    # Selection bias: the undeflated test calls the winner significant most of
    # the time. DSR holds the false-positive rate under its nominal 5%; it is
    # conservative, because it tests against the *expected* best of N.
    assert psr_hits / trials > 0.4, psr_hits / trials
    assert dsr_hits / trials < 0.05, dsr_hits / trials


def test_pbo_is_about_a_coin_flip_when_every_strategy_is_noise():
    rng = np.random.default_rng(10)
    pbo = [cpcv.pbo_from_matrix(rng.normal(0, 0.01, (1000, 30)))['pbo'] for _ in range(20)]
    assert 0.35 <= np.mean(pbo) <= 0.65, np.mean(pbo)


def test_pbo_is_low_when_one_strategy_has_a_real_edge():
    rng = np.random.default_rng(11)
    pbo = []
    for _ in range(20):
        M = rng.normal(0, 0.01, (1000, 30))
        M[:, 0] += 0.0015            # an annualised Sharpe of about 2.4
        pbo.append(cpcv.pbo_from_matrix(M)['pbo'])
    assert np.mean(pbo) < 0.2, np.mean(pbo)
