"""
Unit tests for the research features:
  * calibration.py  — reliability scoring & binning of the price forecaster
  * pead.py         — pooled post-earnings-drift event study
  * datamining.py   — random-strategy sweep + Deflated-Sharpe deflation
  * portfolio.ledoit_wolf_cov — shrinkage estimator properties

Network boundaries (yfinance, ML inference) are monkeypatched; the maths runs
on synthetic data so the tests are fast and deterministic.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

import calibration
import datamining
import pead
import portfolio


# ── calibration.py ──────────────────────────────────────────────────────────────

def test_calibration_of_a_perfect_forecaster(monkeypatch):
    """A forecaster whose p exactly equals the outcome frequency should score a
    near-perfect Brier skill and sit on the diagonal."""
    rng = np.random.default_rng(0)
    n = 4000
    p = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < p).astype(float)     # outcomes drawn AT the stated p

    monkeypatch.setattr(calibration.ml_runtime, 'get_info', lambda: {'loaded': True})
    monkeypatch.setattr(calibration.ml_runtime, '_ensure_loaded', lambda: True)
    calibration.ml_runtime._meta = {'horizon': 5, 'test_metrics': {'auc': 0.5}}
    monkeypatch.setattr(calibration, '_collect',
                        lambda t, h, c: (p, y, ['AAA'], pd.DatetimeIndex(
                            pd.date_range('2024-01-01', periods=n, freq='h'))))
    calibration._CACHE.clear()

    out = calibration.compute_calibration(['AAA'], oos_only=False)
    assert out['available'] is True
    assert out['metrics']['brier_skill'] > 0.05         # positive skill vs base rate
    # Points should lie close to the diagonal (predicted ≈ actual per bin) — the
    # defining property of a well-calibrated forecaster.
    for b in out['bins']:
        assert abs(b['mean_pred'] - b['frac_up']) < 0.08


def test_calibration_handles_no_data(monkeypatch):
    monkeypatch.setattr(calibration.ml_runtime, 'get_info', lambda: {'loaded': True})
    monkeypatch.setattr(calibration.ml_runtime, '_ensure_loaded', lambda: True)
    calibration.ml_runtime._meta = {'horizon': 5}
    monkeypatch.setattr(calibration, '_collect', lambda t, h, c: None)
    calibration._CACHE.clear()
    out = calibration.compute_calibration(['AAA'])
    assert out['available'] is False


# ── datamining.py ───────────────────────────────────────────────────────────────

def _price_df(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    idx = pd.date_range('2020-01-01', periods=n, freq='B')
    return pd.DataFrame({'Close': closes}, index=idx)


def test_datamine_deflates_the_winner(monkeypatch):
    """The best of many random strategies should look good naively but fail the
    Deflated Sharpe — the whole point of the lab."""
    monkeypatch.setattr(datamining.feat, 'get_feature_df', lambda t, period='5y': _price_df())
    out = datamining.run_sweep('AAA', 400, seed=1)
    assert out['available'] is True
    assert out['best_sharpe'] > out['median_sharpe']            # best is an outlier
    # DSR benchmark (expected max of N noise strategies) should exceed the median.
    assert out['expected_max_sharpe'] > out['median_sharpe']
    # And the deflated verdict should not crown pure noise as skill.
    assert out['deflated_significant'] is False


def test_datamine_clamps_n(monkeypatch):
    monkeypatch.setattr(datamining.feat, 'get_feature_df', lambda t, period='5y': _price_df())
    out = datamining.run_sweep('AAA', 999999, seed=1)
    assert out['n_strategies'] <= datamining._MAX_N


def test_datamine_insufficient_history(monkeypatch):
    monkeypatch.setattr(datamining.feat, 'get_feature_df', lambda t, period='5y': _price_df(50))
    out = datamining.run_sweep('AAA', 100)
    assert out['available'] is False


# ── pead.py ─────────────────────────────────────────────────────────────────────

def test_pead_detects_planted_drift(monkeypatch):
    """Plant a real post-earnings drift and confirm the study recovers a positive,
    significant beat-minus-miss spread."""
    n = 1500
    idx = pd.date_range('2019-01-01', periods=n, freq='B')
    rng = np.random.default_rng(2)

    # Flat market; stock = small noise + a drift bump injected after each event.
    spy = pd.DataFrame({'Close': 100 + np.cumsum(rng.normal(0, 0.2, n))}, index=idx)
    stock_ret = rng.normal(0, 0.005, n)

    # Events every ~90 days; beats (+surprise) drift up, misses (−) drift down.
    events, surprises = [], []
    for k, pos in enumerate(range(80, n - 40, 90)):
        surp = 10.0 if k % 2 == 0 else -10.0
        surprises.append(surp)
        events.append(idx[pos])
        drift = (0.004 if surp > 0 else -0.004)
        stock_ret[pos + 1:pos + 21] += drift            # 20-day drift after the event
    stock_close = 100 * np.exp(np.cumsum(stock_ret))
    stock = pd.DataFrame({'Close': stock_close}, index=idx)

    ed_frame = pd.DataFrame(
        {'eps_est': np.nan, 'eps_actual': 1.0, 'surprise_pct': surprises},
        index=pd.DatetimeIndex(events))

    def fake_fetch(t, period='5y'):
        return spy if t == 'SPY' else stock
    monkeypatch.setattr(pead.feat, 'fetch_ohlcv', fake_fetch)
    monkeypatch.setattr(pead.earn, '_fetch_earnings_frame', lambda t: ed_frame)
    pead._CACHE.clear()

    out = pead.compute_pead(['AAA', 'BBB'], window=20)
    assert out['available'] is True
    assert out['spread']['beat_minus_miss_pct'] > 0
    assert out['spread']['t_stat'] > 0
    # A ~0.8%/side planted drift over this many events should clear significance.
    assert out['spread']['significant'] is True


def test_pead_thin_sample_degrades(monkeypatch):
    monkeypatch.setattr(pead.feat, 'fetch_ohlcv',
                        lambda t, period='5y': _price_df().assign(Close=lambda d: d['Close']))
    monkeypatch.setattr(pead.earn, '_fetch_earnings_frame', lambda t: None)
    pead._CACHE.clear()
    out = pead.compute_pead(['AAA'], window=20)
    assert out['available'] is False


# ── Ledoit-Wolf shrinkage ────────────────────────────────────────────────────────

def test_ledoit_wolf_properties():
    rng = np.random.default_rng(7)
    T, N = 300, 6
    L = rng.normal(size=(N, N)) * 0.01
    cov_true = L @ L.T + np.eye(N) * 1e-4
    Y = rng.multivariate_normal(np.zeros(N), cov_true, size=T)
    df = pd.DataFrame(Y, columns=[f'A{i}' for i in range(N)])

    shrunk, delta = portfolio.ledoit_wolf_cov(df)
    assert 0.0 <= delta <= 1.0
    assert np.allclose(shrunk, shrunk.T)                       # symmetric
    assert np.all(np.linalg.eigvalsh(shrunk) > 0)             # positive-definite
    # Shrinkage should not worsen conditioning.
    sample = (df.values - df.values.mean(0)).T @ (df.values - df.values.mean(0)) / T
    assert np.linalg.cond(shrunk) <= np.linalg.cond(sample) + 1e-6


def test_ledoit_wolf_shrinks_noise_more_than_structure():
    """The optimal intensity should be LARGER for pure-noise data (whose sample
    covariance deviates from the target only by chance) than for strongly
    structured data with plenty of samples (where the deviation is real signal
    worth keeping)."""
    def _delta(cov_true, seed):
        rng = np.random.default_rng(seed)
        Y = rng.multivariate_normal(np.zeros(4), cov_true, size=5000)
        return portfolio.ledoit_wolf_cov(pd.DataFrame(Y))[1]

    noise = np.eye(4)
    structured = (np.full((4, 4), 0.9) + np.eye(4) * 0.1) * 0.01
    assert _delta(noise, 1) > _delta(structured, 1)
    assert _delta(structured, 1) < 0.1                        # real structure → keep it
