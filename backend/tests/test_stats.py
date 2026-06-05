"""
Unit tests for stats.py — Probabilistic/Deflated Sharpe and Fama-French.

All tests are pure math or use synthetic in-memory data.
No network calls are made.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import stats as st


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normal_returns(mean_annual: float, vol_annual: float,
                    n: int = 252, seed: int = 42) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    mu    = mean_annual / st._TRADING_DAYS
    sigma = vol_annual  / np.sqrt(st._TRADING_DAYS)
    return rng.normal(mu, sigma, n)


def _make_port_hist(returns: np.ndarray, start: float = 100_000.0) -> list[dict]:
    """Convert a return array into a port_hist list."""
    values = [start]
    for r in returns:
        values.append(values[-1] * (1 + r))
    dates  = pd.date_range('2023-01-01', periods=len(values), freq='B')
    return [{'date': d.strftime('%Y-%m-%d'), 'value': v}
            for d, v in zip(dates, values)]


def _make_ff3_data(n: int = 252, seed: int = 7) -> pd.DataFrame:
    """Synthetic FF3 factor data aligned to the same date range."""
    rng  = np.random.default_rng(seed)
    idx  = pd.date_range('2023-01-01', periods=n, freq='B')
    return pd.DataFrame({
        'mkt_rf': rng.normal(0.0003, 0.010, n),
        'smb':    rng.normal(0.0001, 0.005, n),
        'hml':    rng.normal(0.0001, 0.005, n),
        'rf':     np.full(n, 0.04 / 252),
    }, index=idx)


# ── PSR tests ──────────────────────────────────────────────────────────────────

class TestProbabilisticSharpeRatio:
    def test_psr_above_half_for_positive_sharpe(self):
        # 5 years of data gives enough power to reliably detect Sharpe > 0
        rets = _normal_returns(0.12, 0.15, n=1260, seed=42)
        psr  = st.probabilistic_sharpe_ratio(rets, sr_benchmark_annual=0.0)
        assert psr > 0.5

    def test_psr_below_half_for_negative_sharpe(self):
        rets = _normal_returns(-0.10, 0.15)
        psr  = st.probabilistic_sharpe_ratio(rets, sr_benchmark_annual=0.0)
        assert psr < 0.5

    def test_psr_is_between_zero_and_one(self):
        rets = _normal_returns(0.15, 0.20)
        psr  = st.probabilistic_sharpe_ratio(rets)
        assert 0.0 <= psr <= 1.0

    def test_psr_increases_with_sample_size(self):
        # Same annualised Sharpe, more data → higher confidence
        rets_short = _normal_returns(0.12, 0.15, n=63)
        rets_long  = _normal_returns(0.12, 0.15, n=504)
        psr_short  = st.probabilistic_sharpe_ratio(rets_short)
        psr_long   = st.probabilistic_sharpe_ratio(rets_long)
        assert psr_long > psr_short

    def test_psr_returns_nan_for_too_few_obs(self):
        rets = _normal_returns(0.10, 0.15, n=5)
        psr  = st.probabilistic_sharpe_ratio(rets)
        assert np.isnan(psr)

    def test_psr_returns_nan_for_zero_variance(self):
        rets = np.zeros(100)
        psr  = st.probabilistic_sharpe_ratio(rets)
        assert np.isnan(psr)

    def test_higher_benchmark_lowers_psr(self):
        rets = _normal_returns(0.12, 0.15)
        psr0 = st.probabilistic_sharpe_ratio(rets, sr_benchmark_annual=0.0)
        psr1 = st.probabilistic_sharpe_ratio(rets, sr_benchmark_annual=1.0)
        assert psr0 > psr1


# ── DSR tests ──────────────────────────────────────────────────────────────────

class TestDeflatedSharpeRatio:
    def test_dsr_has_required_keys(self):
        rets   = _normal_returns(0.12, 0.15)
        result = st.deflated_sharpe_ratio(rets)
        for key in ('sr_annual', 'sr_benchmark', 'psr', 'dsr',
                    'is_significant', 'n_strategies'):
            assert key in result

    def test_dsr_le_psr_with_multiple_strategies(self):
        # After correcting for N>1 strategies, DSR should be <= PSR
        rets   = _normal_returns(0.15, 0.15, n=504)
        result = st.deflated_sharpe_ratio(rets, n_strategies=4)
        assert result['psr'] is not None
        assert result['dsr'] is not None
        assert result['dsr'] <= result['psr'] + 1e-9

    def test_dsr_equals_psr_for_n1(self):
        # n_strategies=1 means no multiple-testing correction → DSR = PSR
        rets    = _normal_returns(0.10, 0.15)
        result  = st.deflated_sharpe_ratio(rets, n_strategies=1)
        assert result['sr_benchmark'] == 0.0
        assert abs(result['dsr'] - result['psr']) < 1e-9

    def test_benchmark_increases_with_n_strategies(self):
        rets = _normal_returns(0.10, 0.15)
        r4   = st.deflated_sharpe_ratio(rets, n_strategies=4)
        r10  = st.deflated_sharpe_ratio(rets, n_strategies=10)
        assert r10['sr_benchmark'] > r4['sr_benchmark']

    def test_returns_null_dict_for_insufficient_data(self):
        rets   = _normal_returns(0.10, 0.15, n=5)
        result = st.deflated_sharpe_ratio(rets)
        assert result['psr'] is None
        assert result['dsr'] is None
        assert result['is_significant'] is False

    def test_very_high_sharpe_is_significant(self):
        # Sharpe of 3+ over 3 years, 5 strategies → should be significant
        rets   = _normal_returns(0.45, 0.15, n=756)
        result = st.deflated_sharpe_ratio(rets, n_strategies=5)
        assert result['dsr'] is not None
        assert result['dsr'] > 0.95
        assert result['is_significant'] is True


# ── FF3 CSV parsing ────────────────────────────────────────────────────────────

class TestParseFF3CSV:
    _SAMPLE = """\
Fama/French 3 Factors (daily)
    Mkt-RF       SMB       HML        RF
19230703,  0.10, -0.24, -0.28, 0.009
19230705, -0.05,  0.12,  0.08, 0.009
19230706,  0.20,  0.05, -0.10, 0.009
"""

    def test_parses_to_dataframe(self):
        df = st._parse_ff3_csv(self._SAMPLE)
        assert not df.empty

    def test_has_correct_columns(self):
        df = st._parse_ff3_csv(self._SAMPLE)
        for col in ('mkt_rf', 'smb', 'hml', 'rf'):
            assert col in df.columns

    def test_converts_percent_to_decimal(self):
        df = st._parse_ff3_csv(self._SAMPLE)
        assert abs(df['mkt_rf'].iloc[0] - 0.0010) < 1e-9

    def test_index_is_datetime(self):
        df = st._parse_ff3_csv(self._SAMPLE)
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_returns_empty_for_garbage_input(self):
        df = st._parse_ff3_csv('not a csv file\njust text here')
        assert df.empty


# ── Fama-French decomposition ──────────────────────────────────────────────────

class TestFamaFrenchDecomposition:
    def test_returns_disabled_for_short_history(self):
        hist   = _make_port_hist(_normal_returns(0.10, 0.15, n=20))
        result = st.fama_french_decomposition(hist, ff3_data=_make_ff3_data(20))
        assert result['enabled'] is False

    def test_output_has_required_keys(self):
        rets   = _normal_returns(0.10, 0.15, n=252)
        hist   = _make_port_hist(rets)
        ff3    = _make_ff3_data(253)
        result = st.fama_french_decomposition(hist, ff3_data=ff3)
        for key in ('enabled', 'alpha_annual', 'beta_market', 'beta_smb',
                    'beta_hml', 'r_squared', 't_stats', 'interpretation'):
            assert key in result

    def test_market_following_portfolio_has_beta_near_one(self):
        # Portfolio that exactly tracks the market → beta_market ≈ 1, α ≈ 0.
        # In fama_french_decomposition the i-th portfolio return ends up matched
        # against ff3_data.iloc[i+1] after the inner join on dates, so the
        # synthetic returns must be sourced from ff3.iloc[1:] to align correctly.
        ff3  = _make_ff3_data(253)
        port_rets = ff3['mkt_rf'].values[1:] + ff3['rf'].values[1:]
        hist = _make_port_hist(port_rets)
        result = st.fama_french_decomposition(hist, ff3_data=ff3)
        assert result['enabled'] is True
        assert abs(result['beta_market'] - 1.0) < 0.15

    def test_r_squared_between_zero_and_one(self):
        rets   = _normal_returns(0.10, 0.15, n=252)
        hist   = _make_port_hist(rets)
        ff3    = _make_ff3_data(253)
        result = st.fama_french_decomposition(hist, ff3_data=ff3)
        assert result['enabled'] is True
        assert 0.0 <= result['r_squared'] <= 1.0

    def test_interpretation_is_string(self):
        rets   = _normal_returns(0.10, 0.15, n=252)
        hist   = _make_port_hist(rets)
        ff3    = _make_ff3_data(253)
        result = st.fama_french_decomposition(hist, ff3_data=ff3)
        assert result['enabled'] is True
        assert isinstance(result['interpretation'], str)
        assert len(result['interpretation']) > 20
