"""
Unit tests for the ML transformer pipeline (ml_features.py + ml_runtime.py).

Everything runs offline: synthetic OHLCV, macro/sentiment fetches disabled or
monkeypatched, and the inference runtime is tested both WITHOUT a model file
(graceful degradation) and WITH a mocked ONNX session (signal mapping).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest

import ml_features as mlf
import ml_runtime as mlr
import strategies


# ── Synthetic data ─────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV long enough for indicator warmup + several windows."""
    rng    = np.random.default_rng(seed)
    rets   = rng.normal(0.0005, 0.015, n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    idx    = pd.date_range('2022-01-03', periods=n, freq='B')
    return pd.DataFrame({
        'Open':   closes * (1 + rng.normal(0, 0.003, n)),
        'High':   closes * (1 + np.abs(rng.normal(0, 0.008, n))),
        'Low':    closes * (1 - np.abs(rng.normal(0, 0.008, n))),
        'Close':  closes,
        'Volume': rng.integers(500_000, 5_000_000, n).astype(float),
    }, index=idx)


def _frame(n=200):
    return mlf.build_feature_frame(_make_ohlcv(n), 'TEST',
                                   include_macro=False, include_sentiment=False)


# ── Feature frame ──────────────────────────────────────────────────────────────

class TestFeatureFrame:
    def test_columns_match_contract_in_order(self):
        frame = _frame()
        assert list(frame.columns) == mlf.ALL_FEATURES

    def test_no_nans_after_warmup(self):
        frame = _frame()
        assert not frame.isna().any().any()

    def test_disabled_modalities_are_zero_filled(self):
        frame = _frame()
        assert (frame[mlf.MACRO_FEATURES].values == 0).all()
        assert (frame[mlf.SENTI_FEATURES].values == 0).all()

    def test_price_features_are_scale_invariant(self):
        """10x the price level → identical price-block features (this is what
        makes one model valid across all tickers)."""
        ohlcv = _make_ohlcv()
        big   = ohlcv.copy()
        for col in ('Open', 'High', 'Low', 'Close'):
            big[col] = big[col] * 10.0
        f1 = mlf.build_feature_frame(ohlcv, 'A', include_macro=False, include_sentiment=False)
        f2 = mlf.build_feature_frame(big,   'A', include_macro=False, include_sentiment=False)
        np.testing.assert_allclose(f1[mlf.PRICE_FEATURES].values,
                                   f2[mlf.PRICE_FEATURES].values, atol=1e-9)

    def test_too_short_history_returns_none(self):
        assert mlf.build_feature_frame(_make_ohlcv(40), 'TEST',
                                       include_macro=False, include_sentiment=False) is None

    def test_modality_slices_tile_the_vector_exactly(self):
        n = len(mlf.ALL_FEATURES)
        covered = (list(range(n))[mlf.PRICE_SLICE] +
                   list(range(n))[mlf.MACRO_SLICE] +
                   list(range(n))[mlf.SENTI_SLICE])
        assert covered == list(range(n))


# ── Windowing + normalisation ──────────────────────────────────────────────────

class TestWindows:
    def test_window_shape_and_count(self):
        frame = _frame()
        X, dates = mlf.windows_from_frame(frame)
        assert X.shape == (len(frame) - mlf.SEQ_LEN + 1, mlf.SEQ_LEN, len(mlf.ALL_FEATURES))
        assert len(dates) == X.shape[0]

    def test_window_ends_on_its_prediction_date(self):
        """The last row of window i must be the feature row of dates[i] —
        i.e. a prediction only sees data up to its own date (no look-ahead)."""
        frame = _frame()
        X, dates = mlf.windows_from_frame(frame)
        np.testing.assert_array_equal(X[0][-1],  frame.loc[dates[0]].values.astype(np.float32))
        np.testing.assert_array_equal(X[-1][-1], frame.loc[dates[-1]].values.astype(np.float32))

    def test_normalize_clips_outliers(self):
        X = np.array([[[100.0, -100.0]]], dtype=np.float32)
        out = mlf.normalize(X, means=[0.0, 0.0], stds=[1.0, 1.0], clip=5.0)
        assert out.max() == 5.0 and out.min() == -5.0

    def test_normalize_handles_zero_std(self):
        X   = np.ones((1, 1, 2), dtype=np.float32)
        out = mlf.normalize(X, means=[1.0, 1.0], stds=[0.0, 1.0])
        assert np.isfinite(out).all()


# ── Labels ─────────────────────────────────────────────────────────────────────

class TestLabels:
    def test_direction_label_matches_forward_return(self):
        ohlcv  = _make_ohlcv()
        labels = mlf.build_labels(ohlcv).dropna()
        assert ((labels['y_ret'] > 0) == (labels['y_dir'] == 1.0)).all()

    def test_last_horizon_rows_have_no_labels(self):
        ohlcv  = _make_ohlcv()
        labels = mlf.build_labels(ohlcv)
        assert labels['y_ret'].iloc[-mlf.HORIZON:].isna().all()

    def test_vol_label_is_non_negative(self):
        labels = mlf.build_labels(_make_ohlcv()).dropna()
        assert (labels['y_vol'] >= 0).all()


# ── Runtime without a model (graceful degradation) ─────────────────────────────

class TestRuntimeNoModel:
    @pytest.fixture(autouse=True)
    def _no_model(self, monkeypatch):
        monkeypatch.setattr(mlr, 'MODEL_PATH', '/nonexistent/model.onnx')
        monkeypatch.setattr(mlr, 'META_PATH',  '/nonexistent/meta.json')
        monkeypatch.setattr(mlr, '_session', None)
        monkeypatch.setattr(mlr, '_meta', None)
        monkeypatch.setattr(mlr, '_load_err', None)

    def test_not_available_and_info_explains_why(self):
        assert mlr.is_available() is False
        info = mlr.get_info()
        assert info['loaded'] is False and 'reason' in info

    def test_backtest_signals_returns_none(self):
        assert mlr.backtest_signals(_make_ohlcv(), 'TEST') is None

    def test_ml_strategy_holds_instead_of_crashing(self, monkeypatch):
        monkeypatch.setattr(strategies.feat, 'get_feature_df',
                            lambda *a, **k: _make_ohlcv())
        signal, price = strategies.get_signal('ml', 'TEST')
        assert signal == 'HOLD' and price is not None


# ── Signal mapping with a mocked session ───────────────────────────────────────

class _FakeSession:
    """Stands in for onnxruntime.InferenceSession: p_up from logits, q50, vol."""
    def __init__(self, logits, q50s):
        self._logits, self._q50s = logits, q50s

    def get_inputs(self):
        class _In:
            name = 'input'
        return [_In()]

    def run(self, _outs, feeds):
        n = len(self._logits)
        quantiles = np.zeros((n, 5), dtype=np.float32)
        quantiles[:, 2] = self._q50s
        return (np.array(self._logits, dtype=np.float32).reshape(-1, 1),
                quantiles,
                np.full((n, 1), 0.2, dtype=np.float32))


class TestSignalMapping:
    @pytest.fixture(autouse=True)
    def _fake_model(self, monkeypatch):
        meta = {'seq_len': mlf.SEQ_LEN, 'horizon': mlf.HORIZON,
                'feature_means': [0.0] * len(mlf.ALL_FEATURES),
                'feature_stds':  [1.0] * len(mlf.ALL_FEATURES),
                'thresholds': {'buy_prob': 0.55, 'sell_prob': 0.45}}
        monkeypatch.setattr(mlr, '_meta', meta)
        monkeypatch.setattr(mlr, '_load_err', None)

    def _map(self, monkeypatch, logits, q50s):
        monkeypatch.setattr(mlr, '_session', _FakeSession(logits, q50s))
        pred = mlr.predict_batch(np.zeros((len(logits), mlf.SEQ_LEN,
                                           len(mlf.ALL_FEATURES)), dtype=np.float32))
        return mlr._map_signals(pred['p_up'], pred['quantiles'][:, 2])

    def test_buy_needs_both_heads_to_agree(self, monkeypatch):
        # logit 2.0 → p≈0.88 (confident up); q50 positive vs negative
        sigs = self._map(monkeypatch, [2.0, 2.0], [0.01, -0.01])
        assert sigs[0] == 1      # both agree → BUY
        assert sigs[1] == 0      # distribution disagrees → HOLD

    def test_sell_needs_both_heads_to_agree(self, monkeypatch):
        sigs = self._map(monkeypatch, [-2.0, -2.0], [-0.01, 0.01])
        assert sigs[0] == -1
        assert sigs[1] == 0

    def test_uncertain_probability_holds(self, monkeypatch):
        # logit 0 → p=0.5, inside the (0.45, 0.55) dead zone
        sigs = self._map(monkeypatch, [0.0], [0.01])
        assert sigs[0] == 0

    def test_predict_batch_outputs_probabilities(self, monkeypatch):
        monkeypatch.setattr(mlr, '_session', _FakeSession([0.0, 10.0, -10.0], [0, 0, 0]))
        pred = mlr.predict_batch(np.zeros((3, mlf.SEQ_LEN, len(mlf.ALL_FEATURES)),
                                          dtype=np.float32))
        assert abs(pred['p_up'][0] - 0.5) < 1e-6
        assert pred['p_up'][1] > 0.99 and pred['p_up'][2] < 0.01
