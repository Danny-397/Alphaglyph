"""
ONNX inference runtime for the ML transformer strategy.

The model is trained offline (ml/train.py, Colab GPU) and exported to ONNX;
this module serves it with onnxruntime — ~40 MB instead of PyTorch's
700 MB+, which is what lets it run inside Render's 512 MB free tier.

Two artefacts must sit next to this file (both produced by ml/train.py):

    ml_model.onnx        — the trained transformer
    ml_model_meta.json   — feature list, train-split normalisation stats,
                           decision thresholds, validation metrics

If either is missing, everything degrades gracefully: the strategy holds,
the API reports the model as not loaded, and nothing crashes.  That keeps
the deployed bot stable while the model is being (re)trained.

Model outputs (per window):
    direction  — P(positive return over the next HORIZON days)
    quantiles  — q10/q25/q50/q75/q90 of that return (a distribution, not
                 a point guess — the honest way to forecast returns)
    volatility — annualised vol forecast over the horizon

Signal mapping is deliberately conservative: BUY needs both the classifier
AND the median of the return distribution to agree; SELL likewise.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import numpy as np
import pandas as pd

import features as feat
import ml_features as mlf

logger = logging.getLogger(__name__)

_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv('ML_MODEL_PATH', os.path.join(_DIR, 'ml_model.onnx'))
META_PATH  = os.getenv('ML_META_PATH',  os.path.join(_DIR, 'ml_model_meta.json'))

_lock     = threading.Lock()
_session  = None          # onnxruntime.InferenceSession (lazy)
_meta     = None          # parsed ml_model_meta.json
_load_err = None          # remembered failure so we don't retry every cycle

# Short cache for the interactive ML Insights panel — predictions only change on
# new daily bars, so this keeps the panel snappy without re-running inference on
# every poll/visitor.
_PRED_CACHE: dict = {}
_PRED_TTL = 1800          # 30 minutes


# ── Loading ────────────────────────────────────────────────────────────────────

def _ensure_loaded() -> bool:
    """Lazily load the ONNX session + meta. Returns True when usable."""
    global _session, _meta, _load_err
    if _session is not None:
        return True
    if _load_err is not None:
        return False
    with _lock:
        if _session is not None:
            return True
        if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
            _load_err = 'model files not present'
            return False
        try:
            import onnxruntime as ort
            with open(META_PATH) as fh:
                meta = json.load(fh)
            sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
            _meta, _session = meta, sess
            logger.info('ML model loaded: v%s (%s params, val AUC %.3f)',
                        meta.get('version', '?'), meta.get('n_params', '?'),
                        (meta.get('val_metrics') or {}).get('auc', float('nan')))
            return True
        except Exception as exc:
            _load_err = str(exc)
            logger.error('ML model failed to load: %s', exc)
            return False


def is_available() -> bool:
    return _ensure_loaded()


def get_info() -> dict:
    """Status payload for /api/ml/info and the frontend."""
    if not _ensure_loaded():
        return {'loaded': False, 'reason': _load_err or 'model files not present'}
    return {
        'loaded':      True,
        'version':     _meta.get('version'),
        'trained_at':  _meta.get('created'),
        'seq_len':     _meta.get('seq_len'),
        'horizon':     _meta.get('horizon'),
        'n_params':    _meta.get('n_params'),
        'features':     _meta.get('features'),
        'architecture': _meta.get('architecture'),
        'val_metrics':  _meta.get('val_metrics'),
        'test_metrics': _meta.get('test_metrics'),
        'thresholds':   _meta.get('thresholds'),
    }


# ── Core inference ─────────────────────────────────────────────────────────────

def predict_batch(X: np.ndarray) -> dict | None:
    """
    Run the model on a batch of normalised windows [B, seq_len, n_features].

    Returns {'p_up': [B], 'quantiles': [B, 5], 'vol': [B]} or None.
    """
    if not _ensure_loaded() or X.size == 0:
        return None
    try:
        input_name = _session.get_inputs()[0].name
        dir_logit, quantiles, vol = _session.run(None, {input_name: X.astype(np.float32)})
        return {
            'p_up':      1.0 / (1.0 + np.exp(-dir_logit.reshape(-1))),
            'quantiles': quantiles,
            'vol':       vol.reshape(-1),
        }
    except Exception as exc:
        logger.error('ML inference error: %s', exc)
        return None


def _prepare_windows(ohlcv: pd.DataFrame, ticker: str,
                     include_sentiment: bool | None = None,
                     include_macro: bool | None = None):
    """OHLCV → normalised model windows using the meta's train-split stats."""
    frame = mlf.build_feature_frame(ohlcv, ticker,
                                    include_macro=include_macro,
                                    include_sentiment=include_sentiment)
    if frame is None:
        return None, []
    X, dates = mlf.windows_from_frame(frame, int(_meta['seq_len']))
    if X.size == 0:
        return None, []
    X = mlf.normalize(X, _meta['feature_means'], _meta['feature_stds'],
                      clip=float(_meta.get('clip', 5.0)))
    return X, dates


def _map_signals(p_up: np.ndarray, q50: np.ndarray) -> np.ndarray:
    """
    Probabilities → {-1, 0, +1}.  Both heads must agree, so the model only
    trades when the direction classifier AND the return distribution line up.
    """
    th   = (_meta or {}).get('thresholds') or {}
    buy  = float(th.get('buy_prob', 0.55))
    sell = float(th.get('sell_prob', 0.45))
    out  = np.zeros(len(p_up), dtype=int)
    out[(p_up >= buy) & (q50 > 0)]  = 1
    out[(p_up <= sell) & (q50 < 0)] = -1
    return out


# ── Strategy-facing entry points ───────────────────────────────────────────────

def live_signal(ticker: str) -> tuple[str | None, float | None]:
    """
    Signal for the live bot: fetch recent data, predict the latest window.
    Mirrors the (signal, price) contract of the other strategies.
    """
    if not _ensure_loaded():
        return None, None
    ohlcv = feat.fetch_ohlcv(ticker, period='1y')
    if ohlcv is None or ohlcv.empty:
        return None, None
    price = float(ohlcv['Close'].iloc[-1])

    X, _dates = _prepare_windows(ohlcv, ticker)
    if X is None:
        return 'HOLD', price
    pred = predict_batch(X[-1:])
    if pred is None:
        return 'HOLD', price

    sig = _map_signals(pred['p_up'], pred['quantiles'][:, 2])[0]
    return ('BUY' if sig == 1 else 'SELL' if sig == -1 else 'HOLD'), price


def live_prediction(ticker: str) -> dict | None:
    """
    Full transformer prediction for one ticker's latest window — the rich output
    the BUY/SELL/HOLD signal is distilled from. Used by the ML Insights panel so
    users can see what the model actually forecasts, not just the final action.

    Returns a dict with the direction probability, the q10–q90 return
    distribution (as percentages over the horizon), the annualised vol forecast,
    and the mapped signal — or None when the model isn't loaded.
    """
    if not _ensure_loaded():
        return None

    import time
    hit = _PRED_CACHE.get(ticker)
    if hit and (time.time() - hit[0]) < _PRED_TTL:
        return hit[1]

    ohlcv = feat.fetch_ohlcv(ticker, period='1y')
    if ohlcv is None or ohlcv.empty:
        return {'ticker': ticker, 'available': False}
    price = float(ohlcv['Close'].iloc[-1])

    # Macro (FRED, ~90s cold) and sentiment (GDELT, rate-limited) are both off
    # on the interactive path so it stays fast and reliable. The model was
    # trained with modality dropout to run on price/technical features alone.
    X, _dates = _prepare_windows(ohlcv, ticker, include_sentiment=False, include_macro=False)
    if X is None:
        return {'ticker': ticker, 'price': round(price, 2), 'available': False}
    pred = predict_batch(X[-1:])
    if pred is None:
        return {'ticker': ticker, 'price': round(price, 2), 'available': False}

    q     = pred['quantiles'][0]                         # [q10, q25, q50, q75, q90]
    p_up  = float(pred['p_up'][0])
    vol   = float(pred['vol'][0])
    sig   = int(_map_signals(pred['p_up'], pred['quantiles'][:, 2])[0])
    result = {
        'ticker':    ticker,
        'available': True,
        'price':     round(price, 2),
        'p_up':      round(p_up, 4),
        'horizon':   int((_meta or {}).get('horizon', 5)),
        # Quantiles are returns over the horizon — expose as percentages.
        'quantiles': {
            'q10': round(float(q[0]) * 100, 2),
            'q25': round(float(q[1]) * 100, 2),
            'q50': round(float(q[2]) * 100, 2),
            'q75': round(float(q[3]) * 100, 2),
            'q90': round(float(q[4]) * 100, 2),
        },
        'vol':    round(vol * 100, 2),                   # annualised vol forecast, %
        'signal': 'BUY' if sig == 1 else 'SELL' if sig == -1 else 'HOLD',
    }
    _PRED_CACHE[ticker] = (time.time(), result)
    return result


def backtest_signals(ohlcv: pd.DataFrame, ticker: str) -> pd.Series | None:
    """
    Vectorised signals for the backtest engine: one batched ONNX call over
    every window in the date range.  Each day's prediction uses only data up
    to that day, so there is no look-ahead.

    Returns a {-1, 0, +1} Series indexed like the input, or None when the
    model isn't available (backtest then shows zero ML trades, not an error).
    """
    if not _ensure_loaded():
        return None
    # Macro (FRED, ~90s cold) and sentiment (GDELT, rate-limited) are both off
    # so a multi-ticker backtest can't time out. The model was trained with
    # modality dropout to run on price/technical features alone, and the live
    # signal path does the same — so backtest and live stay consistent.
    X, dates = _prepare_windows(ohlcv, ticker, include_sentiment=False, include_macro=False)
    if X is None:
        return None
    pred = predict_batch(X)
    if pred is None:
        return None
    sigs = _map_signals(pred['p_up'], pred['quantiles'][:, 2])
    return pd.Series(sigs, index=dates).reindex(ohlcv.index).fillna(0).astype(int)
