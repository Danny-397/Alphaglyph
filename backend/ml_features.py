"""
Multi-modal feature engineering for the ML transformer strategy.

This module is the single source of truth for how raw market data becomes
model input — it is imported by BOTH the Colab training pipeline (ml/) and
the live server (ml_runtime.py).  Keeping one implementation eliminates
train/serve skew, the classic way financial ML silently breaks.

Each trading day becomes one feature vector with three modality blocks:

    PRICE  (8)  — log return, intraday range, open→close return, volume
                  z-score, scaled RSI, MACD histogram (price-normalised),
                  SMA20/50 ratio, 5-day return
    MACRO  (3)  — VIX close, 10y−2y Treasury spread, Fed Funds rate
                  (FRED — free, no API key)
    SENTI  (2)  — GDELT news tone for the ticker + tone vs its 21-day mean
                  (GDELT DOC 2.0 — free, no API key, history from 2017)

The model consumes sequences of SEQ_LEN consecutive days.  Macro and
sentiment blocks are zero-filled when a source is unavailable — the model
is trained with "modality dropout" (whole blocks randomly zeroed) so it
degrades gracefully instead of breaking.  This is also what makes future
paid modalities (options flow, social sentiment) drop-in additions: they
join as new blocks the model already knows how to live without.

Normalisation uses per-feature mean/std computed on the TRAINING SPLIT ONLY
(stored in ml_model_meta.json) — never on the data being predicted.
"""

from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import requests

import features as feat

logger = logging.getLogger(__name__)

# ── Model input contract ───────────────────────────────────────────────────────
SEQ_LEN  = 60    # trading days of history per prediction
HORIZON  = 5     # trading days ahead the labels describe

# Per-timestep feature columns, in the exact order the model expects.
PRICE_FEATURES = [
    'f_log_ret',      # 1-day log return
    'f_hl_range',     # (High − Low) / Close — intraday range
    'f_oc_ret',       # (Close − Open) / Open — candle body
    'f_vol_z',        # volume z-score vs trailing 20 days
    'f_rsi',          # (RSI14 − 50) / 50 → roughly [−1, 1]
    'f_macd',         # MACD histogram / Close — price-scale invariant
    'f_sma_ratio',    # SMA20 / SMA50 − 1 — trend posture
    'f_ret_5d',       # trailing 5-day return
]
MACRO_FEATURES = [
    'f_vix',          # VIX close (fear gauge)
    'f_term_spread',  # 10y − 2y Treasury yield spread (recession signal)
    'f_fedfunds',     # effective Fed Funds rate (policy stance)
]
SENTI_FEATURES = [
    'f_news_tone',    # GDELT average article tone for the ticker
    'f_tone_mom',     # tone minus its 21-day rolling mean (sentiment shift)
]
ALL_FEATURES = PRICE_FEATURES + MACRO_FEATURES + SENTI_FEATURES

# Index ranges of each modality block inside the feature vector — used for
# modality dropout in training and zero-filling at inference.
PRICE_SLICE = slice(0, len(PRICE_FEATURES))
MACRO_SLICE = slice(len(PRICE_FEATURES), len(PRICE_FEATURES) + len(MACRO_FEATURES))
SENTI_SLICE = slice(len(PRICE_FEATURES) + len(MACRO_FEATURES), len(ALL_FEATURES))

# ── In-process caches (macro/sentiment change at most daily) ──────────────────
_MACRO_CACHE: dict = {}
_SENTI_CACHE: dict = {}
_CACHE_TTL = 6 * 3600          # 6 hours

_FRED_SERIES = {
    'f_vix':      'VIXCLS',
    'f_dgs10':    'DGS10',
    'f_dgs2':     'DGS2',
    'f_fedfunds': 'DFF',
}
_FRED_URL  = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
_GDELT_URL = 'https://api.gdeltproject.org/api/v2/doc/doc'


# ── Macro block (FRED) ─────────────────────────────────────────────────────────

def fetch_macro(start: str, end: str) -> pd.DataFrame | None:
    """
    Daily macro features from FRED's free CSV endpoint (no API key).

    Returns a DataFrame indexed by date with MACRO_FEATURES columns,
    forward-filled over non-trading days, or None when FRED is unreachable.
    """
    key = ('macro', start, end)
    hit = _MACRO_CACHE.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    # Pad the window back so forward-fill always has a prior observation,
    # even when the requested range starts on a weekend or holiday.
    cosd = (pd.Timestamp(start) - pd.Timedelta(days=14)).strftime('%Y-%m-%d')

    series = {}
    for name, fred_id in _FRED_SERIES.items():
        try:
            # FRED hangs indefinitely without a browser User-Agent, and
            # bounding the dates keeps the CSV small and fast.
            resp = requests.get(
                _FRED_URL,
                params={'id': fred_id, 'cosd': cosd, 'coed': end},
                timeout=20,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                                       'AppleWebKit/537.36'})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), na_values='.')
            date_col, val_col = df.columns[0], df.columns[1]
            df[date_col] = pd.to_datetime(df[date_col])
            series[name] = df.set_index(date_col)[val_col]
        except Exception as exc:
            logger.warning('FRED fetch failed for %s: %s', fred_id, exc)
            return None

    out = pd.DataFrame(series)
    out['f_term_spread'] = out['f_dgs10'] - out['f_dgs2']
    out = out[['f_vix', 'f_term_spread', 'f_fedfunds']]
    out = out.ffill()    # fill gaps using the padded lead-in, THEN trim to range
    out = out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
    out = out.dropna()
    if out.empty:
        return None

    _MACRO_CACHE[key] = (time.time(), out)
    return out


# ── Sentiment block (GDELT) ────────────────────────────────────────────────────

def fetch_sentiment(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Daily news tone for a ticker from the GDELT DOC 2.0 timeline API
    (free, no key).  GDELT coverage starts 2017-01-01 — earlier dates are
    simply absent and the caller zero-fills (modality dropout handles it).

    Returns a DataFrame indexed by date with a single 'f_news_tone' column,
    or None when GDELT is unreachable / has no coverage.
    """
    key = ('senti', ticker, start, end)
    hit = _SENTI_CACHE.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    params = {
        'query':         f'"{ticker}" (stock OR shares OR earnings)',
        'mode':          'timelinetone',
        'format':        'json',
        'startdatetime': pd.Timestamp(start).strftime('%Y%m%d') + '000000',
        'enddatetime':   pd.Timestamp(end).strftime('%Y%m%d') + '235959',
    }
    try:
        resp = requests.get(_GDELT_URL, params=params, timeout=20,
                            headers={'User-Agent': 'TradeBot research (educational)'})
        if resp.status_code == 429:
            # GDELT's shared anonymous quota throttles in bursts — one spaced
            # retry recovers most of them (matters for the 80-ticker dataset
            # build; live inference just zero-fills and moves on).
            time.sleep(15)
            resp = requests.get(_GDELT_URL, params=params, timeout=20,
                                headers={'User-Agent': 'TradeBot research (educational)'})
        resp.raise_for_status()
        timeline = (resp.json().get('timeline') or [])
        if not timeline:
            return None
        points = timeline[0].get('data') or []
        if not points:
            return None
        idx  = pd.to_datetime([p['date'] for p in points]).tz_localize(None)
        vals = [float(p['value']) for p in points]
        out  = pd.DataFrame({'f_news_tone': vals}, index=idx)
        out  = out.resample('D').mean().ffill()
        _SENTI_CACHE[key] = (time.time(), out)
        return out
    except Exception as exc:
        logger.warning('GDELT fetch failed for %s: %s', ticker, exc)
        return None


# ── Feature frame assembly ─────────────────────────────────────────────────────

def build_feature_frame(ohlcv: pd.DataFrame, ticker: str,
                        include_macro: bool = True,
                        include_sentiment: bool = True) -> pd.DataFrame | None:
    """
    Turn a raw OHLCV DataFrame into the model's per-day feature matrix.

    Returns a DataFrame [date × ALL_FEATURES] with the indicator warm-up
    period removed, or None when there isn't enough history.  Macro and
    sentiment columns are zero-filled wherever unavailable.
    """
    if ohlcv is None or len(ohlcv) < feat.MIN_BARS + 5:
        return None

    df    = feat.compute_features(ohlcv)
    close = df['Close']

    out = pd.DataFrame(index=df.index)

    # PRICE block — all price-scale invariant so one model serves every ticker
    out['f_log_ret']   = np.log(close / close.shift(1))
    out['f_hl_range']  = (df['High'] - df['Low']) / close
    out['f_oc_ret']    = (close - df['Open']) / df['Open']
    vol_std            = df['Volume'].rolling(20).std()
    out['f_vol_z']     = ((df['Volume'] - df['vol_ma20']) / vol_std.replace(0, np.nan)).fillna(0)
    out['f_rsi']       = (df['rsi14'] - 50.0) / 50.0
    out['f_macd']      = df['macd_hist'] / close
    out['f_sma_ratio'] = df['sma20'] / df['sma50'] - 1.0
    out['f_ret_5d']    = df['return_5d']

    out = out.dropna()
    if len(out) < SEQ_LEN:
        return None

    start = out.index[0].strftime('%Y-%m-%d')
    end   = out.index[-1].strftime('%Y-%m-%d')

    # MACRO block — shared across tickers, zero-filled on failure
    macro = fetch_macro(start, end) if include_macro else None
    if macro is not None:
        out = out.join(macro.reindex(out.index).ffill())
    for col in MACRO_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    out[MACRO_FEATURES] = out[MACRO_FEATURES].fillna(0.0)

    # SENTI block — per ticker, zero-filled on failure or pre-2017 dates
    senti = fetch_sentiment(ticker, start, end) if include_sentiment else None
    if senti is not None:
        tone = senti['f_news_tone'].reindex(out.index).ffill()
        out['f_news_tone'] = tone
        out['f_tone_mom']  = tone - tone.rolling(21, min_periods=1).mean()
    for col in SENTI_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    out[SENTI_FEATURES] = out[SENTI_FEATURES].fillna(0.0)

    return out[ALL_FEATURES]


def windows_from_frame(frame: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Slice a feature frame into overlapping model-input windows.

    Returns (X, dates):
        X      — float32 array [n_windows, seq_len, n_features]
        dates  — the prediction date of each window (its last day)
    """
    values = frame.values.astype(np.float32)
    n = len(values) - seq_len + 1
    if n <= 0:
        return np.empty((0, seq_len, values.shape[1]), dtype=np.float32), []
    X = np.stack([values[i:i + seq_len] for i in range(n)])
    return X, list(frame.index[seq_len - 1:])


def normalize(X: np.ndarray, means: list[float], stds: list[float],
              clip: float = 5.0) -> np.ndarray:
    """
    Z-score features using TRAINING-set statistics, clipped to ±clip sigma
    so a single wild day can't blow up the attention weights.
    """
    mu    = np.asarray(means, dtype=np.float32)
    sigma = np.asarray(stds,  dtype=np.float32)
    sigma = np.where(sigma <= 0, 1.0, sigma)
    return np.clip((X - mu) / sigma, -clip, clip).astype(np.float32)


# ── Labels (used by the training pipeline only) ────────────────────────────────

def build_labels(ohlcv: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Forward-looking labels for the three model heads, indexed by decision date:

        y_ret  — cumulative log return over the next `horizon` trading days
        y_dir  — 1 if y_ret > 0 else 0
        y_vol  — annualised std of daily log returns over those days

    The last `horizon` rows are NaN (their future isn't known) and must be
    dropped — this is also why splits are purged at boundaries.
    """
    close   = ohlcv['Close']
    log_ret = np.log(close / close.shift(1))

    fwd_ret = np.log(close.shift(-horizon) / close)
    fwd_vol = (log_ret.shift(-horizon)
               .rolling(horizon).std()
               .shift(-(horizon - 1)) * np.sqrt(252))

    return pd.DataFrame({
        'y_ret': fwd_ret,
        'y_dir': (fwd_ret > 0).astype(float).where(fwd_ret.notna()),
        'y_vol': fwd_vol,
    }, index=ohlcv.index)
