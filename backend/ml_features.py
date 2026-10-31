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
import os
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

# ── In-process sentiment cache (macro uses its own covering cache below) ──────
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
# FRED's CSV endpoint is fast for short ranges but times out on multi-year ones,
# so each series is fetched in <=1-year chunks. The assembled frame is cached
# with a covering-range check, so an 80-ticker dataset build fetches FRED only
# once (each ticker just slices the cache) — the fix for the Colab timeouts.

_HTTP_UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
_MACRO_MIN_START = pd.Timestamp('2010-01-01')
_macro_frame: pd.DataFrame | None = None
_macro_lo: pd.Timestamp | None = None
_macro_hi: pd.Timestamp | None = None
_macro_failed = False


def _fred_series_chunked(fred_id: str, lo: pd.Timestamp, hi: pd.Timestamp):
    """Fetch one FRED series in <=1-year chunks; returns a Series or None."""
    parts, cur = [], lo
    while cur <= hi:
        chunk_hi = min(cur + pd.DateOffset(years=1), hi)
        for attempt in range(3):
            try:
                resp = requests.get(
                    _FRED_URL,
                    params={'id': fred_id, 'cosd': cur.strftime('%Y-%m-%d'),
                            'coed': chunk_hi.strftime('%Y-%m-%d')},
                    timeout=25, headers=_HTTP_UA)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text), na_values='.')
                dcol, vcol = df.columns[0], df.columns[1]
                df[dcol] = pd.to_datetime(df[dcol])
                parts.append(df.set_index(dcol)[vcol])
                break
            except Exception as exc:
                logger.warning('FRED %s %d attempt %d/3: %s',
                               fred_id, cur.year, attempt + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        else:
            return None           # a chunk exhausted its retries
        cur = chunk_hi + pd.Timedelta(days=1)
    if not parts:
        return None
    s = pd.concat(parts)
    return s[~s.index.duplicated(keep='last')].sort_index()


def _ensure_macro(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame | None:
    """Ensure the cached macro frame covers [lo, hi]; fetch (chunked) if not."""
    global _macro_frame, _macro_lo, _macro_hi, _macro_failed
    if _macro_failed:
        return None
    lo = max(lo, _MACRO_MIN_START)
    if _macro_frame is not None and _macro_lo <= lo and _macro_hi >= hi:
        return _macro_frame

    flo = min(lo, _macro_lo) if _macro_lo is not None else lo
    fhi = max(hi, _macro_hi) if _macro_hi is not None else hi
    series = {}
    for name, fred_id in _FRED_SERIES.items():
        s = _fred_series_chunked(fred_id, flo, fhi)
        if s is None:
            logger.warning('FRED unavailable — macro features zero-filled for this run.')
            _macro_failed = True
            return None
        series[name] = s

    out = pd.DataFrame(series)
    out['f_term_spread'] = out['f_dgs10'] - out['f_dgs2']
    out = out[['f_vix', 'f_term_spread', 'f_fedfunds']].ffill().dropna()
    _macro_frame, _macro_lo, _macro_hi = out, flo, fhi
    return out


def fetch_macro(start: str, end: str) -> pd.DataFrame | None:
    """
    Daily macro features (VIX, 10y-2y spread, Fed Funds) for [start, end],
    sliced from a chunk-fetched, covering-cached FRED frame.  Returns None
    when FRED is unavailable (caller then zero-fills).
    """
    if os.getenv('ML_SKIP_MACRO'):
        return None
    lo  = pd.Timestamp(start) - pd.Timedelta(days=14)
    raw = _ensure_macro(lo, pd.Timestamp(end))
    if raw is None:
        return None
    out = raw.loc[(raw.index >= lo) & (raw.index <= pd.Timestamp(end))]
    return out if not out.empty else None


# ── Sentiment block (GDELT) ────────────────────────────────────────────────────
# GDELT's free anonymous quota rate-limits hard during bursts (e.g. from Colab).
# After several consecutive failures we disable it for the rest of the run and
# zero-fill — the model's modality dropout is built exactly for this, so a
# build never stalls waiting on news data. Set ML_SKIP_SENTIMENT=1 to skip it
# entirely up front.

_SENTI_FAILS    = 0
_SENTI_DISABLED = False
_SENTI_LAST_CALL = 0.0
_SENTI_SPACING  = 5.5         # GDELT asks for at most one request every 5 s
_SENTI_MAX_FAILS = 6          # consecutive failures before giving up for the run
_SENTI_RETRY_SLEEP = 15       # seconds before a single 429 retry

# GDELT matches words in article text, so a bare ticker is a poor query: it
# rejects a quoted phrase as short as "AAPL" outright, and short tickers such
# as C, V, MA, GE or SO match unrelated words. Query the company's name for
# every ticker in the training universe (ml/dataset.py) and the common ones
# people look up; anything else falls back to the unquoted ticker when it is
# at least four letters, and to no news at all when it is shorter.
_NEWS_NAMES = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet', 'GOOG': 'Alphabet',
    'AMZN': 'Amazon', 'META': 'Meta Platforms', 'NVDA': 'Nvidia', 'AVGO': 'Broadcom',
    'ORCL': 'Oracle', 'CRM': 'Salesforce', 'ADBE': 'Adobe', 'AMD': 'Advanced Micro Devices',
    'INTC': 'Intel', 'QCOM': 'Qualcomm', 'TXN': 'Texas Instruments', 'CSCO': 'Cisco',
    'IBM': 'IBM', 'NFLX': 'Netflix', 'DIS': 'Disney', 'TMUS': 'T-Mobile', 'VZ': 'Verizon',
    'JPM': 'JPMorgan', 'BAC': 'Bank of America', 'WFC': 'Wells Fargo',
    'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley', 'C': 'Citigroup', 'BLK': 'BlackRock',
    'SCHW': 'Charles Schwab', 'AXP': 'American Express', 'V': 'Visa', 'MA': 'Mastercard',
    'COF': 'Capital One', 'UNH': 'UnitedHealth', 'JNJ': 'Johnson & Johnson',
    'LLY': 'Eli Lilly', 'PFE': 'Pfizer', 'MRK': 'Merck', 'ABBV': 'AbbVie',
    'TMO': 'Thermo Fisher', 'ABT': 'Abbott Laboratories', 'BMY': 'Bristol Myers Squibb',
    'AMGN': 'Amgen', 'GILD': 'Gilead', 'CVS': 'CVS Health', 'WMT': 'Walmart',
    'PG': 'Procter & Gamble', 'KO': 'Coca-Cola', 'PEP': 'PepsiCo', 'COST': 'Costco',
    'MCD': "McDonald's", 'NKE': 'Nike', 'SBUX': 'Starbucks', 'TGT': 'Target Corp',
    'HD': 'Home Depot', 'LOW': "Lowe's", 'TSLA': 'Tesla', 'CAT': 'Caterpillar',
    'DE': 'John Deere', 'BA': 'Boeing', 'GE': 'General Electric', 'HON': 'Honeywell',
    'UPS': 'UPS', 'UNP': 'Union Pacific', 'LMT': 'Lockheed Martin', 'RTX': 'Raytheon',
    'MMM': '3M', 'XOM': 'Exxon Mobil', 'CVX': 'Chevron', 'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger', 'EOG': 'EOG Resources', 'LIN': 'Linde', 'FCX': 'Freeport-McMoRan',
    'NEM': 'Newmont', 'NEE': 'NextEra Energy', 'DUK': 'Duke Energy', 'SO': 'Southern Company',
    'AMT': 'American Tower', 'PLD': 'Prologis', 'SPG': 'Simon Property', 'CL': 'Colgate-Palmolive',
    'KMB': 'Kimberly-Clark', 'GIS': 'General Mills', 'MO': 'Altria', 'SPY': 'S&P 500',
    'PLTR': 'Palantir', 'SOFI': 'SoFi', 'COIN': 'Coinbase', 'UBER': 'Uber', 'SHOP': 'Shopify',
}


def news_query(ticker: str) -> str | None:
    """The GDELT query for a ticker, or None when no meaningful one exists."""
    t = ticker.upper()
    name = _NEWS_NAMES.get(t)
    if name:
        # A phrase only when it must be one: GDELT rejects quoted phrases that
        # are too short, and single words need no quotes.
        term = f'"{name}"' if ' ' in name or '&' in name or '-' in name else name
    elif len(t) >= 4 and t.isalpha():
        term = t
    else:
        return None
    # The name alone: GDELT treats a multi-term query over a multi-year range
    # as a "large query" and rate-limits it, while a single term or phrase
    # returns the whole timeline in one request.
    return term



def fetch_sentiment(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """
    Daily news tone for a ticker from the GDELT DOC 2.0 timeline API
    (free, no key).  GDELT coverage starts ~2017 — earlier dates are simply
    absent and the caller zero-fills.  Returns a DataFrame with a single
    'f_news_tone' column, or None when GDELT has no coverage / is unavailable.
    """
    global _SENTI_FAILS, _SENTI_DISABLED
    if _SENTI_DISABLED or os.getenv('ML_SKIP_SENTIMENT'):
        return None

    key = ('senti', ticker, start, end)
    hit = _SENTI_CACHE.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    # Training reads news that ml/fetch_news.py saved to disk, so a dataset
    # build never depends on GDELT's rate limit (ML_NEWS_DIR; unset live).
    news_dir = os.getenv('ML_NEWS_DIR')
    if news_dir:
        path = os.path.join(news_dir, f'{ticker}.csv')
        if not os.path.exists(path):
            return None
        disk = pd.read_csv(path, index_col='date', parse_dates=True)
        disk = disk.loc[(disk.index >= pd.Timestamp(start)) & (disk.index <= pd.Timestamp(end))]
        return disk.resample('D').mean().ffill() if not disk.empty else None

    query = news_query(ticker)
    if query is None:
        return None               # no unambiguous query — not a failure

    global _SENTI_LAST_CALL
    wait = _SENTI_LAST_CALL + _SENTI_SPACING - time.time()
    if wait > 0:
        time.sleep(wait)
    _SENTI_LAST_CALL = time.time()

    params = {
        'query':         query,
        'mode':          'timelinetone',
        'format':        'json',
        'startdatetime': pd.Timestamp(start).strftime('%Y%m%d') + '000000',
        'enddatetime':   pd.Timestamp(end).strftime('%Y%m%d') + '235959',
    }
    try:
        resp = requests.get(_GDELT_URL, params=params, timeout=20,
                            headers={'User-Agent': 'AlphaGlyph research (educational)'})
        if resp.status_code == 429:
            time.sleep(_SENTI_RETRY_SLEEP)
            _SENTI_LAST_CALL = time.time()
            resp = requests.get(_GDELT_URL, params=params, timeout=20,
                                headers={'User-Agent': 'AlphaGlyph research (educational)'})
        resp.raise_for_status()
        _SENTI_FAILS = 0          # a clean response resets the failure streak
        timeline = (resp.json().get('timeline') or [])
        points   = timeline[0].get('data') if timeline else None
        if not points:
            return None           # no coverage for this ticker — not a failure
        idx  = pd.to_datetime([p['date'] for p in points]).tz_localize(None)
        vals = [float(p['value']) for p in points]
        out  = pd.DataFrame({'f_news_tone': vals}, index=idx)
        out  = out.resample('D').mean().ffill()
        _SENTI_CACHE[key] = (time.time(), out)
        return out
    except Exception as exc:
        _SENTI_FAILS += 1
        logger.warning('GDELT fetch failed for %s (%d/%d): %s',
                       ticker, _SENTI_FAILS, _SENTI_MAX_FAILS, exc)
        if _SENTI_FAILS >= _SENTI_MAX_FAILS:
            _SENTI_DISABLED = True
            logger.warning('GDELT repeatedly unavailable — news sentiment disabled '
                           'for the rest of this run (zero-filled).')
        return None


# ── Feature frame assembly ─────────────────────────────────────────────────────

def build_feature_frame(ohlcv: pd.DataFrame, ticker: str,
                        include_macro: bool | None = None,
                        include_sentiment: bool | None = None) -> pd.DataFrame | None:
    """
    Turn a raw OHLCV DataFrame into the model's per-day feature matrix.

    Returns a DataFrame [date × ALL_FEATURES] with the indicator warm-up
    period removed, or None when there isn't enough history.  Macro and
    sentiment columns are zero-filled wherever unavailable.

    include_macro / include_sentiment default to True but honour the
    ML_SKIP_MACRO / ML_SKIP_SENTIMENT env vars when left unset.
    """
    if include_macro is None:
        include_macro = not os.getenv('ML_SKIP_MACRO')
    if include_sentiment is None:
        include_sentiment = not os.getenv('ML_SKIP_SENTIMENT')

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
        # GDELT buckets articles by UTC day, so day t's tone includes stories
        # published after that day's close. Lag it a day so a window ending on
        # day t only sees news that was out before the close it predicts from.
        tone = senti['f_news_tone'].shift(1, freq='D').reindex(out.index).ffill()
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
