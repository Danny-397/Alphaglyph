"""
Centralized feature engineering layer.

All three existing strategies consume this module instead of computing
indicators inline.  When the ML strategy is added, it will call
get_feature_df() to get a ready-made feature matrix — no duplication,
no inconsistency between training data and live signals.

Public API
----------
get_feature_df(ticker, period, start, end)  → DataFrame | None
    Main entry point.  Returns OHLCV + all indicator columns, NaN rows
    (indicator warmup period) already dropped.

fetch_ohlcv(ticker, period, start, end)     → DataFrame | None
    Raw OHLCV download from yfinance.  Used by the backtest engine when
    it needs data without indicator computation.

compute_features(df)                         → DataFrame
    Adds indicator columns to any OHLCV DataFrame in-place copy.
    Called by get_feature_df() and by the backtest engine.

FEATURE_COLS
    Ordered list of every feature column an ML model should consume.
    Keep this list in sync whenever new indicators are added.
"""

import io
import os
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    # Internal per-ticker error registry. Newer yfinance swallows download
    # errors (incl. rate limits) and returns an empty frame instead of raising,
    # so we inspect this dict to tell "rate limited" apart from "invalid ticker".
    from yfinance import shared as _yf_shared
except Exception:        # pragma: no cover
    _yf_shared = None

logger = logging.getLogger(__name__)

# In-process OHLCV cache: key -> (fetched_at, DataFrame).
# Yahoo Finance rate-limits aggressively (especially from datacenter IPs like
# Render). Caching successful downloads for a short TTL drastically cuts the
# number of calls — a re-run of the same backtest serves entirely from cache.
_CACHE: dict = {}
_CACHE_TTL = 900          # 15 minutes
_MAX_ATTEMPTS = 3         # retries on transient rate-limit errors

# Approximate calendar days per yfinance period string (for the Stooq fallback,
# which works in date ranges). Generous so indicators have enough warm-up.
_PERIOD_DAYS = {
    '5d': 10, '1mo': 45, '3mo': 130, '6mo': 220,
    '1y': 400, '2y': 760, '5y': 1850, 'ytd': 400, 'max': 4000,
}


def _is_rate_limit(exc) -> bool:
    s = str(exc).lower()
    return 'rate' in s or 'too many' in s or '429' in s


# ── Source 0: Tiingo (primary when TIINGO_API_KEY is set — cloud-reliable) ────

_TIINGO_BASE = 'https://api.tiingo.com/tiingo/daily'


def _try_tiingo(ticker, period, start, end):
    """
    Tiingo end-of-day prices. Free tier, reliable from datacenter IPs.
    Requires the TIINGO_API_KEY environment variable; returns ('skip') if unset.
    Returns (DataFrame | None, status) where status is 'ok'|'nodata'|'rate_limited'|'skip'.
    Uses adjusted OHLCV to mirror yfinance's auto_adjust=True.
    """
    token = os.getenv('TIINGO_API_KEY', '').strip()
    if not token:
        return None, 'skip'

    if start and end:
        s_date, e_date = start, end
    else:
        days = _PERIOD_DAYS.get(period, 220)
        e_date = datetime.now().strftime('%Y-%m-%d')
        s_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = f'{_TIINGO_BASE}/{ticker}/prices'
    params = {'startDate': s_date, 'endDate': e_date, 'token': token, 'format': 'json'}
    try:
        resp = requests.get(url, params=params, timeout=15,
                            headers={'Content-Type': 'application/json'})
        if resp.status_code == 404:
            return None, 'nodata'          # ticker not found — authoritative
        if resp.status_code == 429:
            return None, 'rate_limited'
        if resp.status_code in (401, 403):
            logger.warning('Tiingo: invalid or missing API key (HTTP %s)', resp.status_code)
            return None, 'skip'
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None, 'nodata'
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        df = df.set_index('date').sort_index()
        out = pd.DataFrame({
            'Open':   df['adjOpen'],
            'High':   df['adjHigh'],
            'Low':    df['adjLow'],
            'Close':  df['adjClose'],
            'Volume': df['adjVolume'],
        }).dropna()
        return (out, 'ok') if not out.empty else (None, 'nodata')
    except Exception as exc:
        logger.warning('Tiingo fetch failed for %s: %s', ticker, exc)
        return None, 'skip'


# ── Source 1: yfinance (with retry/backoff) ───────────────────────────────────

def _try_yfinance(ticker, period, interval, start, end):
    """Return (DataFrame | None, status) where status is 'ok'|'nodata'|'rate_limited'."""
    def _yahoo_error(sym):
        if _yf_shared is None:
            return ''
        try:
            return str(_yf_shared._ERRORS.get(sym, ''))
        except Exception:
            return ''

    for attempt in range(_MAX_ATTEMPTS):
        try:
            if start and end:
                df = yf.download(ticker, start=start, end=end,
                                 progress=False, auto_adjust=True)
            else:
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, auto_adjust=True)
        except Exception as exc:
            if _is_rate_limit(exc):
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                return None, 'rate_limited'
            logger.error('yfinance error for %s: %s', ticker, exc)
            return None, 'nodata'

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df, 'ok'

        if _is_rate_limit(_yahoo_error(ticker)):
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            return None, 'rate_limited'
        return None, 'nodata'
    return None, 'rate_limited'


# ── Source 2: Stooq fallback (free, no API key, lenient limits) ───────────────

def _try_stooq(ticker, period, start, end):
    """
    Free daily OHLCV from stooq.com — used when yfinance is rate-limited.
    Works from cloud IPs where Yahoo blocks. Daily resolution only.
    Returns a DataFrame or None.
    """
    if start and end:
        d1, d2 = start.replace('-', ''), end.replace('-', '')
    else:
        days = _PERIOD_DAYS.get(period, 220)
        d2 = datetime.now().strftime('%Y%m%d')
        d1 = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    url = (f'https://stooq.com/q/d/l/?s={ticker.lower()}.us'
           f'&d1={d1}&d2={d2}&i=d')
    try:
        resp = requests.get(url, timeout=15,
                            headers={'User-Agent': 'Mozilla/5.0 (TradeBot)'})
        resp.raise_for_status()
        text = resp.text.strip()
        # Stooq returns "No data" or an HTML error page for unknown symbols
        if not text or text.lower().startswith('<') or 'no data' in text.lower():
            return None
        df = pd.read_csv(io.StringIO(text))
        if df.empty or 'Close' not in df.columns or 'Date' not in df.columns:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        keep = [c for c in ('Open', 'High', 'Low', 'Close', 'Volume') if c in df.columns]
        df = df[keep].dropna()
        return df if not df.empty else None
    except Exception as exc:
        logger.warning('Stooq fallback failed for %s: %s', ticker, exc)
        return None


# ── Combined fetch with caching ───────────────────────────────────────────────

def _download(ticker: str, period: str, interval: str,
              start: str, end: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV with caching, trying yfinance first and Stooq as a fallback.

    Returns a DataFrame on success, None when the symbol genuinely has no data.
    Raises RuntimeError('rate_limited') only if BOTH sources are unavailable
    due to rate limiting.
    """
    key = (ticker, period, interval, start, end)
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1].copy()

    statuses = []

    def _cache_return(df):
        _CACHE[key] = (time.time(), df)
        return df.copy()

    # 1. Tiingo (primary when an API key is configured)
    df, st = _try_tiingo(ticker, period, start, end)
    if df is not None and not df.empty:
        return _cache_return(df)
    if st != 'skip':
        statuses.append(st)

    # 2. yfinance (free, no key, but heavily rate-limited)
    df, st = _try_yfinance(ticker, period, interval, start, end)
    if df is not None and not df.empty:
        return _cache_return(df)
    statuses.append(st)

    # 3. Stooq (free fallback, daily only)
    if interval == '1d':
        df = _try_stooq(ticker, period, start, end)
        if df is not None and not df.empty:
            return _cache_return(df)

    # Nothing returned data. A definitive 'nodata' (e.g. Tiingo 404) wins —
    # treat the symbol as invalid. Otherwise, if any source was rate-limited,
    # surface that so the UI can say "try again" rather than "doesn't exist".
    if 'nodata' in statuses:
        return None
    if 'rate_limited' in statuses:
        raise RuntimeError('rate_limited')
    return None


def validate_symbol(symbol: str) -> str:
    """
    Check whether a ticker exists on Yahoo Finance.

    Returns one of: 'valid', 'not_found', 'rate_limited'.
    """
    sym = (symbol or '').strip().upper()
    if not sym:
        return 'not_found'
    try:
        df = _download(sym, '1mo', '1d', None, None)
        return 'valid' if (df is not None and not df.empty) else 'not_found'
    except RuntimeError:
        return 'rate_limited'


# ── Feature column registry ────────────────────────────────────────────────────
# When you build the ML model, use df[FEATURE_COLS] as the input matrix.
# Add new indicators here and to compute_features() together.
FEATURE_COLS = [
    'close',
    'volume',
    'sma20',
    'sma50',
    'rsi14',
    'macd_line',
    'macd_signal',
    'macd_hist',
    'vol_ma20',
    'return_1d',
    'return_5d',
]

# Minimum bars needed for all indicators to be fully warmed up (SMA-50 is longest)
MIN_BARS = 55


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str = '6mo', interval: str = '1d',
                start: str = None, end: str = None) -> pd.DataFrame | None:
    """
    Download OHLCV data from yfinance (cached, with retry/backoff).

    The ticker is upper-cased so lowercase input ('aapl') still works.
    Returns None on no data or persistent rate-limiting (callers skip the symbol).
    """
    sym = (ticker or '').strip().upper()
    if not sym:
        return None
    try:
        return _download(sym, period, interval, start, end)
    except RuntimeError:
        logger.warning('Rate limited fetching %s — skipping', sym)
        return None


# ── Indicator computation ──────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicator columns to a raw OHLCV DataFrame.

    Does NOT drop NaN rows — callers decide whether to dropna or keep
    the warmup period for other purposes.
    """
    df = df.copy()
    close  = df['Close']
    volume = df['Volume']

    # ── Trend: simple moving averages ─────────────────────────────────────
    df['sma20'] = close.rolling(20).mean()
    df['sma50'] = close.rolling(50).mean()

    # ── Momentum: RSI (14-period, Wilder EMA method) ──────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=True, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=True, min_periods=14).mean()
    rs           = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi14']  = 100 - (100 / (1 + rs))
    # When avg_loss is 0 after warmup: pure uptrend → 100, no movement → 50
    warmed = avg_gain.notna() & avg_loss.notna()
    df.loc[warmed & (avg_loss == 0) & (avg_gain > 0),  'rsi14'] = 100.0
    df.loc[warmed & (avg_loss == 0) & (avg_gain == 0), 'rsi14'] = 50.0

    # ── Momentum: MACD (12 / 26 / 9) ─────────────────────────────────────
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    df['macd_line']   = ema12 - ema26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd_line'] - df['macd_signal']

    # ── Volume: rolling average ────────────────────────────────────────────
    df['vol_ma20'] = volume.rolling(20).mean()

    # ── Price returns ──────────────────────────────────────────────────────
    df['return_1d'] = close.pct_change(1)
    df['return_5d'] = close.pct_change(5)

    # ── Lowercase aliases (convenience for ML feature matrix) ─────────────
    df['close']  = close
    df['volume'] = volume

    return df


# ── Main entry point ───────────────────────────────────────────────────────────

def get_feature_df(ticker: str, period: str = '6mo',
                   start: str = None, end: str = None) -> pd.DataFrame | None:
    """
    Fetch OHLCV data and return a fully-featured DataFrame ready for signal
    generation or ML training.

    NaN rows (indicator warmup period) are dropped before returning.

    Usage
    -----
    Strategy signals:
        df = features.get_feature_df('AAPL')
        signal = 'BUY' if df['rsi14'].iloc[-1] < 30 else 'HOLD'

    ML training:
        df = features.get_feature_df('AAPL', start='2022-01-01', end='2024-01-01')
        X = df[features.FEATURE_COLS].values
        y = build_labels(df)          # your label function
        model.fit(X, y)

    ML live prediction:
        df = features.get_feature_df('AAPL')
        row = df[features.FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        signal = model.predict(row)[0]
    """
    raw = fetch_ohlcv(ticker, period=period, start=start, end=end)
    if raw is None or len(raw) < MIN_BARS:
        return None
    df = compute_features(raw)
    df.dropna(inplace=True)
    if df.empty:
        return None
    return df
