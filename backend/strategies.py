"""
Trading strategy signal generators.

Each strategy calls features.get_feature_df() to get a consistent,
fully-computed feature DataFrame, then applies its signal logic on top.

Adding the ML strategy
----------------------
1. Train your model and save it (e.g. joblib.dump(model, 'ml_model.pkl'))
2. Load it at module startup (see ml_signal below)
3. Replace the HOLD stub with: signal = model.predict(feature_row)[0]
4. The feature matrix is already built — features.FEATURE_COLS is the
   exact column list the model should be trained on.

signal return values:  'BUY' | 'SELL' | 'HOLD' | None  (None = data error)
"""

import logging
import pandas as pd
import features as feat

logger = logging.getLogger(__name__)

WATCHLIST = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'JPM', 'SPY']

VALID_STRATEGIES = ('ma_crossover', 'rsi', 'macd', 'ml', 'dip_buyer')

# Dip Buyer thresholds — match backtest.py (position within the 52-week range).
_DIP_BUY_THRESH  = 0.30
_DIP_SELL_THRESH = 0.80


# ── Strategy 1 — Moving Average Crossover ─────────────────────────────────────

def ma_crossover_signal(ticker: str) -> tuple[str | None, float | None]:
    df = feat.get_feature_df(ticker, period='6mo')
    if df is None or len(df) < 2:
        return None, None

    price  = float(df['Close'].iloc[-1])
    curr20 = float(df['sma20'].iloc[-1])
    curr50 = float(df['sma50'].iloc[-1])

    # Trend stance: long while the fast average is above the slow one.
    if curr20 > curr50:
        return 'BUY', price
    if curr20 < curr50:
        return 'SELL', price
    return 'HOLD', price


# ── Strategy 2 — RSI Mean Reversion ──────────────────────────────────────────

def rsi_signal(ticker: str) -> tuple[str | None, float | None]:
    df = feat.get_feature_df(ticker, period='6mo')
    if df is None or df.empty:
        return None, None

    price   = float(df['Close'].iloc[-1])
    rsi_val = float(df['rsi14'].iloc[-1])

    if rsi_val < 30:
        return 'BUY', price
    if rsi_val > 70:
        return 'SELL', price
    return 'HOLD', price


# ── Strategy 3 — MACD Momentum ────────────────────────────────────────────────

def macd_signal(ticker: str) -> tuple[str | None, float | None]:
    df = feat.get_feature_df(ticker, period='6mo')
    if df is None or len(df) < 2:
        return None, None

    price  = float(df['Close'].iloc[-1])
    curr_m = float(df['macd_line'].iloc[-1])
    curr_s = float(df['macd_signal'].iloc[-1])

    # Momentum stance: long while MACD is above its signal line.
    if curr_m > curr_s:
        return 'BUY', price
    if curr_m < curr_s:
        return 'SELL', price
    return 'HOLD', price


# ── Strategy 4 — ML transformer ───────────────────────────────────────────────
# A multi-modal transformer (price + indicators + macro + news sentiment)
# trained offline (see ml/README.md) and served via ONNX by ml_runtime.py.
# Predicts a return distribution, direction probability, and vol forecast
# over the next 5 trading days; trades only when direction AND the median
# of the return distribution agree.  Holds gracefully if no model file is
# deployed yet.

def ml_signal(ticker: str) -> tuple[str | None, float | None]:
    import ml_runtime
    if ml_runtime.is_available():
        return ml_runtime.live_signal(ticker)

    # No trained model deployed — behave like a HOLD strategy, don't crash.
    df = feat.get_feature_df(ticker, period='6mo')
    if df is None or df.empty:
        return None, None
    price = float(df['Close'].iloc[-1])
    logger.info('ML model not deployed — HOLD for %s (train with ml/train.py)', ticker)
    return 'HOLD', price


# ── Strategy 5 — Dip Buyer (52-week value) ────────────────────────────────────
# Buys when a stock is near its 52-week low and sells once it recovers toward the
# high. The backtest engine adds patient, reserve-keeping tranche sizing on top;
# this just reports the current stance for the live scanner.

def dip_buyer_signal(ticker: str) -> tuple[str | None, float | None]:
    df = feat.get_feature_df(ticker, period='1y')
    if df is None or len(df) < 40:
        return None, None
    price = float(df['Close'].iloc[-1])
    low   = float(df['Close'].rolling(252, min_periods=40).min().iloc[-1])
    high  = float(df['Close'].rolling(252, min_periods=40).max().iloc[-1])
    if high > low:
        pct = (price - low) / (high - low)
        if pct < _DIP_BUY_THRESH:
            return 'BUY', price
        if pct > _DIP_SELL_THRESH:
            return 'SELL', price
    return 'HOLD', price


# ── Dispatcher ────────────────────────────────────────────────────────────────

def get_signal(strategy: str, ticker: str) -> tuple[str | None, float | None]:
    dispatch = {
        'ma_crossover': ma_crossover_signal,
        'rsi':          rsi_signal,
        'macd':         macd_signal,
        'ml':           ml_signal,
        'dip_buyer':    dip_buyer_signal,
    }
    fn = dispatch.get(strategy)
    if fn is None:
        logger.error('Unknown strategy: %s', strategy)
        return None, None
    return fn(ticker)


# ── Indicator snapshot for dashboard ─────────────────────────────────────────

def get_indicator_data(ticker: str, strategy: str) -> dict:
    """Returns current indicator values for the watchlist table."""
    df = feat.get_feature_df(ticker, period='6mo')
    if df is None or df.empty:
        return {}

    result = {'price': round(float(df['Close'].iloc[-1]), 2)}

    if strategy == 'ma_crossover':
        result['sma20'] = round(float(df['sma20'].iloc[-1]), 2)
        result['sma50'] = round(float(df['sma50'].iloc[-1]), 2)

    elif strategy == 'rsi':
        result['rsi'] = round(float(df['rsi14'].iloc[-1]), 2)

    elif strategy == 'macd':
        result['macd']      = round(float(df['macd_line'].iloc[-1]),   4)
        result['signal']    = round(float(df['macd_signal'].iloc[-1]), 4)
        result['histogram'] = round(float(df['macd_hist'].iloc[-1]),   4)

    elif strategy == 'ml':
        # Expose full feature vector so a future ML dashboard can display it
        row = df.iloc[-1]
        for col in feat.FEATURE_COLS:
            if col in df.columns:
                val = row[col]
                if pd.notna(val):
                    result[col] = round(float(val), 4)

    return result
