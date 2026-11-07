"""
Options strategy backtester.

Backtests single- and multi-leg option strategies (with or without stock) on
an underlying's real daily history: it opens the position, marks it to market
every day, applies the exit rules, rolls into the next one, and reports the
equity curve, every trade and the risk statistics.

Honest about its prices
-----------------------
Free historical option quotes do not exist, so option prices are *modelled*,
not traded quotes. Every price comes from Black-Scholes on that day's close:

- implied volatility: from the VIX for SPY. For any other ticker, its own
  21-day realized volatility scaled by the market's implied-to-realized ratio
  that day (ATM-adjusted VIX / SPY realized volatility, clipped to 0.8-2.0),
  so IV rises in fearful markets as it does in real chains;
- at-the-money IV of 0.87 x the VIX for SPY (the VIX includes the skew);
- a term structure: longer-dated IV higher than 30-day IV in calm markets,
  lower in panics;
- a volatility skew: out-of-the-money puts priced richer and calls cheaper,
  steeper on the put side, by standardized moneyness (optional);
- the risk-free rate from the 3-month Treasury bill (FRED DGS3MO);
- no dividends, and European exercise (no early assignment).

Costs are charged on every option and stock trade: a per-contract commission,
and a bid/ask cost as a share of each option's price (with a one-cent floor).
Options held to expiry settle at intrinsic value with no spread cost.

The result is labelled "model-priced" everywhere it is shown.
"""

from __future__ import annotations

import logging
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

import features as feat
import stats as st

logger = logging.getLogger(__name__)

CONTRACT = 100                      # shares per option contract
TRADING_DAYS = 252
RV_WINDOW = 21                      # trading days in the realized-vol estimate
VRP_CLIP = (0.8, 2.0)               # implied/realized ratio bounds
IV_CLIP = (0.05, 2.5)
# Skew per unit of standardized moneyness, steeper on the put side, as in index
# chains: a 16-delta put trades near 1.4x at-the-money IV, a 16-delta call near 0.85x.
SKEW_PUT, SKEW_CALL = 0.40, 0.15
SKEW_CLIP = (0.75, 2.0)
VIX_TO_ATM = 0.87                   # the VIX includes the skew; 30-day ATM IV sits ~13% below it
UNBOUNDED_MOVE = 0.30               # cash reserved against an uncovered short call: a 30% rise
MAX_LEGS = 6
MAX_YEARS = 15
INDEX_ETFS = {'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'SLV', 'XLF', 'XLE', 'XLK', 'EEM', 'HYG'}


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(kind: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European call/put value per share. T in years; T <= 0 gives intrinsic value."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if kind == 'call' else max(0.0, K - S)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sq
    d2 = d1 - sq
    if kind == 'call':
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def bs_delta(kind: str, S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        if kind == 'call':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _ncdf(d1) if kind == 'call' else _ncdf(d1) - 1.0


def term_iv(iv30: float, T: float) -> float:
    """ATM IV at maturity T from the 30-day level: upward sloping in calm markets, inverted in panics.

    The slope exponent falls as volatility rises (about +0.10 at 15% IV, flat near 40%,
    negative above), matching how the VIX term structure moves between contango and
    backwardation.
    """
    if T <= 0:
        return iv30
    beta = min(max(0.10 - 0.4 * (iv30 - 0.15), -0.15), 0.15)
    return iv30 * (max(T, 1 / 365) / (30 / 365)) ** beta


def skewed_iv(iv_atm: float, S: float, K: float, T: float, skew: bool = True) -> float:
    """IV for strike K and maturity T: the term structure, then the skew by standardized moneyness."""
    iv_atm = term_iv(iv_atm, T)
    if not skew or T <= 0:
        return iv_atm
    m = math.log(K / S) / (iv_atm * math.sqrt(max(T, 1 / 365)))
    factor = 1.0 - (SKEW_PUT if m < 0 else SKEW_CALL) * m
    return iv_atm * min(max(factor, SKEW_CLIP[0]), SKEW_CLIP[1])


# ── Strikes and expiries ──────────────────────────────────────────────────────

def strike_step(S: float, ticker: str = '') -> float:
    """Listed strike spacing, approximately: index ETFs list $1 strikes."""
    if ticker in INDEX_ETFS:
        return 1.0 if S >= 25 else 0.5
    if S < 25:
        return 0.5
    if S < 100:
        return 1.0
    if S < 250:
        return 2.5
    return 5.0


def default_spread(ticker: str) -> float:
    """Bid/ask cost per side as a share of an option's price: index ETFs trade far tighter."""
    return 0.01 if ticker.upper() in INDEX_ETFS else 0.03


def round_strike(K: float, step: float) -> float:
    return round(max(step, round(K / step) * step), 4)


def expiry_for(day: date, dte: int) -> date:
    """The Friday nearest to day + dte, always after day (options expire on Fridays)."""
    target = day + timedelta(days=max(1, dte))
    friday = target + timedelta(days=(4 - target.weekday()) % 7)
    back = friday - timedelta(days=7)
    if back > day and (target - back) < (friday - target):
        friday = back
    return friday


# ── Strategy definitions ──────────────────────────────────────────────────────
# A leg: {"type": "call"|"put"|"stock", "side": "long"|"short", "qty": int,
#         "strike": {"mode": "atm"} | {"mode": "pct", "value": +5}  (percent from the price)
#                 | {"mode": "delta", "value": 0.30}   (|delta| of this option)
#                 | {"mode": "offset", "leg": i, "value": -2.5}  (percent of the price from leg i's strike),
#         "dte": optional per-leg days to expiry (calendars)}

def _L(kind, side, strike=None, qty=1, dte=None):
    leg = {'type': kind, 'side': side, 'qty': qty}
    if kind != 'stock':
        leg['strike'] = strike or {'mode': 'atm'}
    if dte:
        leg['dte'] = dte
    return leg


ATM = {'mode': 'atm'}

PRESETS = {
    # one leg
    'long_call':  {'name': 'Long call', 'group': 'Single leg', 'view': 'Bullish',
                   'legs': [_L('call', 'long', ATM)], 'alloc': 10},
    'long_put':   {'name': 'Long put', 'group': 'Single leg', 'view': 'Bearish',
                   'legs': [_L('put', 'long', ATM)], 'alloc': 10},
    'short_put':  {'name': 'Cash-secured put', 'group': 'Single leg', 'view': 'Neutral to bullish',
                   'legs': [_L('put', 'short', {'mode': 'delta', 'value': 0.30})], 'alloc': 100},
    'short_call': {'name': 'Naked short call', 'group': 'Single leg', 'view': 'Neutral to bearish',
                   'legs': [_L('call', 'short', {'mode': 'delta', 'value': 0.20})], 'alloc': 50},
    # two legs
    'covered_call':    {'name': 'Covered call', 'group': 'Two legs', 'view': 'Neutral to bullish',
                        'legs': [_L('stock', 'long'), _L('call', 'short', {'mode': 'delta', 'value': 0.30})],
                        'alloc': 100},
    'protective_put':  {'name': 'Protective put', 'group': 'Two legs', 'view': 'Bullish, hedged',
                        'legs': [_L('stock', 'long'), _L('put', 'long', {'mode': 'pct', 'value': -5})],
                        'alloc': 100},
    'bull_call_spread': {'name': 'Bull call spread', 'group': 'Two legs', 'view': 'Moderately bullish',
                         'legs': [_L('call', 'long', ATM), _L('call', 'short', {'mode': 'pct', 'value': 5})],
                         'alloc': 10},
    'bear_put_spread':  {'name': 'Bear put spread', 'group': 'Two legs', 'view': 'Moderately bearish',
                         'legs': [_L('put', 'long', ATM), _L('put', 'short', {'mode': 'pct', 'value': -5})],
                         'alloc': 10},
    'bull_put_spread':  {'name': 'Bull put spread', 'group': 'Two legs', 'view': 'Neutral to bullish',
                         'legs': [_L('put', 'short', {'mode': 'delta', 'value': 0.30}),
                                  _L('put', 'long', {'mode': 'offset', 'leg': 0, 'value': -5})],
                         'alloc': 10},
    'bear_call_spread': {'name': 'Bear call spread', 'group': 'Two legs', 'view': 'Neutral to bearish',
                         'legs': [_L('call', 'short', {'mode': 'delta', 'value': 0.30}),
                                  _L('call', 'long', {'mode': 'offset', 'leg': 0, 'value': 5})],
                         'alloc': 10},
    'long_straddle':   {'name': 'Long straddle', 'group': 'Two legs', 'view': 'Big move, either way',
                        'legs': [_L('call', 'long', ATM), _L('put', 'long', ATM)], 'alloc': 10},
    'short_straddle':  {'name': 'Short straddle', 'group': 'Two legs', 'view': 'Quiet market',
                        'legs': [_L('call', 'short', ATM), _L('put', 'short', ATM)], 'alloc': 50},
    'long_strangle':   {'name': 'Long strangle', 'group': 'Two legs', 'view': 'Big move, either way',
                        'legs': [_L('call', 'long', {'mode': 'delta', 'value': 0.25}),
                                 _L('put', 'long', {'mode': 'delta', 'value': 0.25})], 'alloc': 10},
    'short_strangle':  {'name': 'Short strangle', 'group': 'Two legs', 'view': 'Quiet market',
                        'legs': [_L('call', 'short', {'mode': 'delta', 'value': 0.16}),
                                 _L('put', 'short', {'mode': 'delta', 'value': 0.16})], 'alloc': 50},
    'calendar_call':   {'name': 'Call calendar', 'group': 'Two legs', 'view': 'Quiet near term',
                        'legs': [_L('call', 'short', ATM), _L('call', 'long', ATM, dte=60)], 'alloc': 10},
    # three legs
    'collar':          {'name': 'Protective collar', 'group': 'Three legs', 'view': 'Bullish, hedged',
                        'legs': [_L('stock', 'long'), _L('put', 'long', {'mode': 'pct', 'value': -5}),
                                 _L('call', 'short', {'mode': 'pct', 'value': 5})], 'alloc': 100},
    'call_butterfly':  {'name': 'Call butterfly', 'group': 'Three legs', 'view': 'Pinned near the price',
                        'legs': [_L('call', 'long', {'mode': 'pct', 'value': -5}),
                                 _L('call', 'short', ATM, qty=2),
                                 _L('call', 'long', {'mode': 'pct', 'value': 5})], 'alloc': 10},
    # four legs
    'iron_condor':     {'name': 'Iron condor', 'group': 'Four legs', 'view': 'Range-bound',
                        'legs': [_L('put', 'short', {'mode': 'delta', 'value': 0.16}),
                                 _L('put', 'long', {'mode': 'offset', 'leg': 0, 'value': -2.5}),
                                 _L('call', 'short', {'mode': 'delta', 'value': 0.16}),
                                 _L('call', 'long', {'mode': 'offset', 'leg': 2, 'value': 2.5})], 'alloc': 10},
    'iron_butterfly':  {'name': 'Iron butterfly', 'group': 'Four legs', 'view': 'Pinned near the price',
                        'legs': [_L('put', 'short', ATM), _L('call', 'short', ATM),
                                 _L('put', 'long', {'mode': 'pct', 'value': -5}),
                                 _L('call', 'long', {'mode': 'pct', 'value': 5})], 'alloc': 10},
}


# ── Plain-English descriptions ────────────────────────────────────────────────

_NUM_WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}


def _strike_words(leg: dict, legs: list) -> str:
    s = leg.get('strike') or ATM
    mode, v = s.get('mode'), s.get('value')
    if mode == 'atm' or (mode == 'pct' and not v):
        return 'at the money'
    if mode == 'pct':
        return f"{abs(v):g}% {'above' if v > 0 else 'below'} the price"
    if mode == 'delta':
        return f'at {v:.2f} delta'
    if mode == 'offset':
        ref = legs[s['leg']]
        return (f"{abs(v):g}% of the price {'above' if v > 0 else 'below'} the "
                f"{ref['side']} {ref['type']}'s strike")
    return ''


def describe_leg(leg: dict, legs: list, dte: int) -> str:
    q = int(leg.get('qty', 1))
    if leg['type'] == 'stock':
        return f"{leg['side']} {q * CONTRACT} shares"
    days = leg.get('dte') or dte
    return (f"{leg['side']} {q} {leg['type']}{'s' if q > 1 else ''} {_strike_words(leg, legs)}, "
            f"{days} days to expiry")


def describe_strategy(name: str, legs: list, dte: int) -> dict:
    """'Protective collar: a three-leg backtest — long 100 shares, long 1 put 5% below ...'."""
    n = len(legs)
    word = _NUM_WORDS.get(n, str(n))
    parts = [describe_leg(leg, legs, dte) for leg in legs]
    kinds = []
    for leg in legs:
        k = f"{leg['side']} {'stock' if leg['type'] == 'stock' else leg['type']}"
        if k not in kinds:
            kinds.append(k)
    return {
        'legs_count': n,
        'headline': f"{name}: a {word}-leg backtest" + (f" ({', '.join(kinds)})" if n > 1 else f" ({kinds[0]})"),
        'legs': parts,
        'sentence': f"{name}: a {word}-leg backtest — " + '; '.join(parts) + '.',
    }


# ── Validation ────────────────────────────────────────────────────────────────

def clean_legs(raw) -> list:
    """Validate a user-built list of legs. Raises ValueError with a readable message."""
    if not isinstance(raw, list) or not raw:
        raise ValueError('add at least one leg')
    if len(raw) > MAX_LEGS:
        raise ValueError(f'at most {MAX_LEGS} legs')
    legs = []
    for i, leg in enumerate(raw):
        if not isinstance(leg, dict):
            raise ValueError(f'leg {i + 1} is not valid')
        kind, side = leg.get('type'), leg.get('side')
        if kind not in ('call', 'put', 'stock') or side not in ('long', 'short'):
            raise ValueError(f'leg {i + 1}: type must be call, put or stock, side long or short')
        try:
            qty = int(leg.get('qty', 1))
        except (TypeError, ValueError):
            raise ValueError(f'leg {i + 1}: quantity must be a whole number')
        if not 1 <= qty <= 10:
            raise ValueError(f'leg {i + 1}: quantity must be 1-10')
        out = {'type': kind, 'side': side, 'qty': qty}
        if kind == 'stock':
            if side == 'short':
                raise ValueError('short stock legs are not supported')
            legs.append(out)
            continue
        s = leg.get('strike') or ATM
        mode = s.get('mode') if isinstance(s, dict) else None
        if mode == 'atm':
            out['strike'] = ATM
        elif mode in ('pct', 'delta', 'offset'):
            try:
                v = float(s.get('value'))
            except (TypeError, ValueError):
                raise ValueError(f'leg {i + 1}: strike value must be a number')
            if mode == 'pct' and not -50 <= v <= 50:
                raise ValueError(f'leg {i + 1}: strike must be within 50% of the price')
            if mode == 'delta' and not 0.02 <= v <= 0.98:
                raise ValueError(f'leg {i + 1}: delta must be between 0.02 and 0.98')
            if mode == 'offset':
                ref = s.get('leg')
                if not isinstance(ref, int) or not 0 <= ref < i or legs[ref]['type'] == 'stock':
                    raise ValueError(f'leg {i + 1}: an offset must refer to an earlier option leg')
                if not -50 <= v <= 50 or v == 0:
                    raise ValueError(f'leg {i + 1}: offset must be a non-zero percent within 50')
                out['strike'] = {'mode': 'offset', 'leg': ref, 'value': v}
            else:
                out['strike'] = {'mode': mode, 'value': v}
        else:
            raise ValueError(f'leg {i + 1}: strike mode must be atm, pct, delta or offset')
        if leg.get('dte') is not None:
            try:
                d = int(leg['dte'])
            except (TypeError, ValueError):
                raise ValueError(f'leg {i + 1}: days to expiry must be a whole number')
            if not 1 <= d <= 365:
                raise ValueError(f'leg {i + 1}: days to expiry must be 1-365')
            out['dte'] = d
        legs.append(out)
    if all(leg['type'] == 'stock' for leg in legs):
        raise ValueError('add at least one option leg')
    return legs


# ── Market inputs ─────────────────────────────────────────────────────────────

def realized_vol(close: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    r = np.log(close).diff()
    return r.rolling(window, min_periods=max(10, window // 2)).std() * math.sqrt(TRADING_DAYS)


def implied_vol_series(close: pd.Series, ticker: str, vix: pd.Series | None,
                       spy_close: pd.Series | None) -> tuple[pd.Series, str]:
    """ATM implied volatility per day, and a sentence saying how it was estimated."""
    rv = realized_vol(close)
    if vix is not None and len(vix):
        v = (vix / 100.0 * VIX_TO_ATM).reindex(close.index).ffill()
        if ticker == 'SPY':
            return v.clip(*IV_CLIP), '0.87 × the VIX (the VIX includes the skew, so at-the-money IV runs below it)'
        if spy_close is not None and len(spy_close):
            rv_spy = realized_vol(spy_close.reindex(close.index).ffill())
            ratio = (v / rv_spy).clip(*VRP_CLIP)
            iv = (rv * ratio).clip(*IV_CLIP)
            return iv, (f'{ticker}’s 21-day realized volatility × the market’s implied-to-realized '
                        'ratio that day (0.87 × VIX ÷ SPY realized volatility)')
    return (rv * 1.1).clip(*IV_CLIP), (f'{ticker}’s 21-day realized volatility × 1.1 '
                                       '(the VIX was unavailable)')


_FRED_URL = 'https://fred.stlouisfed.org/graph/fredgraph.csv'
_FRED_CACHE: dict = {}              # series id -> (fetched_at, Series covering 1990 to that day)
_FRED_TTL = 6 * 3600


def _fred(series_id: str, lo, hi) -> pd.Series | None:
    """A daily FRED series for [lo, hi]: one request for the whole history, cached for six hours.

    Kept fast and bounded so a backtest never waits on FRED for long (the server's request
    timeout is 120 s); when FRED is down the caller falls back (realized volatility, 3% rate).
    """
    import io
    import time

    import requests
    hit = _FRED_CACHE.get(series_id)
    if hit is None or time.time() - hit[0] > _FRED_TTL:
        series = None
        for attempt in range(2):
            try:
                resp = requests.get(_FRED_URL, params={'id': series_id, 'cosd': '1990-01-01'}, timeout=15)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text), na_values='.')
                series = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0])).dropna()
                break
            except Exception as exc:      # pragma: no cover - network
                logger.warning('FRED %s attempt %d failed: %s', series_id, attempt + 1, exc)
        if series is None or series.empty:
            return None
        hit = (time.time(), series)
        _FRED_CACHE[series_id] = hit
    s = hit[1]
    return s[(s.index >= pd.Timestamp(lo)) & (s.index <= pd.Timestamp(hi))]


def load_market(ticker: str, start: str, end: str) -> dict:
    """Closes for the ticker (and SPY), the VIX and T-bill rates, with a warm-up for volatility."""
    lo = (pd.Timestamp(start) - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
    df = feat.fetch_ohlcv(ticker, start=lo, end=end)
    if df is None or df.empty:
        return {'error': f'No price history for {ticker}'}
    close = df['Close'].astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    spy = None
    if ticker != 'SPY':
        sdf = feat.fetch_ohlcv('SPY', start=lo, end=end)
        if sdf is not None and not sdf.empty:
            spy = sdf['Close'].astype(float)
            spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    vix = _fred('VIXCLS', lo, end)
    rate = _fred('DGS3MO', lo, end)
    return {'close': close, 'spy': spy, 'vix': vix, 'rate': rate}


# ── Position mechanics ───────────────────────────────────────────────────────

def _sign(leg):
    return 1 if leg['side'] == 'long' else -1


def build_position(legs: list, day: date, S: float, iv_atm: float, r: float, dte: int,
                   ticker: str, skew: bool) -> list:
    """Concrete legs (strike, expiry, entry price) for a position opened on `day` at price S."""
    step = strike_step(S, ticker)
    built = []
    for leg in legs:
        if leg['type'] == 'stock':
            built.append({**leg, 'strike': None, 'expiry': None, 'price': S})
            continue
        exp = expiry_for(day, leg.get('dte') or dte)
        T = (exp - day).days / 365.0
        s = leg['strike']
        if s['mode'] == 'atm':
            K = round_strike(S, step)
        elif s['mode'] == 'pct':
            K = round_strike(S * (1 + s['value'] / 100.0), step)
        elif s['mode'] == 'offset':
            base = built[s['leg']]['strike']
            K = round_strike(base + S * s['value'] / 100.0, step)
            if K == base:                       # keep a spread a spread
                K = base + (step if s['value'] > 0 else -step)
        else:                                   # delta
            want = s['value']
            grid = [round_strike(S * (1 + p / 200.0), step) for p in range(-100, 101)]
            grid = sorted(set(k for k in grid if k > 0))
            K = min(grid, key=lambda k: abs(abs(bs_delta(leg['type'], S, k, T, r,
                                                         skewed_iv(iv_atm, S, k, T, skew))) - want))
        iv = skewed_iv(iv_atm, S, K, T, skew)
        built.append({**leg, 'strike': float(K), 'expiry': exp,
                      'price': bs_price(leg['type'], S, K, T, r, iv)})
    return built


def leg_value(leg: dict, day: date, S: float, iv_atm: float, r: float, skew: bool) -> float:
    """Per-share value of one leg on `day`."""
    if leg['type'] == 'stock':
        return S
    T = max(0.0, (leg['expiry'] - day).days / 365.0)
    return bs_price(leg['type'], S, leg['strike'], T, r, skewed_iv(iv_atm, S, leg['strike'], T, skew))


def position_value(pos: list, day, S, iv_atm, r, skew) -> float:
    """Value of one unit of the position (dollars), long legs positive, short negative."""
    return sum(_sign(leg) * leg['qty'] * CONTRACT * leg_value(leg, day, S, iv_atm, r, skew) for leg in pos)


def payoff_at_expiry(pos: list, x: float) -> float:
    """P/L of one unit if the underlying is at x when the first option expires (stock marked at x)."""
    total = 0.0
    for leg in pos:
        if leg['type'] == 'stock':
            val = x
        elif leg['type'] == 'call':
            val = max(0.0, x - leg['strike'])
        else:
            val = max(0.0, leg['strike'] - x)
        total += _sign(leg) * leg['qty'] * CONTRACT * (val - leg['price'])
    return total


def capital_required(pos: list, S: float) -> tuple[float, bool]:
    """Cash one unit ties up: its worst loss at expiry, or a 30% rally for uncovered short calls."""
    calls_up = sum(_sign(leg) * leg['qty'] for leg in pos if leg['type'] in ('call', 'stock'))
    unbounded = calls_up < 0
    top = S * (1 + UNBOUNDED_MOVE) if unbounded else S * 3
    strikes = sorted({leg['strike'] for leg in pos if leg['strike']} | {0.0, S, top})
    worst = min(payoff_at_expiry(pos, x) for x in strikes if x <= top)
    debit = sum(_sign(leg) * leg['qty'] * CONTRACT * leg['price'] for leg in pos)
    return max(-worst, debit, 1.0), unbounded


def trade_cost(pos: list, prices: list, spread_pct: float, commission: float,
               stock_slip: float, settle: bool = False) -> float:
    """Commission plus bid/ask cost for opening or closing one unit at `prices` (per share)."""
    cost = 0.0
    for leg, px in zip(pos, prices):
        n = leg['qty']
        if leg['type'] == 'stock':
            cost += n * CONTRACT * px * stock_slip
        elif not settle:
            cost += n * (commission + (CONTRACT * max(0.01, px * spread_pct) if spread_pct > 0 else 0.0))
    return cost


# ── The backtest ──────────────────────────────────────────────────────────────

def run(ticker: str, start: str, end: str, legs: list, name: str = 'Custom strategy',
        dte: int = 30, initial_capital: float = 100_000, alloc_pct: float = 100,
        exit_dte: int | None = None, profit_take: float | None = None, stop_loss: float | None = None,
        spread_pct: float | None = None, commission: float = 0.65, stock_slip: float = 0.0003,
        skew: bool = True, market: dict | None = None) -> dict:
    """Backtest `legs` on `ticker` between start and end. See the module docstring."""
    ticker = ticker.upper()
    if spread_pct is None:
        spread_pct = default_spread(ticker)
    m = market or load_market(ticker, start, end)
    if 'error' in m:
        return m
    close = m['close']
    iv_all, iv_note = implied_vol_series(close, ticker, m.get('vix'), m.get('spy'))
    rate = m.get('rate')
    r_all = ((rate / 100.0).reindex(close.index).ffill().bfill() if rate is not None and len(rate)
             else pd.Series(0.03, index=close.index))
    r_all = r_all.fillna(0.03)

    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
    days = close.index[mask]
    if len(days) < 30:
        return {'error': 'Need at least 30 trading days in the window'}

    cash = float(initial_capital)
    pos, units, entry = None, 0, None
    equity, trades, notes = [], [], []
    total_costs = 0.0
    first_pos = None
    days_in = 0
    bench_shares = initial_capital / float(close[days[0]])
    last_day = days[-1].date()

    def close_position(day, S, iv, r, reason):
        nonlocal cash, pos, units, entry, total_costs
        settle = reason == 'expiry'
        prices = [leg_value(leg, day, S, iv, r, skew) for leg in pos]
        value = sum(_sign(leg) * leg['qty'] * CONTRACT * p for leg, p in zip(pos, prices))
        # Expiring options settle at intrinsic; any longer-dated legs are sold at the model price.
        cost = 0.0
        for leg, p in zip(pos, prices):
            done = settle and leg['type'] != 'stock' and leg['expiry'] <= day
            cost += trade_cost([leg], [p], spread_pct, commission, stock_slip, settle=done)
        cost *= units
        cash += units * value - cost
        total_costs += cost
        pnl = units * (value - entry['value']) - cost - entry['cost']
        trades.append({
            'opened': entry['day'].isoformat(), 'closed': day.isoformat(), 'reason': reason,
            'units': units, 'underlying_open': round(entry['S'], 2), 'underlying_close': round(S, 2),
            'legs': [{'side': leg['side'], 'type': leg['type'], 'qty': leg['qty'] * units,
                      'strike': leg['strike'], 'expiry': leg['expiry'].isoformat() if leg['expiry'] else None,
                      'open': round(leg['price'], 2), 'close': round(p, 2)} for leg, p in zip(pos, prices)],
            'net_open': round(units * entry['value'], 2), 'net_close': round(units * value, 2),
            'costs': round(cost + entry['cost'], 2), 'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / entry['basis'] * 100, 2) if entry['basis'] else None,
        })
        pos, units, entry = None, 0, None

    prev_day = None
    for ts in days:
        day = ts.date()
        S = float(close[ts])
        iv = float(iv_all.get(ts, np.nan))
        r = float(r_all.get(ts, 0.03))
        if not np.isfinite(iv):
            iv = float(iv_all.dropna().iloc[0]) if iv_all.notna().any() else 0.25
        if prev_day is not None and cash > 0:
            cash *= 1 + r * (day - prev_day).days / 365.0    # idle cash earns the T-bill rate
        prev_day = day

        if pos is not None:
            first_exp = min(leg['expiry'] for leg in pos if leg['expiry'])
            value = position_value(pos, day, S, iv, r, skew)
            pnl = units * (value - entry['value'])
            ref = abs(entry['value']) * units or 1.0
            if day >= first_exp:
                close_position(day, S, iv, r, 'expiry')
            elif exit_dte is not None and (first_exp - day).days <= exit_dte:
                close_position(day, S, iv, r, f'{exit_dte} days before expiry')
            elif profit_take is not None and pnl >= profit_take / 100.0 * ref:
                close_position(day, S, iv, r, f'profit target ({profit_take:g}%)')
            elif stop_loss is not None and pnl <= -stop_loss / 100.0 * ref:
                close_position(day, S, iv, r, f'stop loss ({stop_loss:g}%)')
            elif day == last_day:
                close_position(day, S, iv, r, 'end of test')

        remaining = (last_day - day).days
        if pos is None and day != last_day and remaining >= max(5, dte // 2):
            eq_now = cash
            cand = build_position(legs, day, S, iv, r, dte, ticker, skew)
            need, unbounded = capital_required(cand, S)
            n = int(max(0.0, eq_now) * alloc_pct / 100.0 // need)
            if n >= 1:
                open_value = position_value(cand, day, S, iv, r, skew)
                cost = n * trade_cost(cand, [leg['price'] for leg in cand], spread_pct, commission, stock_slip)
                cash -= n * open_value + cost
                total_costs += cost
                pos, units = cand, n
                entry = {'day': day, 'S': S, 'value': open_value, 'cost': cost, 'basis': n * need}
                if first_pos is None:
                    first_pos = {'day': day.isoformat(), 'S': S, 'legs': cand, 'need': need,
                                 'unbounded': unbounded, 'units': n}
            elif not notes:
                notes.append(f'On {day.isoformat()} one unit needed ${need:,.0f} of capital, more than '
                             f'{alloc_pct:g}% of the account; positions are skipped until it fits.')

        held = pos is not None
        days_in += held
        val = cash + (units * position_value(pos, day, S, iv, r, skew) if held else 0.0)
        equity.append({'date': day.isoformat(), 'value': round(val, 2),
                       'benchmark': round(bench_shares * S, 2), 'iv': round(iv * 100, 1)})
        if val <= 0:
            notes.append(f'The account was wiped out on {day.isoformat()}.')
            break

    rf_daily = float(r_all[days].mean()) / TRADING_DAYS
    return _summarize(ticker, name, legs, dte, initial_capital, equity, trades, total_costs,
                      days_in, first_pos, notes, iv_note, alloc_pct, exit_dte, profit_take, stop_loss,
                      spread_pct, commission, skew, rf_daily)


def _metrics(values: np.ndarray, days: int, rf_daily: float = 0.0) -> dict:
    rets = np.diff(values) / values[:-1]
    years = max(days / TRADING_DAYS, 1 / TRADING_DAYS)
    total = values[-1] / values[0] - 1
    cagr = (values[-1] / values[0]) ** (1 / years) - 1 if values[-1] > 0 else -1.0
    vol = float(rets.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
    ex = rets - rf_daily
    sd = ex.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = float(ex.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    peak = np.maximum.accumulate(values)
    dd = float(((values - peak) / peak).min())
    psr = st.probabilistic_sharpe_ratio(ex) if len(rets) >= 10 else float('nan')
    return {'total_return': round(total * 100, 2), 'cagr': round(cagr * 100, 2),
            'volatility': round(vol * 100, 2), 'sharpe': round(sharpe, 2),
            'max_drawdown': round(dd * 100, 2),
            'psr': None if not np.isfinite(psr) else round(psr * 100, 1),
            'final_value': round(float(values[-1]), 2)}


def _summarize(ticker, name, legs, dte, capital, equity, trades, total_costs, days_in, first_pos,
               notes, iv_note, alloc_pct, exit_dte, profit_take, stop_loss, spread_pct, commission, skew,
               rf_daily=0.0):
    if len(equity) < 2:
        return {'error': 'Not enough data to run the backtest'}
    vals = np.array([e['value'] for e in equity], dtype=float)
    bench = np.array([e['benchmark'] for e in equity], dtype=float)
    m = _metrics(vals, len(vals), rf_daily)
    b = _metrics(bench, len(bench), rf_daily)
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    m.update({
        'trades': len(trades), 'win_rate': round(len(wins) / len(trades) * 100, 1) if trades else None,
        'avg_win': round(float(np.mean(wins)), 2) if wins else None,
        'avg_loss': round(float(np.mean(losses)), 2) if losses else None,
        'best_trade': max((t['pnl'] for t in trades), default=None),
        'worst_trade': min((t['pnl'] for t in trades), default=None),
        'total_costs': round(total_costs, 2),
        'time_in_market': round(days_in / len(equity) * 100, 1),
    })
    desc = describe_strategy(name, legs, dte)
    example = None
    if first_pos:
        p = first_pos['legs']
        S0 = first_pos['S']
        xs = np.linspace(S0 * 0.7, S0 * 1.3, 61)
        example = {
            'date': first_pos['day'], 'price': round(S0, 2), 'units': first_pos['units'],
            'capital_per_unit': round(first_pos['need'], 2), 'unbounded': first_pos['unbounded'],
            'legs': [{'side': leg['side'], 'type': leg['type'], 'qty': leg['qty'], 'strike': leg['strike'],
                      'expiry': leg['expiry'].isoformat() if leg['expiry'] else None,
                      'price': round(leg['price'], 2)} for leg in p],
            'payoff': [{'x': round(float(x), 2), 'pnl': round(payoff_at_expiry(p, float(x)), 2)} for x in xs],
        }
    exits = ['held to expiry'] if exit_dte is None else [f'closed {exit_dte} days before expiry']
    if profit_take is not None:
        exits.append(f'or at {profit_take:g}% profit')
    if stop_loss is not None:
        exits.append(f'or at a {stop_loss:g}% loss')
    return {
        'ticker': ticker,
        'strategy': {**desc, 'name': name, 'dte': dte, 'alloc_pct': alloc_pct,
                     'rules': f"Opens a new position the day the last one closes, {dte} days to expiry, "
                              f"sized to use up to {alloc_pct:g}% of the account; {' '.join(exits)}."},
        'metrics': m,
        'benchmark': {**b, 'label': f'Buy and hold {ticker}'},
        'equity': equity,
        'trades': trades,
        'example': example,
        'notes': notes,
        'model': {
            'priced': 'model',
            'iv': iv_note,
            'skew': skew,
            'costs': f'${commission:.2f} per contract plus {spread_pct * 100:g}% of each option’s price '
                     'per side for the bid/ask (at least one cent a share)',
            'limits': ['Option prices are reconstructed with Black-Scholes, not real quotes; the skew and '
                       'term structure are modelled shapes, so strategies that depend on them (calendars, '
                       'far out-of-the-money wings) are the least reliable.',
                       'No dividends; European exercise, so no early assignment.',
                       'Daily closes only: exits are checked once a day, at the close.'],
        },
    }


def run_preset(key: str, ticker: str, start: str, end: str, **kw) -> dict:
    p = PRESETS[key]
    kw.setdefault('alloc_pct', p['alloc'])
    return run(ticker, start, end, p['legs'], name=p['name'], **kw)


def presets_summary(dte: int = 30) -> list:
    """Every preset with its leg description, for the strategy picker."""
    out = []
    for key, p in PRESETS.items():
        d = describe_strategy(p['name'], p['legs'], dte)
        out.append({'key': key, 'name': p['name'], 'group': p['group'], 'view': p['view'],
                    'alloc': p['alloc'], 'legs': p['legs'], 'legs_count': d['legs_count'],
                    'headline': d['headline'], 'description': d['legs']})
    return out
