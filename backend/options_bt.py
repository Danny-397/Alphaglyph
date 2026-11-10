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
- at-the-money IV of VIX_TO_ATM x the VIX for SPY (the VIX includes the skew).
  The factor is fitted so that a replica of Cboe's PutWrite index matches the
  real index over 2007-2015, and checked on 2016 on (tools/options_validation.py);
- a term structure from the real VIX curve that day: the VIX9D, VIX, VIX3M and
  VIX6M give implied volatility at 9, 30, 93 and 183 days, interpolated in total
  variance. Where a point is missing (the VIX9D starts in 2011), a modelled
  slope fills in: upward in calm markets, inverted in panics;
- a volatility skew: out-of-the-money puts priced richer and calls cheaper, by
  standardized moneyness, fitted on 2007-2015 against Cboe's PPUT and BXMD
  indices and checked on 2016 on with four more (optional);
- the risk-free rate from the 3-month Treasury bill (FRED DGS3MO);
- dividends: options are priced on unadjusted prices with the trailing 12-month
  dividend yield (Black-Scholes-Merton), stock legs are paid each dividend in
  cash on its ex-date, and a short call is assigned early the day before an
  ex-date when the dividend is worth more than its remaining time value. Puts
  are treated as European. Without dividend data, prices are dividend-adjusted
  and the yield is zero, and the result says so.
- earnings (single stocks): an option open on a report's reaction day carries
  the report's expected move as extra variance (the mean square of the previous
  eight reactions, so nothing is learnt early), which drops out after the report:
  the volatility crush. Ordinary realized volatility leaves report days out;
- splits: options trade the actual price (Tiingo's unadjusted close, or Yahoo's
  split-adjusted close multiplied back by later splits). On a split's ex-date every
  open leg is adjusted as the exchange does: strike divided by the ratio, contracts
  multiplied by it.

Costs are charged on every option and stock trade: a per-contract commission,
and a bid/ask cost as a share of each option's price (with a one-cent floor).
Options held to expiry settle at intrinsic value with no spread cost.

The result is labelled "model-priced" everywhere it is shown.
"""

from __future__ import annotations

import bisect
import io
import json
import logging
import math
import os
import time
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
# Skew per unit of standardized moneyness, fitted on 2007-2015 against Cboe's protective-put
# (PPUT, puts 5% out of the money) and 30-delta covered-call (BXMD) indices and checked on 2016
# on (tools/options_validation.py): a 5%-out-of-the-money 30-day put trades near 1.2x
# at-the-money IV, a 30-delta call near 0.8x.
SKEW_PUT, SKEW_CALL = 0.175, 0.325
SKEW_CLIP = (0.75, 2.0)
VIX_TO_ATM = 0.875                  # fitted on 2007-2015 against Cboe's PutWrite index (see the docstring)
CURVE_TENORS = (9, 93, 183)         # days covered by the VIX9D, VIX3M and VIX6M
CURVE_SYMBOLS = ('^VIX9D', '^VIX3M', '^VIX6M')
UNBOUNDED_MOVE = 0.30               # cash reserved against an uncovered short call: a 30% rise
MAX_LEGS = 6
WARMUP_DAYS = 400                   # calendar days fetched before the window (200-day SMA, IV rank)
TREND_DAYS = 200
IVR_DAYS = 252
TRENDS = ('any', 'above', 'below')
EARNINGS_RULES = ('any', 'avoid', 'only')
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


def model_term_ratio(iv30: float, days: float) -> float:
    """Modelled IV at `days` over 30-day IV, used where the real VIX curve is missing.

    The slope exponent falls as volatility rises (about +0.10 at 15% IV, flat near 40%,
    negative above), matching how the VIX term structure moves between contango and
    backwardation.
    """
    beta = min(max(0.10 - 0.4 * (iv30 - 0.15), -0.15), 0.15)
    return (max(days, 1.0) / 30.0) ** beta


def bs_greeks(kind: str, S: float, K: float, T: float, r: float, sigma: float) -> tuple:
    """(delta, gamma, theta per year, vega per 1.00 of volatility) per share."""
    if T <= 0 or sigma <= 0:
        return (bs_delta(kind, S, K, T, r, sigma), 0.0, 0.0, 0.0)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    gamma = pdf / (S * sigma * sq)
    vega = S * pdf * sq
    decay = -S * pdf * sigma / (2 * sq)
    disc = r * K * math.exp(-r * T)
    if kind == 'call':
        return (_ncdf(d1), gamma, decay - disc * _ncdf(d2), vega)
    return (_ncdf(d1) - 1.0, gamma, decay + disc * _ncdf(-d2), vega)


def term_iv(iv30: float, T: float, curve: tuple | None = None) -> float:
    """ATM IV at maturity T (years) from the 30-day level.

    `curve` is the day's term structure as ((days, IV / 30-day IV), ...) sorted by days,
    including (30, 1.0); between points total variance is interpolated linearly, beyond
    the ends the end ratio is held. Without a curve the modelled slope is used.
    """
    if T <= 0:
        return iv30
    days = max(T * 365.0, 1.0)
    if not curve:
        return iv30 * model_term_ratio(iv30, days)
    if days <= curve[0][0]:
        return iv30 * curve[0][1]
    if days >= curve[-1][0]:
        return iv30 * curve[-1][1]
    for (d0, r0), (d1, r1) in zip(curve, curve[1:]):
        if d0 <= days <= d1:
            w0, w1 = r0 * r0 * d0, r1 * r1 * d1
            w = max(w0 + (w1 - w0) * (days - d0) / (d1 - d0), 1e-12)
            return iv30 * math.sqrt(w / days)
    return iv30                                    # pragma: no cover - curve is sorted


def day_curve(iv30: float, real: tuple | None) -> tuple:
    """The day's curve points: real ratios where known, the modelled slope where not."""
    pts = [(30.0, 1.0)]
    for i, d in enumerate(CURVE_TENORS):
        r = float(real[i]) if real is not None else float('nan')
        if not (math.isfinite(r) and r > 0):
            r = model_term_ratio(iv30, d)
        pts.append((float(d), r))
    return tuple(sorted(pts))


class VolCurve(tuple):
    """A day's term-structure points, plus the earnings reports ahead: ((days away, variance), ...).

    An option that is still open on a report's reaction day carries that report's variance
    on top of the curve's; once the report is out it drops away (the volatility crush).
    """
    events: tuple = ()


def with_events(curve: tuple, events: tuple) -> tuple:
    if not events:
        return curve
    c = VolCurve(curve)
    c.events = events
    return c


def event_iv(iv: float, T: float, curve) -> float:
    """IV raised by the variance of every earnings report before the option expires."""
    ev = getattr(curve, 'events', ())
    if not ev or T <= 0:
        return iv
    days = T * 365.0 + 1e-9
    extra = sum(v for d, v in ev if d <= days)
    return math.sqrt(iv * iv + extra / T) if extra > 0 else iv


def skewed_iv(iv_atm: float, S: float, K: float, T: float, skew: bool = True,
              curve: tuple | None = None) -> float:
    """IV for strike K and maturity T: the term structure, earnings ahead, then the skew by moneyness."""
    iv_atm = event_iv(term_iv(iv_atm, T, curve), T, curve)
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


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7 + 14)


def monthly_expiry_for(day: date, dte: int) -> date:
    """The standard monthly expiry (third Friday) nearest to day + dte, always after day.

    Cboe's strategy indices roll on these dates; weekly expiries (expiry_for) let a backtest
    roll every four to five weeks instead, which drifts off the monthly cycle.
    """
    target = day + timedelta(days=max(1, dte))
    cands = []
    for k in (-1, 0, 1):
        y, m = target.year, target.month + k
        y, m = (y - 1, 12) if m < 1 else ((y + 1, 1) if m > 12 else (y, m))
        cands.append(third_friday(y, m))
    later = [c for c in cands if c > day]
    return min(later, key=lambda c: (abs((c - target).days), c))


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
    # a cycle: puts until assigned, then covered calls until the shares are called away
    'wheel':           {'name': 'The wheel', 'group': 'Cycle', 'view': 'Neutral to bullish',
                        'legs': [_L('put', 'short', {'mode': 'delta', 'value': 0.30})], 'alloc': 100,
                        'wheel': {'call': _L('call', 'short', {'mode': 'delta', 'value': 0.30})}},
}


def describe_wheel(legs: list, wheel: dict, dte: int) -> str:
    """'Sells a put at 0.30 delta; if assigned, holds the shares and sells a call at 0.30 delta ...'."""
    return (f"Sells a put {_strike_words(legs[0], legs)}, {dte} days to expiry; if it expires in the money the "
            f"shares are bought at the strike, and a call {_strike_words(wheel['call'], [wheel['call']])} is sold "
            f"on them every {dte} days until they are called away, when puts start again.")


def check_wheel(legs: list, wheel: dict | None, stagger_days) -> None:
    """The wheel needs one short put to start from and runs one position at a time."""
    if wheel is None:
        return
    if len(legs) != 1 or legs[0]['type'] != 'put' or legs[0]['side'] != 'short':
        raise ValueError('The wheel starts from a single short put')
    if stagger_days is not None:
        raise ValueError('The wheel runs one position at a time; leave "a new position every" empty')


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

def clean_entry(raw) -> dict | None:
    """Validated entry rules, or None when there are none. Raises ValueError with a readable message.

    vix_min / vix_max: open only when the VIX is in this range; ivr_min / ivr_max: only when
    the underlying's IV rank (where today's IV sits in its past year, 0-100) is in this range;
    trend: 'above' or 'below' its 200-day average.
    """
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError('Entry rules must be an object')
    out = {}
    for key, lo, hi, label in (('vix_min', 5, 150, 'Lowest VIX'), ('vix_max', 5, 150, 'Highest VIX'),
                               ('ivr_min', 0, 100, 'Lowest IV rank'), ('ivr_max', 0, 100, 'Highest IV rank')):
        v = raw.get(key)
        if v in (None, ''):
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f'{label} must be a number')
        if not (lo <= v <= hi) or not math.isfinite(v):
            raise ValueError(f'{label} must be between {lo} and {hi}')
        out[key] = v
    trend = str(raw.get('trend') or 'any').lower()
    if trend not in TRENDS:
        raise ValueError('Trend must be any, above or below')
    if trend != 'any':
        out['trend'] = trend
    earn = str(raw.get('earnings') or 'any').lower()
    if earn not in EARNINGS_RULES:
        raise ValueError('Earnings must be any, avoid or only')
    if earn != 'any':
        out['earnings'] = earn
    for a, b, label in (('vix_min', 'vix_max', 'VIX'), ('ivr_min', 'ivr_max', 'IV rank')):
        if a in out and b in out and out[a] > out[b]:
            raise ValueError(f'The lowest {label} is above the highest')
    return out or None


def describe_entry(entry: dict | None, ticker: str) -> str:
    """'only when the VIX is 20 or higher and SPY is above its 200-day average', or ''."""
    if not entry:
        return ''
    parts = []
    lo, hi = entry.get('vix_min'), entry.get('vix_max')
    if lo is not None and hi is not None:
        parts.append(f'the VIX is between {lo:g} and {hi:g}')
    elif lo is not None:
        parts.append(f'the VIX is {lo:g} or higher')
    elif hi is not None:
        parts.append(f'the VIX is {hi:g} or lower')
    lo, hi = entry.get('ivr_min'), entry.get('ivr_max')
    if lo is not None and hi is not None:
        parts.append(f'its IV rank is between {lo:g} and {hi:g}')
    elif lo is not None:
        parts.append(f'its IV rank is {lo:g} or higher')
    elif hi is not None:
        parts.append(f'its IV rank is {hi:g} or lower')
    if entry.get('trend'):
        parts.append(f"{ticker} is {entry['trend']} its {TREND_DAYS}-day average")
    if entry.get('earnings') == 'avoid':
        parts.append('it would not hold through an earnings report')
    elif entry.get('earnings') == 'only':
        parts.append('it holds through an earnings report')
    return 'only when ' + ' and '.join(parts)


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

def realized_vol(close: pd.Series, window: int = RV_WINDOW, exclude=None) -> pd.Series:
    """Annualized rolling volatility; `exclude` (dates) drops earnings reactions, priced separately."""
    r = np.log(close).diff()
    if exclude is not None and len(exclude):
        r[r.index.isin(exclude)] = np.nan
    return r.rolling(window, min_periods=max(10, window // 2)).std() * math.sqrt(TRADING_DAYS)


def implied_vol_series(close: pd.Series, ticker: str, vix: pd.Series | None,
                       spy_close: pd.Series | None, exclude=None) -> tuple[pd.Series, str]:
    """ATM implied volatility per day, and a sentence saying how it was estimated."""
    rv = realized_vol(close, exclude=exclude)
    if vix is not None and len(vix):
        v = (vix / 100.0 * VIX_TO_ATM).reindex(close.index).ffill()
        if ticker == 'SPY':
            return v.clip(*IV_CLIP), (f'{VIX_TO_ATM:g} × the VIX (the VIX includes the skew, so at-the-money '
                                      'IV runs below it; the factor is fitted to Cboe’s PutWrite index)')
        if spy_close is not None and len(spy_close):
            rv_spy = realized_vol(spy_close.reindex(close.index).ffill())
            ratio = (v / rv_spy).clip(*VRP_CLIP)
            iv = (rv * ratio).clip(*IV_CLIP)
            return iv, (f'{ticker}’s 21-day realized volatility × the market’s implied-to-realized '
                        f'ratio that day ({VIX_TO_ATM:g} × VIX ÷ SPY realized volatility)')
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


CBOE_CSV = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv'
_CBOE_CACHE: dict = {}              # index -> (fetched_at, Series of closes)


def cboe_closes(index: str) -> pd.Series | None:
    """Daily closes of a Cboe index (VIX, VIX9D, VIX3M, VIX6M) from Cboe's own history file.

    One request for the whole history, cached for six hours. Cboe serves these to any
    client, including the server, where Yahoo refuses the same symbols.
    """

    import requests
    hit = _CBOE_CACHE.get(index)
    if hit is None or time.time() - hit[0] > _FRED_TTL:
        try:
            resp = requests.get(CBOE_CSV.format(index), timeout=15)
            resp.raise_for_status()
            series = cboe_frame(resp.text)
        except Exception as exc:          # pragma: no cover - network
            logger.warning('Cboe %s history unavailable: %s', index, exc)
            return None
        if series is None:
            return None
        hit = (time.time(), series)
        _CBOE_CACHE[index] = hit
    return hit[1]


def cboe_frame(text: str) -> pd.Series | None:
    """Parse a Cboe DATE,OPEN,HIGH,LOW,CLOSE history file into closes by day."""
    df = pd.read_csv(io.StringIO(text))
    cols = {c.strip().upper(): c for c in df.columns}
    if 'DATE' not in cols or 'CLOSE' not in cols:
        return None
    s = pd.Series(pd.to_numeric(df[cols['CLOSE']], errors='coerce').values,
                  index=pd.to_datetime(df[cols['DATE']], format='%m/%d/%Y', errors='coerce'))
    s = s[s.index.notna()].dropna()
    s = s[s > 0]
    return s.sort_index() if len(s) else None


def vix_curve(lo: str, hi: str) -> pd.DataFrame | None:
    """Daily ratios of the VIX9D, VIX3M and VIX6M to the VIX (columns = CURVE_TENORS), or None.

    From Cboe's history files; the general price fetch is only a fallback, because on the
    server it tries Yahoo first, which refuses index symbols there, and took most of a
    request to give up.
    """
    def closes(sym):
        c = cboe_closes(sym.lstrip('^'))
        if c is None:
            df = feat.fetch_ohlcv(sym, start=lo, end=hi)
            if df is None or df.empty:
                return None
            c = df['Close'].astype(float)
            c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        return c[(c.index >= pd.Timestamp(lo)) & (c.index <= pd.Timestamp(hi))]
    try:
        vix = closes('^VIX')
        if vix is None or vix.empty:
            return None
        cols = {}
        for d, sym in zip(CURVE_TENORS, CURVE_SYMBOLS):
            c = closes(sym)
            cols[d] = (c / vix) if c is not None else pd.Series(np.nan, index=vix.index)
        out = pd.DataFrame(cols).reindex(vix.index)
        return out.where((out > 0.3) & (out < 3.0))   # drop bad prints
    except Exception as exc:                          # pragma: no cover - network
        logger.warning('VIX curve unavailable: %s', exc)
        return None


def split_factor(index: pd.DatetimeIndex, splits: pd.Series | None) -> pd.Series:
    """Per day, the product of every split ratio whose ex-date comes later.

    A split-adjusted price times this factor is the price that actually traded that day
    (NVDA's split-adjusted $6 in January 2020 times 40 is the $240 it traded at).
    """
    f = pd.Series(1.0, index=index)
    if splits is None or not len(splits):
        return f
    for d, ratio in splits.items():
        if ratio and math.isfinite(ratio) and ratio > 0 and ratio != 1:
            f[f.index < d] *= float(ratio)
    return f


def _tiingo_actions(ticker: str, lo: str, hi: str) -> tuple | None:
    """(actual closes, dividends, splits) from Tiingo, which works from datacenter IPs. None if unavailable."""
    import os
    token = os.getenv('TIINGO_API_KEY', '').strip()
    if not token:
        return None
    try:
        import requests
        resp = requests.get(f'https://api.tiingo.com/tiingo/daily/{ticker}/prices',
                            params={'startDate': lo, 'endDate': hi, 'token': token, 'format': 'json'},
                            timeout=15)
        if resp.status_code != 200:
            return None
        return tiingo_frame(resp.json())
    except Exception as exc:                       # pragma: no cover - network
        logger.warning('Tiingo actions unavailable for %s: %s', ticker, exc)
        return None


def tiingo_frame(rows: list) -> tuple | None:
    """Parse Tiingo daily rows: `close` is the unadjusted price, `divCash` and `splitFactor` the actions."""
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if not {'date', 'close'} <= set(df.columns):
        return None
    df.index = pd.to_datetime(df.pop('date'), utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_index()
    raw = df['close'].astype(float).dropna()
    dv = df['divCash'].astype(float) if 'divCash' in df else pd.Series(0.0, index=df.index)
    sp = df['splitFactor'].astype(float) if 'splitFactor' in df else pd.Series(1.0, index=df.index)
    return raw, dv[dv > 0], sp[(sp > 0) & (sp != 1)]


def dividend_data(ticker: str, lo: str, hi: str) -> tuple:
    """(actual traded closes, dividends per share by ex-date, split ratios by ex-date), or Nones.

    Tiingo first (actual prices and both actions in one call); otherwise Yahoo, whose
    "unadjusted" closes and dividends are still split-adjusted, so they are multiplied back
    by the split factor. Trading on split-adjusted prices would put NVDA at $6 in 2020: the
    wrong strike spacing, a one-cent floor that dwarfs the premium and 40 times the contracts.
    """
    t = _tiingo_actions(ticker, lo, hi)
    if t is not None:
        return t
    try:
        import yfinance as yf
        df = yf.download(ticker, start=lo, end=hi, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None, None, None
        c = df['Close']
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        c = c.astype(float)
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        tk = yf.Ticker(ticker)
        dv = tk.dividends
        dv = dv.astype(float) if dv is not None else pd.Series(dtype=float)
        if len(dv):
            dv.index = pd.to_datetime(dv.index).tz_localize(None).normalize()
        sp = tk.splits
        sp = sp.astype(float) if sp is not None else pd.Series(dtype=float)
        if len(sp):
            sp.index = pd.to_datetime(sp.index).tz_localize(None).normalize()
        sp = sp[(sp > 0) & (sp != 1)]
        dv = dv[dv > 0]
        c = c.dropna() * split_factor(c.dropna().index, sp)
        if len(dv):
            dv = dv * split_factor(dv.index, sp)
        return c, dv, sp
    except Exception as exc:                       # pragma: no cover - network
        logger.warning('dividend data unavailable for %s: %s', ticker, exc)
        return None, None, None


_EARN_CACHE: dict = {}              # ticker -> (fetched_at, events)
_EARN_TTL = 6 * 3600
EARN_LOOKBACK = 8                   # previous reports averaged for the next one's typical move
EARN_FALLBACK_DAYS = 3.0            # with no earlier report: a move of 3 ordinary days' volatility


OPTIONS_EARNINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'options_earnings.json')
_OPT_EARN: dict | None = None


def _saved_earnings() -> dict:
    """backend/options_earnings.json (tools/refresh_options_earnings.py), read once."""
    global _OPT_EARN
    if _OPT_EARN is None:
        try:
            with open(OPTIONS_EARNINGS_PATH, encoding='utf-8') as fh:
                _OPT_EARN = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning('options earnings file unavailable: %s', exc)
            _OPT_EARN = {}
    return _OPT_EARN


def earnings_events(ticker: str) -> list | None:
    """Earnings reports as [{'date': Timestamp, 'when': 'amc'|'bmo'|None}], oldest first.

    [] for index ETFs (no earnings); None when no dates could be found. The saved file
    comes first, so a backtest gives the same answer here and on the server, where Yahoo
    is blocked; then Yahoo (about 25 years, with the time of day); then the Predictor's
    earnings snapshot (about six years of dates, no times).
    """
    ticker = ticker.upper()
    if ticker in INDEX_ETFS:
        return []
    saved = (_saved_earnings().get('tickers') or {}).get(ticker)
    if saved:
        return [{'date': pd.Timestamp(d), 'when': w} for d, w in saved]
    hit = _EARN_CACHE.get(ticker)
    if hit and time.time() - hit[0] < _EARN_TTL:
        return hit[1]
    out = None
    try:
        if os.getenv('RENDER'):                    # Yahoo refuses Render's IPs: don't wait on it there
            raise RuntimeError('Yahoo is not reachable from the server')
        import yfinance as yf
        raw = yf.Ticker(ticker).get_earnings_dates(limit=100)
        if raw is not None and len(raw):
            out = []
            for ts in raw.index:
                ts = pd.Timestamp(ts)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert('America/New_York').tz_localize(None)
                when = 'amc' if ts.hour >= 16 else ('bmo' if 0 < ts.hour < 10 else None)
                out.append({'date': ts.normalize(), 'when': when})
    except Exception as exc:                       # pragma: no cover - network
        logger.warning('earnings dates unavailable for %s: %s', ticker, exc)
    if not out:
        try:
            import earnings as earn_mod
            snap = earn_mod._snapshot_frame(ticker)
            if snap is not None and len(snap):
                out = [{'date': pd.Timestamp(d).normalize(), 'when': None} for d in snap.index]
        except Exception as exc:                   # pragma: no cover - defensive
            logger.warning('earnings snapshot unavailable for %s: %s', ticker, exc)
            out = None
    out = sorted(out, key=lambda e: e['date']) if out else None
    _EARN_CACHE[ticker] = (time.time(), out)
    return out


def reaction_days(close: pd.Series, events: list) -> list:
    """The trading day on which each report moves the price.

    After the close: the next trading day; before the open: that day; time unknown: the
    larger move of the two, which is how the report shows up in the prices.
    """
    idx = close.index
    rets = np.log(close).diff()
    out = set()
    for e in events or ():
        j = idx.searchsorted(e['date'])            # first trading day on or after the date
        if j >= len(idx):
            continue
        same_day = idx[j] == e['date']
        if e['when'] == 'amc':
            k = j + 1 if same_day else j
        elif e['when'] == 'bmo':
            k = j
        elif same_day and j + 1 < len(idx):
            k = j if abs(rets.iloc[j]) >= abs(rets.iloc[j + 1]) else j + 1
        else:
            k = j
        if 0 < k < len(idx):
            out.add(idx[k])
    return sorted(out)


def event_moves(close: pd.Series, days: list, rv: pd.Series) -> dict:
    """Each report's expected move as a variance: the mean square of the previous reactions.

    Only reports before it count, so a backtest never learns a reaction early. With none
    yet, it is EARN_FALLBACK_DAYS ordinary days' volatility.
    """
    rets = np.log(close).diff()
    out, past = {}, []
    for d in days:
        if past:
            recent = past[-EARN_LOOKBACK:]
            var = float(np.mean(np.square(recent)))
        else:
            daily = float(rv.get(d, np.nan)) / math.sqrt(TRADING_DAYS)
            var = (EARN_FALLBACK_DAYS * daily) ** 2 if math.isfinite(daily) else 0.0025
        out[d] = var
        r = float(rets.get(d, np.nan))
        if math.isfinite(r):
            past.append(r)
    return out


def load_market(ticker: str, start: str, end: str) -> dict:
    """Closes for the ticker (and SPY), the VIX, its term structure and T-bill rates, with a warm-up."""
    # A year and more of warm-up: the 200-day average and a 252-day IV rank need it.
    lo = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
    df = feat.fetch_ohlcv(ticker, start=lo, end=end)
    if df is None or df.empty:
        return {'error': f'No price history for {ticker}'}
    close = df['Close'].astype(float).dropna()      # Yahoo adds today's row with no close yet
    if close.empty:
        return {'error': f'No price history for {ticker}'}
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    spy = None
    if ticker != 'SPY':
        sdf = feat.fetch_ohlcv('SPY', start=lo, end=end)
        if sdf is not None and not sdf.empty:
            spy = sdf['Close'].astype(float).dropna()
            spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    vix = _fred('VIXCLS', lo, end)
    rate = _fred('DGS3MO', lo, end)
    curve = vix_curve(lo, end)
    raw, divs, splits = dividend_data(ticker, lo, end)
    return {'close': close, 'spy': spy, 'vix': vix, 'rate': rate, 'curve': curve,
            'raw_close': raw, 'dividends': divs, 'splits': splits, 'earnings': earnings_events(ticker)}


# ── Position mechanics ───────────────────────────────────────────────────────

def _sign(leg):
    return 1 if leg['side'] == 'long' else -1


def _delta_strike(grid: list, kind: str, want: float, S: float, T: float, r: float,
                  iv_atm: float, skew: bool, curve: tuple | None, q: float = 0.0) -> float:
    """The strike on a sorted grid whose |delta| is closest to `want`.

    |delta| falls with the strike for calls and rises for puts, so a bisection finds the
    crossing in a handful of prices; the strikes around it are then compared directly, so
    the answer is the same as checking every strike, about twenty times faster.
    """
    def miss(k):
        dq = math.exp(-q * T)
        return abs(dq * bs_delta(kind, S * dq, k, T, r, skewed_iv(iv_atm, S, k, T, skew, curve))) - want
    sign = -1.0 if kind == 'call' else 1.0          # make the miss increase with the strike
    lo, hi = 0, len(grid) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sign * miss(grid[mid]) < 0:
            lo = mid
        else:
            hi = mid
    near = grid[max(0, lo - 3):min(len(grid), hi + 4)]
    return min(near, key=lambda k: abs(miss(k)))


def build_position(legs: list, day: date, S: float, iv_atm: float, r: float, dte: int,
                   ticker: str, skew: bool, curve: tuple | None = None, monthly: bool = False,
                   q: float = 0.0) -> list:
    """Concrete legs (strike, expiry, entry price) for a position opened on `day` at price S."""
    step = strike_step(S, ticker)
    built = []
    for leg in legs:
        if leg['type'] == 'stock':
            built.append({**leg, 'strike': None, 'expiry': None, 'price': S})
            continue
        exp = (monthly_expiry_for if monthly else expiry_for)(day, leg.get('dte') or dte)
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
            K = _delta_strike(grid, leg['type'], want, S, T, r, iv_atm, skew, curve, q)
        iv = skewed_iv(iv_atm, S, K, T, skew, curve)
        built.append({**leg, 'strike': float(K), 'expiry': exp,
                      'price': bs_price(leg['type'], S * math.exp(-q * T), K, T, r, iv)})
    return built


def leg_value(leg: dict, day: date, S: float, iv_atm: float, r: float, skew: bool,
              curve: tuple | None = None, q: float = 0.0) -> float:
    """Per-share value of one leg on `day`."""
    if leg['type'] == 'stock':
        return S
    T = max(0.0, (leg['expiry'] - day).days / 365.0)
    return bs_price(leg['type'], S * math.exp(-q * T), leg['strike'], T, r,
                    skewed_iv(iv_atm, S, leg['strike'], T, skew, curve))


def position_marks(pos: list, day: date, S: float, iv_atm: float, r: float, skew: bool,
                   curve: tuple | None = None, q: float = 0.0) -> dict:
    """One unit's value and Greeks (dollars), plus each leg's IV, for P&L attribution.

    delta, gamma: dollars per $1 move of the underlying (and per $1 squared); theta: dollars
    per year; vega: dollars per 1.00 of volatility, kept per leg because each leg's IV moves
    by a different amount (skew and term structure).
    """
    value = delta = gamma = theta = 0.0
    short_delta = 0.0
    legs = []
    for leg in pos:
        k = _sign(leg) * leg['qty'] * CONTRACT
        if leg['type'] == 'stock':
            value += k * S
            delta += k
            legs.append((None, 0.0))
            continue
        T = max(0.0, (leg['expiry'] - day).days / 365.0)
        sig = skewed_iv(iv_atm, S, leg['strike'], T, skew, curve)
        dq = math.exp(-q * T)
        value += k * bs_price(leg['type'], S * dq, leg['strike'], T, r, sig)
        d, g, th, vg = bs_greeks(leg['type'], S * dq, leg['strike'], T, r, sig)
        delta += k * d * dq                 # Greeks with respect to the spot, not the discounted spot
        if leg['side'] == 'short':
            short_delta = max(short_delta, abs(d * dq))
        gamma += k * g * dq * dq
        theta += k * th
        legs.append((sig, k * vg * dq))
    return {'S': S, 'day': day, 'value': value, 'delta': delta, 'gamma': gamma, 'theta': theta,
            'legs': legs, 'short_delta': short_delta}


ATTR_KEYS = ('delta', 'gamma', 'theta', 'vega', 'other')


def attribute(prev: dict, now: dict) -> dict:
    """Split one unit's change in value between two marks with the earlier mark's Greeks.

    delta x dS + 1/2 gamma x dS^2 + theta x dt + sum of each leg's vega x its own dIV; whatever
    is left (higher-order terms, rate moves, the jump to intrinsic at expiry) is 'other', so
    the parts always add up to the actual change.
    """
    dS = now['S'] - prev['S']
    dt = (now['day'] - prev['day']).days / 365.0
    vega = sum(v * ((s1 if s1 is not None else 0.0) - (s0 if s0 is not None else 0.0))
               for (s0, v), (s1, _) in zip(prev['legs'], now['legs']) if s0 is not None)
    out = {'delta': prev['delta'] * dS, 'gamma': 0.5 * prev['gamma'] * dS * dS,
           'theta': prev['theta'] * dt, 'vega': vega}
    out['other'] = (now['value'] - prev['value']) - sum(out.values())
    return out


def position_value(pos: list, day, S, iv_atm, r, skew, curve: tuple | None = None, q: float = 0.0) -> float:
    """Value of one unit of the position (dollars), long legs positive, short negative."""
    return sum(_sign(leg) * leg['qty'] * CONTRACT * leg_value(leg, day, S, iv_atm, r, skew, curve, q)
               for leg in pos)


def early_assignment(pos: list, day: date, S: float, iv_atm: float, r: float, skew: bool,
                     curve: tuple | None, q: float, dividend: float) -> bool:
    """True when a short call in the position would be assigned before tomorrow's ex-date.

    A holder exercises an in-the-money call the day before the stock goes ex-dividend when
    the dividend is worth more than the time value given up by exercising.
    """
    for leg in pos:
        if leg['type'] != 'call' or leg['side'] != 'short' or S <= leg['strike']:
            continue
        time_value = leg_value(leg, day, S, iv_atm, r, skew, curve, q) - (S - leg['strike'])
        if time_value < dividend:
            return True
    return False


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


def split_words(ratio: float) -> str:
    """4.0 -> '4-for-1 split', 0.1 -> '1-for-10 reverse split', 1.5 -> '3-for-2 split'."""
    from fractions import Fraction
    fr = Fraction(ratio).limit_denominator(20)
    kind = 'reverse split' if ratio < 1 else 'split'
    return f'{fr.numerator}-for-{fr.denominator} {kind}'


def apply_split(pos: list, entry: dict, ratio: float) -> None:
    """Adjust an open position for a split: strikes and prices / ratio, contracts and shares x ratio.

    The position's value is unchanged; the last mark is rescaled the same way, so the
    next day's P&L attribution compares like with like.
    """
    for leg in pos:
        leg['qty'] = leg['qty'] * ratio
        leg['price'] = leg['price'] / ratio
        if leg['type'] != 'stock':
            leg['strike'] = leg['strike'] / ratio
    mk = entry['mark']
    mk['S'] = mk['S'] / ratio
    mk['delta'] = mk['delta'] * ratio
    mk['gamma'] = mk['gamma'] * ratio * ratio
    entry['split'] = f'{split_words(ratio)} while open: strikes and contracts adjusted'


def pricing_context(m: dict, ticker: str) -> dict:
    """The inputs run() prices options with, as daily series, for pricing outside a backtest.

    Same steps as run(): earnings reaction days left out of realized volatility, the real
    VIX curve, the T-bill rate, and the trailing dividend yield on actual (unsplit) prices.
    tools/options_quotes_check.py uses it to price real quoted contracts with the model.
    """
    ticker = ticker.upper()
    close = m['close']
    raw_close, divs, splits = m.get('raw_close'), m.get('dividends'), m.get('splits')
    use_divs = raw_close is not None and len(raw_close) and divs is not None
    ev_list = m.get('earnings')
    react = reaction_days(close, ev_list) if ev_list else []
    iv_all, _ = implied_vol_series(close, ticker, m.get('vix'), m.get('spy'), exclude=react or None)
    ev_var = event_moves(close, react, realized_vol(close, exclude=react)) if react else {}
    curve = m.get('curve')
    curve = curve.reindex(close.index).ffill(limit=5) if curve is not None else None
    rate = m.get('rate')
    r_all = ((rate / 100.0).reindex(close.index).ffill().bfill() if rate is not None and len(rate)
             else pd.Series(0.03, index=close.index)).fillna(0.03)
    px = raw_close.reindex(close.index).ffill() if use_divs else close
    fac = split_factor(close.index, splits) if use_divs and splits is not None and len(splits) else \
        pd.Series(1.0, index=close.index)
    if use_divs:
        div_adj = divs.groupby(level=0).sum().reindex(close.index, fill_value=0.0) / fac
        q_all = (div_adj.rolling('365D').sum() / (px / fac)).clip(0.0, 0.2).fillna(0.0)
    else:
        q_all = pd.Series(0.0, index=close.index)
    return {'ticker': ticker, 'px': px, 'iv': iv_all, 'r': r_all, 'q': q_all, 'curve': curve,
            'react': [d.date() for d in react], 'ev_var': ev_var}


def context_at(ctx: dict, day) -> dict | None:
    """(S, ATM IV, rate, curve with the earnings ahead, dividend yield) on a trading day, or None."""
    ts = pd.Timestamp(day)
    if ts not in ctx['px'].index:
        return None
    d = ts.date()
    iv = float(ctx['iv'].get(ts, np.nan))
    if not math.isfinite(iv):
        return None
    real = tuple(ctx['curve'].loc[ts]) if ctx['curve'] is not None else None
    cv = day_curve(iv, real)
    react = ctx['react']
    if react:
        k = bisect.bisect_right(react, d)
        cv = with_events(cv, tuple(((e - d).days, ctx['ev_var'][pd.Timestamp(e)])
                                   for e in react[k:k + 4] if (e - d).days <= 400))
    return {'S': float(ctx['px'][ts]), 'iv': iv, 'r': float(ctx['r'][ts]), 'q': float(ctx['q'][ts]), 'curve': cv,
            'next_report': next((e for e in react if e > d), None)}


def model_price(kind: str, c: dict, K: float, T: float, skew: bool = True) -> tuple[float, float]:
    """(price per share, IV) of an option with the model's inputs `c` from context_at."""
    sig = skewed_iv(c['iv'], c['S'], K, T, skew, c['curve'])
    return bs_price(kind, c['S'] * math.exp(-c['q'] * T), K, T, c['r'], sig), sig


def implied_vol(kind: str, price: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float | None:
    """The volatility at which Black-Scholes-Merton gives `price`, by bisection; None if out of range."""
    if T <= 0 or price <= 0:
        return None
    Sq = S * math.exp(-q * T)
    lo, hi = 0.005, 5.0
    if not (bs_price(kind, Sq, K, T, r, lo) <= price <= bs_price(kind, Sq, K, T, r, hi)):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(kind, Sq, K, T, r, mid) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── The backtest ──────────────────────────────────────────────────────────────

def run(ticker: str, start: str, end: str, legs: list, name: str = 'Custom strategy',
        dte: int = 30, initial_capital: float = 100_000, alloc_pct: float = 100,
        exit_dte: int | None = None, profit_take: float | None = None, stop_loss: float | None = None,
        spread_pct: float | None = None, commission: float = 0.65, stock_slip: float = 0.0003,
        skew: bool = True, market: dict | None = None, entry: dict | None = None,
        iv_scale: float = 1.0, sizing: str = 'risk', expiries: str = 'weekly',
        exit_earnings: bool = False, exit_delta: float | None = None, stagger_days: int | None = None,
        wheel: dict | None = None) -> dict:
    """Backtest `legs` on `ticker` between start and end. See the module docstring."""
    ticker = ticker.upper()
    rules = entry               # `entry` is reused below for the open position
    try:
        check_wheel(legs, wheel, stagger_days)
    except ValueError as exc:
        return {'error': str(exc)}
    if spread_pct is None:
        spread_pct = default_spread(ticker)
    m = market or load_market(ticker, start, end)
    if 'error' in m:
        return m
    close = m['close']                    # dividend-adjusted: the buy-and-hold benchmark (total return)
    raw_close, divs = m.get('raw_close'), m.get('dividends')
    use_divs = raw_close is not None and len(raw_close) and divs is not None
    splits = m.get('splits')
    # Earnings: each report's reaction day, priced as a separate jump (see event_moves), so the
    # ordinary volatility is measured without those days.
    ev_list = m.get('earnings')
    react = reaction_days(close, ev_list) if ev_list else []
    iv_all, iv_note = implied_vol_series(close, ticker, m.get('vix'), m.get('spy'), exclude=react or None)
    ev_var = event_moves(close, react, realized_vol(close, exclude=react)) if react else {}
    react_d = [d.date() for d in react]
    react_set = set(react_d)
    earn_rule = (rules or {}).get('earnings')
    if iv_scale != 1.0:
        iv_all = (iv_all * iv_scale).clip(*IV_CLIP)
    real_curve = m.get('curve')
    if real_curve is not None:
        real_curve = real_curve.reindex(close.index).ffill(limit=5)
    rate = m.get('rate')
    r_all = ((rate / 100.0).reindex(close.index).ffill().bfill() if rate is not None and len(rate)
             else pd.Series(0.03, index=close.index))
    r_all = r_all.fillna(0.03)

    mask = (close.index >= pd.Timestamp(start)) & (close.index <= pd.Timestamp(end))
    days = close.index[mask]
    if len(days) < 30:
        return {'error': 'Need at least 30 trading days in the window'}

    cash = float(initial_capital)
    book = []                   # open positions: {'pos': legs, 'units': n, 'entry': {...}}
    last_open = None            # the day the latest position opened (for stagger_days)
    wheel_shares = None         # the wheel: units of shares held after a put was assigned
    wheel_counts = {'assigned': 0, 'called_away': 0}
    max_open = 0
    earn_skips = 0
    tied_sum = 0.0              # share of the account tied up in positions, summed over days
    equity, trades, notes = [], [], []
    total_costs = 0.0
    first_pos = None
    days_in = 0
    curve_days = 0
    interest = 0.0
    dividends_paid = 0.0
    assignments = 0
    bench_shares = initial_capital / float(close[days[0]])
    bench_arr = close.reindex(days).to_numpy(dtype=float)
    last_day = days[-1].date()

    def close_position(P, day, S, iv, r, cv, q, reason, settle=False, assigned=False, carry=False):
        """Close one position. settle: expiring options settle at intrinsic, no spread; assigned:
        short calls are exercised (intrinsic, no spread); carry: stock legs change hands without
        a trade (wheel shares kept, delivered or called away), so no slippage."""
        nonlocal cash, total_costs
        pos, units, entry = P['pos'], P['units'], P['entry']
        prices = [leg_value(leg, day, S, iv, r, skew, cv, q) for leg in pos]
        if assigned:                      # short calls are exercised against us: settled at intrinsic
            prices = [max(0.0, S - leg['strike']) if leg['type'] == 'call' and leg['side'] == 'short' else p
                      for leg, p in zip(pos, prices)]
        value = sum(_sign(leg) * leg['qty'] * CONTRACT * p for leg, p in zip(pos, prices))
        # Expiring options settle at intrinsic; any longer-dated legs are sold at the model price.
        cost = 0.0
        for leg, p in zip(pos, prices):
            if carry and leg['type'] == 'stock':
                continue
            done = (settle and leg['type'] != 'stock' and leg['expiry'] <= day) or \
                (assigned and leg['type'] == 'call' and leg['side'] == 'short')
            cost += trade_cost([leg], [p], spread_pct, commission, stock_slip, settle=done)
        cost *= units
        cash += units * value - cost
        total_costs += cost
        pnl = units * (value - entry['value']) - cost - entry['cost'] + entry['divs']
        attr = {k: round(units * v, 2) for k, v in entry['attr'].items()}
        attr['costs'] = round(-(cost + entry['cost']), 2)
        attr['dividends'] = round(entry['divs'], 2)
        attr['other'] = round(attr['other'] + round(pnl, 2) - sum(attr.values()), 2)   # rounding
        trades.append({
            'opened': entry['day'].isoformat(), 'closed': day.isoformat(), 'reason': reason,
            'units': units, 'underlying_open': round(entry['S'], 2), 'underlying_close': round(S, 2),
            'legs': [{'side': leg['side'], 'type': leg['type'], 'qty': round(leg['qty'] * units, 4),
                      'strike': None if leg['strike'] is None else round(leg['strike'], 2),
                      'expiry': leg['expiry'].isoformat() if leg['expiry'] else None,
                      'open': round(leg['price'], 2), 'close': round(p, 2)} for leg, p in zip(pos, prices)],
            'net_open': round(units * entry['value'], 2), 'net_close': round(units * value, 2),
            'costs': round(cost + entry['cost'], 2), 'dividends': round(entry['divs'], 2), 'pnl': round(pnl, 2),
            'split': entry.get('split'), 'earnings': entry['earnings'],
            'pnl_pct': round(pnl / entry['basis'] * 100, 2) if entry['basis'] else None,
            'attribution': attr,
        })
        book.remove(P)

    def open_position(cand, n, day, S, iv, r, cv, q, need, carry=False):
        """Open n units of `cand`; carry: its stock legs are shares already held (no trade)."""
        nonlocal cash, total_costs, first_pos, last_open
        open_value = position_value(cand, day, S, iv, r, skew, cv, q)
        traded = [leg for leg in cand if not (carry and leg['type'] == 'stock')]
        cost = n * trade_cost(traded, [leg['price'] for leg in traded], spread_pct, commission, stock_slip)
        cash -= n * open_value + cost
        total_costs += cost
        entry = {'day': day, 'S': S, 'value': open_value, 'cost': cost, 'basis': n * need,
                 'mark': position_marks(cand, day, S, iv, r, skew, cv, q), 'divs': 0.0,
                 'attr': dict.fromkeys(ATTR_KEYS, 0.0), 'earnings': 0}
        book.append({'pos': cand, 'units': n, 'entry': entry})
        last_open = day
        if first_pos is None:
            first_pos = {'day': day.isoformat(), 'S': S, 'legs': [dict(leg) for leg in cand],
                         'need': need, 'unbounded': capital_required(cand, S)[1], 'units': n}

    prev_day = None
    # Plain arrays for the daily loop (pandas lookups per day were a large share of the run time).
    # Options and stock legs trade the unadjusted price; dividends arrive as cash.
    px = raw_close.reindex(close.index).ffill() if use_divs else close
    S_arr = px.reindex(days).to_numpy(dtype=float)
    if use_divs and not np.all(np.isfinite(S_arr)):
        use_divs = False
        px = close
        S_arr = close.reindex(days).to_numpy(dtype=float)
    # Splits: options trade the actual price, so on an ex-date every open leg is adjusted as the
    # exchange does (strike divided by the ratio, contracts multiplied). Yields and the trend
    # rule use split-adjusted prices, which have no jump.
    split_arr = np.ones(len(days))
    fac = pd.Series(1.0, index=close.index)
    if use_divs and splits is not None and len(splits):
        fac = split_factor(close.index, splits)
        for d, ratio in splits.groupby(level=0).prod().items():
            j = days.searchsorted(d)
            if 0 < j < len(days):
                split_arr[j] *= float(ratio)
    if use_divs:
        div_on = divs.groupby(level=0).sum()
        div_arr = div_on.reindex(days).fillna(0.0).to_numpy(dtype=float)
        div_adj = div_on.reindex(close.index, fill_value=0.0) / fac
        trailing = div_adj.rolling('365D').sum()
        S_adj = S_arr / fac.reindex(days).to_numpy(dtype=float)
        q_arr = np.clip((trailing.reindex(days).to_numpy(dtype=float) / S_adj), 0.0, 0.2)
    else:
        div_arr = np.zeros(len(days))
        q_arr = np.zeros(len(days))
    splits_applied = []
    iv_arr = iv_all.reindex(days).to_numpy(dtype=float)
    r_arr = r_all.reindex(days).fillna(0.03).to_numpy(dtype=float)
    iv_fill = float(iv_all.dropna().iloc[0]) if iv_all.notna().any() else 0.25
    cv_arr = real_curve.reindex(days).to_numpy(dtype=float) if real_curve is not None else None
    allow = np.ones(len(days), dtype=bool)
    if rules:
        if 'vix_min' in rules or 'vix_max' in rules:
            vx = m.get('vix')
            vix_arr = (vx.reindex(close.index).ffill().reindex(days).to_numpy(dtype=float)
                       if vx is not None and len(vx) else np.full(len(days), np.nan))
            with np.errstate(invalid='ignore'):
                allow &= vix_arr >= rules.get('vix_min', -np.inf)
                allow &= vix_arr <= rules.get('vix_max', np.inf)
        if 'ivr_min' in rules or 'ivr_max' in rules:
            ivr = iv_all.rolling(IVR_DAYS, min_periods=IVR_DAYS // 2).rank(pct=True) * 100
            ivr_arr = ivr.reindex(days).to_numpy(dtype=float)
            with np.errstate(invalid='ignore'):
                allow &= ivr_arr >= rules.get('ivr_min', -np.inf)
                allow &= ivr_arr <= rules.get('ivr_max', np.inf)
        if rules.get('trend'):
            # On the same prices as S (unadjusted when dividends are on), or the comparison is skewed.
            tp = px / fac
            sma = tp.rolling(TREND_DAYS, min_periods=TREND_DAYS).mean().reindex(days).to_numpy(dtype=float)
            tp_arr = tp.reindex(days).to_numpy(dtype=float)
            with np.errstate(invalid='ignore'):
                allow &= (tp_arr > sma) if rules['trend'] == 'above' else (tp_arr < sma)
    for i, ts in enumerate(days):
        day = ts.date()
        S = float(S_arr[i])
        iv = float(iv_arr[i])
        r = float(r_arr[i])
        q = float(q_arr[i]) if np.isfinite(q_arr[i]) else 0.0
        if not np.isfinite(iv):
            iv = iv_fill
        if prev_day is not None and cash > 0:
            earned = cash * r * (day - prev_day).days / 365.0    # idle cash earns the T-bill rate
            cash += earned
            interest += earned
        prev_day = day
        real = tuple(cv_arr[i]) if cv_arr is not None else None
        cv = day_curve(iv, real)
        if react_d:
            k = bisect.bisect_right(react_d, day)
            cv = with_events(cv, tuple(((e - day).days, ev_var[pd.Timestamp(e)])
                                       for e in react_d[k:k + 4] if (e - day).days <= 400))
        react_next = i + 1 < len(days) and days[i + 1].date() in react_set
        curve_days += real is not None and bool(np.all(np.isfinite(real)))

        if split_arr[i] != 1.0:
            splits_applied.append(f'{day.isoformat()} ({split_words(split_arr[i])})')
            for P in book:
                apply_split(P['pos'], P['entry'], float(split_arr[i]))
        div_next = float(div_arr[i + 1]) if i + 1 < len(days) else 0.0
        for P in list(book):
            pos, units, entry = P['pos'], P['units'], P['entry']
            first_exp = min(leg['expiry'] for leg in pos if leg['expiry'])
            if day in react_set:
                entry['earnings'] += 1
            if div_arr[i] > 0:            # held at yesterday's close: the stock legs are paid the dividend
                shares = units * sum(leg['qty'] * CONTRACT for leg in pos if leg['type'] == 'stock')
                if shares:
                    paid = shares * float(div_arr[i])
                    cash += paid
                    entry['divs'] += paid
                    dividends_paid += paid
            mark = position_marks(pos, day, S, iv, r, skew, cv, q)
            for k, v in attribute(entry['mark'], mark).items():
                entry['attr'][k] += v
            entry['mark'] = mark
            value = mark['value']
            # The wheel's covered calls: exits apply to the call alone (the shares are kept).
            on_shares = wheel is not None and any(leg['type'] == 'stock' for leg in pos)
            if on_shares:
                call0 = next(leg for leg in pos if leg['type'] == 'call')
                c_open = call0['price'] * call0['qty'] * CONTRACT
                c_now = leg_value(call0, day, S, iv, r, skew, cv, q) * call0['qty'] * CONTRACT
                pnl, ref = units * (c_open - c_now), c_open * units or 1.0
            else:
                pnl = units * (value - entry['value'])
                ref = abs(entry['value']) * units or 1.0
            exit_why = None
            if day >= first_exp:
                exit_why = 'expiry'
            elif exit_dte is not None and (first_exp - day).days <= exit_dte:
                exit_why = f'{exit_dte} days before expiry'
            elif profit_take is not None and pnl >= profit_take / 100.0 * ref:
                exit_why = f'profit target ({profit_take:g}%)'
            elif stop_loss is not None and not on_shares and pnl <= -stop_loss / 100.0 * ref:
                exit_why = f'stop loss ({stop_loss:g}%)'
            elif exit_delta is not None and mark['short_delta'] >= exit_delta:
                exit_why = f'short delta reached {exit_delta:g}'
            elif exit_earnings and not on_shares and react_next and days[i + 1].date() <= first_exp:
                exit_why = 'before earnings'
            elif day == last_day:
                exit_why = 'end of test'
            elif div_next > 0 and early_assignment(pos, day, S, iv, r, skew, cv, q, div_next):
                exit_why = 'assigned early (dividend)'

            if exit_why is None:
                continue
            if exit_why == 'end of test':
                close_position(P, day, S, iv, r, cv, q, exit_why)
                wheel_shares = None
            elif wheel is None:
                if exit_why == 'assigned early (dividend)':
                    assignments += 1
                close_position(P, day, S, iv, r, cv, q, exit_why, settle=exit_why == 'expiry',
                               assigned=exit_why == 'assigned early (dividend)')
            elif not on_shares:
                # The wheel's put: assigned at expiry when in the money, so the shares are bought.
                put = next(leg for leg in pos if leg['type'] == 'put')
                if exit_why == 'expiry' and S < put['strike']:
                    close_position(P, day, S, iv, r, cv, q, 'expiry: assigned, bought the shares', settle=True)
                    wheel_counts['assigned'] += 1
                    wheel_shares = (units, put['qty'])
                else:
                    close_position(P, day, S, iv, r, cv, q, exit_why, settle=exit_why == 'expiry')
            else:
                call0 = next(leg for leg in pos if leg['type'] == 'call')
                early = exit_why == 'assigned early (dividend)'
                away = early or (exit_why == 'expiry' and S > call0['strike'])
                if early:
                    assignments += 1
                why = ('shares called away early (dividend)' if early else
                       'expiry: shares called away' if away else exit_why)
                close_position(P, day, S, iv, r, cv, q, why, settle=exit_why == 'expiry', assigned=early,
                               carry=True)
                if away:
                    wheel_counts['called_away'] += 1
                    wheel_shares = None

        remaining = (last_day - day).days
        eq_now = cash + sum(P['units'] * P['entry']['mark']['value'] for P in book)
        if wheel_shares is not None and not book:
            # Shares held: sell the next covered call on them straight away, whatever the entry rules.
            n, qty = wheel_shares
            call_legs = [{'type': 'stock', 'side': 'long', 'qty': qty}, {**wheel['call'], 'qty': qty}]
            cand = build_position(call_legs, day, S, iv, r, dte, ticker, skew, cv, monthly=expiries == 'monthly',
                                  q=q)
            open_position(cand, n, day, S, iv, r, cv, q, capital_required(cand, S)[0], carry=True)
        cand = None
        due = not book if stagger_days is None else (not book or (day - last_open).days >= stagger_days)
        if due and wheel_shares is None and day != last_day and remaining >= max(5, dte // 2) and allow[i] \
                and not (exit_earnings and react_next):
            cand = build_position(legs, day, S, iv, r, dte, ticker, skew, cv, monthly=expiries == 'monthly', q=q)
            if earn_rule:
                c_exp = min(leg['expiry'] for leg in cand if leg['expiry'])
                k = bisect.bisect_right(react_d, day)
                spans = k < len(react_d) and react_d[k] <= c_exp
                if spans != (earn_rule == 'only'):
                    cand = None
                    earn_skips += 1
        if cand is not None:
            need, unbounded = capital_required(cand, S)
            # 'risk' sizes by each position's worst case; 'notional' holds one unit per 100 shares'
            # worth of the underlying, as Cboe's strategy indices do (used to replicate them).
            per_unit = S * CONTRACT if sizing == 'notional' else need
            budget = max(0.0, eq_now) * alloc_pct / 100.0
            free = eq_now - sum(P['entry']['basis'] for P in book)       # staggered: capital not yet tied up
            n = int(min(budget, free) // per_unit)
            if n >= 1:
                open_position(cand, n, day, S, iv, r, cv, q, need)
            elif budget < per_unit and not notes:
                notes.append(f'On {day.isoformat()} one unit needed ${need:,.0f} of capital, more than '
                             f'{alloc_pct:g}% of the account; positions are skipped until it fits.')

        held = bool(book)
        days_in += held
        max_open = max(max_open, len(book))
        row = {'date': day.isoformat(), 'benchmark': round(bench_shares * float(bench_arr[i]), 2),
               'iv': round(iv * 100, 1)}
        val = cash
        dl = th = vg = 0.0
        for P in book:
            mk = P['entry']['mark']
            val += P['units'] * mk['value']
            dl += P['units'] * mk['delta']
            th += P['units'] * mk['theta']
            vg += P['units'] * sum(v for _, v in mk['legs']) / 100.0
        # Exposures for the chart: dollars per 1% move, per calendar day, per volatility point.
        row.update({'delta': round(dl * S * 0.01, 2), 'theta': round(th / 365.0, 2), 'vega': round(vg, 2)})
        row['value'] = round(val, 2)
        if val > 0:
            tied_sum += min(1.0, sum(P['entry']['basis'] for P in book) / val)
        equity.append(row)
        if val <= 0:
            notes.append(f'The account was wiped out on {day.isoformat()}.')
            break

    rf_daily = float(r_all[days].mean()) / TRADING_DAYS
    share = curve_days / max(1, len(equity))
    if share >= 0.99:
        term_note = 'The real VIX term structure each day (VIX9D, VIX, VIX3M, VIX6M), applied to every ticker.'
    elif share > 0:
        term_note = (f'The real VIX term structure on {share * 100:.0f}% of days (VIX9D, VIX, VIX3M, VIX6M), '
                     'a modelled slope where a point was missing; applied to every ticker.')
    else:
        term_note = 'A modelled slope (the VIX term structure was unavailable).'
    if rules and set(rules) - {'earnings'}:
        n_ok = int(allow[:len(equity)].sum())
        notes.append(f'The entry rules allowed new positions on {n_ok} of {len(equity)} trading days '
                     f'({n_ok / max(1, len(equity)) * 100:.0f}%).')
    if earn_rule:
        notes.append(f'The earnings rule held back a new position on {earn_skips} day{"s" if earn_skips != 1 else ""}.')
    extra = path_stats(equity)
    extra['capital_used'] = round(tied_sum / max(1, len(equity)) * 100, 1)
    out = _summarize(ticker, name, legs, dte, initial_capital, equity, trades, total_costs,
                     days_in, first_pos, notes, iv_note, alloc_pct, exit_dte, profit_take, stop_loss,
                     spread_pct, commission, skew, rf_daily)
    if 'strategy' in out:
        out['strategy']['entry'] = rules
        when = describe_entry(rules, ticker)
        if when:
            out['strategy']['rules'] = out['strategy']['rules'].replace(
                'Opens a new position the day the last one closes',
                f'Opens a new position {when}, from the day the last one closes', 1)
    if 'model' in out:
        if use_divs:
            out['model']['dividends'] = (
                f'Options priced on unadjusted prices with the trailing 12-month dividend yield; stock legs were '
                f'paid ${dividends_paid:,.0f} in dividends; short calls assigned early before an ex-date '
                f'{assignments} time{"s" if assignments != 1 else ""}.')
        else:
            out['model']['dividends'] = ('Dividend data was unavailable: prices are dividend-adjusted and the '
                                         'dividend yield is zero.')
        out['metrics']['early_assignments'] = assignments
        if not use_divs:
            out['model']['splits'] = ('Actual traded prices were unavailable, so prices are split-adjusted: for a '
                                      'stock that split, strike spacing, contract counts and the one-cent cost '
                                      'floor are approximate.')
        elif splits_applied:
            out['model']['splits'] = ('Traded at actual prices. Splits on ' + ', '.join(splits_applied) +
                                      ' adjusted any open position as the exchange does: strikes divided by '
                                      'the ratio, contracts multiplied by it.')
        else:
            out['model']['splits'] = 'Traded at actual prices; no splits in the window.'
    if expiries == 'monthly' and 'strategy' in out:
        out['strategy']['rules'] = out['strategy']['rules'].replace(
            f'{dte} days to expiry', f'on the monthly (third-Friday) expiry nearest {dte} days out', 1)
        out['strategy']['expiries'] = 'monthly'
    if 'metrics' in out:
        out['metrics'].update(extra)
    if 'model' in out:
        in_window = [d for d in react_d if days[0].date() <= d <= last_day]
        if react_d and in_window:
            last_var = ev_var[pd.Timestamp(in_window[-1])]
            out['model']['earnings'] = (
                f'{len(in_window)} earnings report{"s" if len(in_window) != 1 else ""} in the window. An option still '
                f'open on a report day carries the report’s expected move (the root mean square of the previous '
                f'{EARN_LOOKBACK} reactions; {math.sqrt(last_var) * 100:.1f}% for the latest) as extra variance, which '
                'drops out once the report is out, as implied volatility collapses after earnings. Ordinary '
                'volatility is measured without report days.')
        elif ev_list is None:
            out['model']['earnings'] = (f'No earnings dates were found for {ticker}, so options are priced without an '
                                        'earnings premium and report-day moves count as ordinary volatility.')
        elif ticker in INDEX_ETFS:
            out['model']['earnings'] = f'{ticker} is an index fund: no earnings reports.'
        st_ = out.get('strategy')
        if st_ is not None and exit_delta is not None:
            st_['rules'] = st_['rules'][:-1] + f", or closed when a short option's delta reaches {exit_delta:g}."
            st_['exit_delta'] = exit_delta
        if st_ is not None and stagger_days is not None:
            st_['rules'] = st_['rules'].replace(
                'Opens a new position the day the last one closes',
                f'Opens a new position every {stagger_days} days, alongside any still open', 1).replace(
                ', from the day the last one closes', f', every {stagger_days} days, alongside any still open', 1)
            st_['stagger_days'] = stagger_days
            peak = max_open
            notes.append(f'Staggered: up to {peak} position{"s" if peak != 1 else ""} open at once.')
        if st_ is not None and wheel is not None:
            st_['wheel'] = describe_wheel(legs, wheel, dte)
            st_['headline'] = (f'{name}: a cycle backtest (short put; once assigned, long {legs[0]["qty"] * CONTRACT} '
                               'shares and a short call)')
            st_['rules'] = st_['wheel'] + ' ' + st_['rules'].split(';', 1)[-1].strip().capitalize()
            st_['rules'] = st_['rules'].replace('Held to expiry.', 'Each option is held to expiry.')
            out['metrics']['wheel'] = dict(wheel_counts)
            n_as = wheel_counts['assigned']
            notes.append(f"Puts assigned {n_as} time{'s' if n_as != 1 else ''}; "
                         f"shares called away {wheel_counts['called_away']} time"
                         f"{'s' if wheel_counts['called_away'] != 1 else ''}. Entry and earnings rules apply when "
                         'selling puts; on assigned shares a call is always sold.')
        if exit_earnings and 'strategy' in out:
            out['strategy']['rules'] = out['strategy']['rules'][:-1] + ', or closed the day before an earnings report.'
            out['strategy']['exit_earnings'] = True
    if iv_scale != 1.0 and 'model' in out:
        out['model']['iv_scale'] = iv_scale
    if 'model' in out:
        out['model']['term'] = term_note
        out['model']['rf_daily'] = round(rf_daily, 10)
        total = {k: round(sum(t['attribution'][k] for t in trades), 2) for k in ATTR_KEYS + ('costs', 'dividends')}
        total['interest'] = round(interest, 2)
        out['attribution'] = total
    return out


def path_stats(equity: list) -> dict:
    """Worst and best calendar month, and the longest stretch below a previous high, in calendar days."""
    if len(equity) < 2:
        return {'worst_month': None, 'best_month': None, 'longest_drawdown_days': None}
    s = pd.Series([e['value'] for e in equity], index=pd.to_datetime([e['date'] for e in equity]))
    ends = s.groupby(s.index.to_period('M')).last()
    starts = pd.concat([s.iloc[:1], ends.iloc[:-1]]).to_numpy()
    month = ends.to_numpy() / starts - 1
    peak = s.cummax()
    under = s < peak * (1 - 1e-9)
    longest, began = 0, None
    for d, u in under.items():
        if u and began is None:
            began = d
        elif not u and began is not None:
            longest = max(longest, (d - began).days)
            began = None
    if began is not None:
        longest = max(longest, (s.index[-1] - began).days)
    return {'worst_month': round(float(month.min()) * 100, 2), 'best_month': round(float(month.max()) * 100, 2),
            'longest_drawdown_days': int(longest)}


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
                       'Dividends use the trailing yield, not announced amounts; only short calls are '
                       'assigned early (before ex-dates), puts are treated as European.',
                       'Daily closes only: exits are checked once a day, at the close.'],
        },
    }


def run_preset(key: str, ticker: str, start: str, end: str, **kw) -> dict:
    p = PRESETS[key]
    kw.setdefault('alloc_pct', p['alloc'])
    kw.setdefault('wheel', p.get('wheel'))
    return run(ticker, start, end, p['legs'], name=p['name'], **kw)


def presets_summary(dte: int = 30) -> list:
    """Every preset with its leg description, for the strategy picker."""
    out = []
    for key, p in PRESETS.items():
        d = describe_strategy(p['name'], p['legs'], dte)
        row = {'key': key, 'name': p['name'], 'group': p['group'], 'view': p['view'],
               'alloc': p['alloc'], 'legs': p['legs'], 'legs_count': d['legs_count'],
               'headline': d['headline'], 'description': d['legs']}
        if p.get('wheel'):
            row['wheel'] = p['wheel']
            row['note'] = describe_wheel(p['legs'], p['wheel'], dte)
        out.append(row)
    return out
