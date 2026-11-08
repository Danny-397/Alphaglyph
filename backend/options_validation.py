"""
How close are the options backtester's model prices to real option returns?

No free source has historical option quotes, but Cboe publishes the returns of
real option strategies as indices. The S&P 500 PutWrite index (PUT) sells a
one-month at-the-money put every month and keeps the strike in T-bills; Yahoo
carries its full daily history. The backtester can run the same strategy on SPY
with its model prices, so the two return streams can be compared directly.

Fit, then test
--------------
The one free level in the model, VIX_TO_ATM (at-the-money IV as a share of the
VIX), is fitted on 2007-2015 so that the replica's annual return matches the
index's. The comparison that matters is 2016 on, which the fit never saw.

The replica is not identical to the index: the index trades SPX options at
Cboe's special opening quotation on the third Friday, the replica trades SPY
options at the close every four to five weeks. The fitted level absorbs the
average effect of those differences, so it is an effective level, not a
measurement of implied volatility.

The term structure is checked the same way: the modelled slope (used only where
the real VIX curve is missing) against the VIX9D, VIX3M and VIX6M.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import options_bt as ob

PUTWRITE_LEGS = [{'type': 'put', 'side': 'short', 'qty': 1, 'strike': {'mode': 'atm'}}]
FIT = ('2007-01-03', '2015-12-31')
TEST_START = '2016-01-04'
GRID = [round(0.75 + 0.005 * i, 3) for i in range(41)]          # 0.750 ... 0.950


def _cagr(s: pd.Series) -> float:
    years = len(s) / ob.TRADING_DAYS
    return (float(s.iloc[-1]) / float(s.iloc[0])) ** (1 / years) - 1


def compare(model: pd.Series, real: pd.Series) -> dict:
    """Return statistics of a model equity curve against a real index over the same days."""
    real = real.reindex(model.index).ffill().bfill()
    mm = model.resample('ME').last().pct_change().dropna()
    rm = real.resample('ME').last().pct_change().dropna()
    mm, rm = mm.align(rm, join='inner')

    def side(s, monthly):
        return {'cagr': round(_cagr(s) * 100, 2),
                'monthly_vol': round(float(monthly.std(ddof=1) * math.sqrt(12) * 100), 2),
                'max_drawdown': round(float((s / s.cummax() - 1).min() * 100), 2),
                'worst_month': round(float(monthly.min() * 100), 2)}

    var = float(rm.var(ddof=1))
    return {
        'model': side(model, mm), 'real': side(real, rm),
        'cagr_gap': round((_cagr(model) - _cagr(real)) * 100, 2),
        'monthly_corr': round(float(mm.corr(rm)), 3),
        'tracking_error': round(float((mm - rm).std(ddof=1) * math.sqrt(12) * 100), 2),
        'beta': round(float(np.cov(mm, rm, ddof=1)[0, 1] / var), 3) if var > 0 else None,
        'months': int(len(mm)),
    }


def replica(start: str, end: str, market: dict, level: float | None = None,
            use_curve: bool = True) -> pd.Series:
    """The PutWrite replica's equity curve on SPY, without costs (the index has none)."""
    keep = ob.VIX_TO_ATM
    if level is not None:
        ob.VIX_TO_ATM = level
    try:
        m = market if use_curve else {**market, 'curve': None}
        r = ob.run('SPY', start, end, PUTWRITE_LEGS, name='PutWrite replica', dte=30, alloc_pct=100,
                   initial_capital=10_000_000, spread_pct=0.0, commission=0.0, stock_slip=0.0, market=m)
    finally:
        ob.VIX_TO_ATM = keep
    if 'error' in r:
        raise RuntimeError(r['error'])
    return pd.Series([e['value'] for e in r['equity']],
                     index=pd.to_datetime([e['date'] for e in r['equity']]))


def fit_level(market: dict, real: pd.Series, grid=GRID) -> float:
    """The VIX_TO_ATM on the grid whose replica best matches the index's CAGR over FIT."""
    target = _cagr(real.reindex(replica(*FIT, market).index).ffill().bfill())
    return min(grid, key=lambda k: abs(_cagr(replica(*FIT, market, level=k)) - target))


def term_check(vix: pd.Series, curve: pd.DataFrame, level: float) -> list:
    """Modelled slope against the real VIX curve: mean ratio to 30-day IV, error and correlation."""
    out = []
    iv30 = (vix / 100 * level)
    for days in ob.CURVE_TENORS:
        real = curve[days].dropna()
        iv = iv30.reindex(real.index).dropna()
        real = real.reindex(iv.index)
        model = pd.Series([ob.model_term_ratio(x, days) for x in iv], index=iv.index)
        out.append({'days': days, 'from': real.index[0].date().isoformat(),
                    'real_mean': round(float(real.mean()), 3), 'model_mean': round(float(model.mean()), 3),
                    'mean_abs_error': round(float((real - model).abs().mean()), 3),
                    'corr': round(float(real.corr(model)), 2)})
    return out


def monthly_index(s: pd.Series) -> list:
    m = s.resample('ME').last()
    return [round(float(v) / float(s.iloc[0]) * 100, 2) for v in m]
