"""
How close are the options backtester's model prices to real option returns?

No free source has historical option quotes, but Cboe publishes the returns of
real option strategies as indices. The S&P 500 PutWrite index (PUT) sells a
one-month at-the-money put every month and keeps the strike in T-bills; Yahoo
carries its full daily history. The backtester can run the same strategy on SPY
with its model prices, so the two return streams can be compared directly.

Fit, then test
--------------
The volatility level, VIX_TO_ATM (at-the-money IV as a share of the VIX), is
fitted on 2007-2015 so that the PutWrite replica's annual return matches the
index's; the skew is then fitted the same way (below). The comparison that
matters is 2016 on, which the fits never saw.

The replica is not identical to the index: the index trades SPX options at
Cboe's special opening quotation on the third Friday, the replica trades SPY
options at the close every four to five weeks. The fitted level absorbs the
average effect of those differences, so it is an effective level, not a
measurement of implied volatility.

The term structure is checked the same way: the modelled slope (used only where
the real VIX curve is missing) against the VIX9D, VIX3M and VIX6M.

Five more Cboe indices test what the PutWrite index can't: out-of-the-money
strikes and calls. Cboe publishes their daily history as CSV files. The skew is
fitted on 2007-2015 against two of them (PPUT for the put side, BXMD for the
call side); everything is then compared on 2016 on.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import options_bt as ob

PUTWRITE_LEGS = [{'type': 'put', 'side': 'short', 'qty': 1, 'strike': {'mode': 'atm'}}]


def _leg(kind, side, strike=None, dte=None):
    leg = {'type': kind, 'side': side, 'qty': 1}
    if strike:
        leg['strike'] = strike
    if dte:
        leg['dte'] = dte
    return leg


# Cboe strategy indices and their replicas on SPY (each index's published method, simplified).
INDICES = {
    'PUT': {'name': 'PutWrite', 'what': 'sell a one-month at-the-money put', 'legs': PUTWRITE_LEGS},
    'BXM': {'name': 'BuyWrite', 'what': 'own the index, sell a one-month at-the-money call',
            'legs': [_leg('stock', 'long'), _leg('call', 'short', {'mode': 'atm'})]},
    'BXMD': {'name': '30-delta BuyWrite', 'what': 'own the index, sell a one-month 30-delta call',
             'legs': [_leg('stock', 'long'), _leg('call', 'short', {'mode': 'delta', 'value': 0.30})]},
    'PPUT': {'name': '5% Put Protection', 'what': 'own the index, buy a one-month put 5% out of the money',
             'legs': [_leg('stock', 'long'), _leg('put', 'long', {'mode': 'pct', 'value': -5})]},
    'CLL': {'name': '95-110 Collar', 'what': 'own the index, buy a three-month put 5% below, sell a one-month '
                                            'call 10% above',
            'legs': [_leg('stock', 'long'), _leg('put', 'long', {'mode': 'pct', 'value': -5}, dte=90),
                     _leg('call', 'short', {'mode': 'pct', 'value': 10})]},
    'CNDR': {'name': 'Iron Condor', 'what': 'sell one-month 20-delta puts and calls, buy 5-delta wings',
             'legs': [_leg('put', 'short', {'mode': 'delta', 'value': 0.20}),
                      _leg('put', 'long', {'mode': 'delta', 'value': 0.05}),
                      _leg('call', 'short', {'mode': 'delta', 'value': 0.20}),
                      _leg('call', 'long', {'mode': 'delta', 'value': 0.05})]},
}
SKEW_GRID_PUT = [round(0.10 + 0.025 * i, 3) for i in range(17)]     # 0.100 ... 0.500
SKEW_GRID_CALL = [round(0.025 * i, 3) for i in range(17)]           # 0.000 ... 0.400
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
            use_curve: bool = True, legs: list | None = None, skew: tuple | None = None,
            schedule: str = 'monthly') -> pd.Series:
    """A Cboe index replica's equity curve on SPY (PutWrite by default), without costs.

    Sized by notional, one unit per 100 shares' worth of SPY, and rolled on the third-Friday
    monthly expiries, as the indices are (schedule='weekly' rolls every four to five weeks
    instead, for comparison). `level` and `skew` (put, call) temporarily replace the
    model's settings, for fitting.
    """
    keep = (ob.VIX_TO_ATM, ob.SKEW_PUT, ob.SKEW_CALL)
    if level is not None:
        ob.VIX_TO_ATM = level
    if skew is not None:
        ob.SKEW_PUT, ob.SKEW_CALL = skew
    try:
        m = market if use_curve else {**market, 'curve': None}
        r = ob.run('SPY', start, end, legs or PUTWRITE_LEGS, name='Cboe index replica', dte=30, alloc_pct=100,
                   initial_capital=10_000_000, spread_pct=0.0, commission=0.0, stock_slip=0.0, market=m,
                   sizing='notional', expiries=schedule)
    finally:
        ob.VIX_TO_ATM, ob.SKEW_PUT, ob.SKEW_CALL = keep
    if 'error' in r:
        raise RuntimeError(r['error'])
    return pd.Series([e['value'] for e in r['equity']],
                     index=pd.to_datetime([e['date'] for e in r['equity']]))


def fit_level(market: dict, real: pd.Series, grid=GRID) -> float:
    """The VIX_TO_ATM on the grid whose replica best matches the index's CAGR over FIT."""
    target = _cagr(real.reindex(replica(*FIT, market).index).ffill().bfill())
    return min(grid, key=lambda k: abs(_cagr(replica(*FIT, market, level=k)) - target))


def fit_skew(market: dict, indices: dict) -> dict:
    """Put and call skew on their grids that best match PPUT and BXMD over FIT (separately:
    the PPUT put is out of the money and the BXMD call is out of the money, so each index
    depends on one side of the skew)."""
    def best(code, grid, side):
        real = indices[code]
        legs = INDICES[code]['legs']

        def miss(v):
            sk = (v, ob.SKEW_CALL) if side == 'put' else (ob.SKEW_PUT, v)
            eq = replica(*FIT, market, legs=legs, skew=sk)
            return abs(_cagr(eq) - _cagr(real.reindex(eq.index).ffill().bfill()))
        return min(grid, key=miss)
    return {'put': best('PPUT', SKEW_GRID_PUT, 'put'), 'call': best('BXMD', SKEW_GRID_CALL, 'call')}


def index_checks(market: dict, indices: dict, start: str, end: str, before: tuple | None = None) -> list:
    """Every index against its replica over [start, end], with the model as it stands, and the
    annual-return gap with the skew `before` (put, call) the fit, for comparison."""
    out = []
    for code, spec in INDICES.items():
        real = indices.get(code)
        if real is None or real.index[0] > pd.Timestamp(start):
            continue
        eq = replica(start, end, market, legs=spec['legs'])
        prev = None
        if before is not None:
            prev = compare(replica(start, end, market, legs=spec['legs'], skew=before), real)['cagr_gap']
        wk = compare(replica(start, end, market, legs=spec['legs'], schedule='weekly'), real)
        out.append({'code': code, 'name': spec['name'], 'what': spec['what'], **compare(eq, real),
                    'cagr_gap_before_skew_fit': prev,
                    'weekly_schedule': {'cagr_gap': wk['cagr_gap'], 'monthly_corr': wk['monthly_corr'],
                                        'tracking_error': wk['tracking_error'],
                                        'worst_month': wk['model']['worst_month']},
                    'chart': {'model': monthly_index(eq),
                              'real': monthly_index(real.reindex(eq.index).ffill().bfill())}})
    return out


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
