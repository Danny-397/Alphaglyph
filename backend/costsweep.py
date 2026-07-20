"""
Transaction-cost sensitivity sweep.

The backtest engine already charges commission + slippage on every trade side.
This module answers the question that actually matters: *how much of the edge
survives realistic frictions, and at what cost level does it disappear entirely?*

It reruns the same strategy across a grid of per-side cost levels (0 bp → 100 bp)
and reports the net-return / Sharpe decay curve, then solves for two break-even
points by linear interpolation:

  * break-even vs zero      — the cost at which the strategy stops making money
  * break-even vs benchmark — the cost at which its edge over buy-and-hold (SPY)
                              vanishes; i.e. where the "alpha" is entirely eaten
                              by frictions

A strategy whose alpha dies at 5 bp is a backtest artefact; one that still beats
the benchmark at 50 bp is far more believable. That gap is the whole point.
"""

from __future__ import annotations

import logging

import backtest as backtester

logger = logging.getLogger(__name__)

# Per-side cost grid, in basis points (1 bp = 0.01%). Dense where it matters
# (real-world large-cap costs are ~3–10 bp/side) and stretched out to an
# unrealistic 100 bp to show the full decay.
_COST_GRID_BPS = [0, 3, 5, 10, 15, 20, 30, 50, 75, 100]


def _interp_crossing(xs: list[float], ys: list[float], target: float) -> float | None:
    """First x (ascending) where the curve y crosses `target`, linearly
    interpolated. None if it never crosses within the grid."""
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1] - target, ys[i] - target
        if y0 == 0:
            return xs[i - 1]
        if y0 > 0 >= y1 or y0 < 0 <= y1:          # sign change → crossing
            if y1 == y0:
                return xs[i]
            t = y0 / (y0 - y1)
            return round(xs[i - 1] + t * (xs[i] - xs[i - 1]), 2)
    return None


def run_cost_sweep(strategy: str, tickers: list[str], start_date: str,
                   end_date: str, initial_capital: float = 100_000.0,
                   risk_tolerance: str = 'moderate',
                   custom_rules: dict | None = None) -> dict:
    """Sweep per-side transaction cost and report the edge-decay curve."""
    points: list[dict] = []
    benchmark = None

    for bps in _COST_GRID_BPS:
        cost = bps / 10_000.0            # bp → fraction; charged entirely as commission
        try:
            r = backtester.run_backtest(
                strategy, tickers, start_date, end_date,
                initial_capital, False, risk_tolerance,
                commission_pct=cost, slippage_pct=0.0,
                custom_rules=custom_rules)
        except Exception as exc:
            logger.warning('cost sweep failed at %d bp: %s', bps, exc)
            continue
        if 'error' in r:
            return {'available': False, 'reason': r['error']}
        m = r['metrics']
        if benchmark is None:
            benchmark = m.get('benchmark_return')
        points.append({
            'cost_bps':      bps,
            'net_return':    m.get('total_return'),
            'sharpe':        m.get('sharpe_ratio'),
            'total_trades':  m.get('total_trades'),
            'total_costs':   m.get('total_costs'),
            'excess_return': round((m.get('total_return') or 0) - (benchmark or 0), 2),
        })

    if len(points) < 2:
        return {'available': False, 'reason': 'Not enough data to run the sweep.'}

    xs      = [p['cost_bps'] for p in points]
    nets    = [p['net_return'] for p in points]
    excess  = [p['excess_return'] for p in points]

    be_zero  = _interp_crossing(xs, nets, 0.0)
    be_bench = _interp_crossing(xs, excess, 0.0)

    return {
        'available':       True,
        'strategy':        strategy,
        'tickers':         tickers,
        'benchmark_return': round(benchmark, 2) if benchmark is not None else None,
        'points':          points,
        'breakeven_zero_bps':  be_zero,
        'breakeven_bench_bps': be_bench,
        'verdict':         _verdict(points, be_zero, be_bench),
    }


def _verdict(points: list[dict], be_zero, be_bench) -> str:
    beats_at_0 = (points[0]['excess_return'] or 0) > 0
    if not beats_at_0:
        return ("Even at zero cost the strategy doesn't beat buy-and-hold — "
                "there's no edge for frictions to erode.")
    if be_bench is None:
        return ("The edge over buy-and-hold survives the entire cost grid "
                "(up to 100 bp/side) — unusually robust; double-check the trade "
                "count is realistic.")
    if be_bench <= 5:
        return (f"The edge over buy-and-hold vanishes at just {be_bench:.0f} bp/side "
                "— well inside real-world costs, so this 'alpha' is essentially a "
                "frictionless backtest artefact.")
    if be_bench <= 20:
        return (f"The edge over buy-and-hold survives to {be_bench:.0f} bp/side, "
                "around the upper end of realistic large-cap costs — plausible but "
                "cost-sensitive.")
    return (f"The edge over buy-and-hold persists up to {be_bench:.0f} bp/side, "
            "comfortably beyond realistic costs — a genuinely robust result.")
