"""
Is an options strategy's edge real? The same tests AlphaGlyph applies to stock
strategies, applied to options.

- Deflated Sharpe on the leaderboard. Comparing all 19 presets and picking the
  best is 19 tries: the luckiest of 19 strategies with no edge would still show
  a positive Sharpe, so each row's Sharpe is tested against that bar instead of
  against zero (Bailey and Lopez de Prado, 2014).
- A settings sweep with PBO. Options traders tune days to expiry and exit rules,
  and every setting tried is another chance to fit noise. The sweep runs one
  strategy across a grid of settings, then CSCV asks how often the setting that
  looked best in one half of the history landed in the bottom half of the other
  (the probability of backtest overfitting), and deflates the best setting's
  Sharpe for the number tried.
"""

from __future__ import annotations

import numpy as np

import options_bt as ob
import stats as st
from cpcv import pbo_from_matrix

SWEEP_DTE = (7, 14, 30, 45, 60)
SWEEP_EXITS = (
    ('expiry', 'Hold to expiry'),
    ('profit50', 'Take 50% profit'),
    ('half', 'Close halfway to expiry'),
    ('stop100', 'Stop at a 100% loss'),
)
SWEEP_MAX_YEARS = 5                 # 20 backtests must finish inside the server's 120 s timeout
PBO_GROUPS = 10


def excess_returns(result: dict) -> np.ndarray:
    """Daily returns over the T-bill rate, from a backtest result."""
    v = np.array([e['value'] for e in result['equity']], dtype=float)
    rets = np.diff(v) / v[:-1]
    return rets - float(result.get('model', {}).get('rf_daily', 0.0))


def deflate(rows: list, returns: list) -> dict:
    """Add each row's Deflated Sharpe (as a percentage) for len(rows) tries; return the bar."""
    n = len(rows)
    bar = None
    for row, r in zip(rows, returns):
        d = st.deflated_sharpe_ratio(r, n_strategies=n)
        row['dsr'] = None if d['dsr'] is None else round(d['dsr'] * 100, 1)
        bar = d['sr_benchmark'] if d['sr_benchmark'] is not None else bar
    return {'n': n, 'sharpe_bar': None if bar is None else round(bar, 2)}


SENSITIVITY = (
    ('IV 10% lower', {'iv_scale': 0.9}),
    ('IV 10% higher', {'iv_scale': 1.1}),
    ('No skew', {'skew': False}),
)


def model_sensitivity(run_kw: dict, base: dict) -> dict:
    """Rerun a backtest with the volatility model nudged: how much of the result is the model?

    run_kw are the arguments of the base run, market included. Returns each variant's
    return figures and the spread of annual returns across all of them.
    """
    rows = []
    for label, change in SENSITIVITY:
        if change.get('skew') is False and not run_kw.get('skew', True):
            continue
        r = ob.run(**{**run_kw, **change})
        if 'error' in r:
            continue
        m = r['metrics']
        rows.append({'label': label, 'cagr': m['cagr'], 'sharpe': m['sharpe'],
                     'max_drawdown': m['max_drawdown'], 'total_return': m['total_return']})
    cagrs = [base['metrics']['cagr']] + [x['cagr'] for x in rows]
    return {'variants': rows, 'cagr_low': min(cagrs), 'cagr_high': max(cagrs),
            'cagr_range': round(max(cagrs) - min(cagrs), 2)}


def _exit_args(code: str, dte: int) -> dict:
    if code == 'profit50':
        return {'profit_take': 50.0}
    if code == 'half':
        return {'exit_dte': max(1, dte // 2)}
    if code == 'stop100':
        return {'stop_loss': 100.0}
    return {}


def sweep(ticker: str, start: str, end: str, legs: list, name: str, alloc_pct: float,
          market: dict, current_dte: int | None = None, **kw) -> dict:
    """Run one strategy over SWEEP_DTE x SWEEP_EXITS and test the best setting for overfitting.

    kw carries the shared settings (capital, costs, skew); the sweep sets dte and the exits.
    """
    for k in ('dte', 'exit_dte', 'profit_take', 'stop_loss'):
        kw.pop(k, None)
    cells, cols = [], []
    for dte in SWEEP_DTE:
        for code, label in SWEEP_EXITS:
            r = ob.run(ticker, start, end, legs, name=name, alloc_pct=alloc_pct, dte=dte, market=market,
                       **_exit_args(code, dte), **kw)
            if 'error' in r:
                return r
            m = r['metrics']
            cells.append({'dte': dte, 'exit': code, 'exit_label': label, 'cagr': m['cagr'],
                          'sharpe': m['sharpe'], 'max_drawdown': m['max_drawdown'], 'trades': m['trades']})
            cols.append(excess_returns(r))
    T = min(len(c) for c in cols)
    M = np.column_stack([c[:T] for c in cols])
    for c, col in zip(cells, M.T):
        c['flat'] = bool(np.allclose(col, col[0]))
    best = max(range(len(cells)), key=lambda i: cells[i]['sharpe'])
    d = st.deflated_sharpe_ratio(M[:, best], n_strategies=len(cells))
    pbo = pbo_from_matrix(M, n_groups=PBO_GROUPS, embargo=5, ticker=ticker)
    out = {
        'ticker': ticker, 'name': name, 'start': start, 'end': end,
        'dte': list(SWEEP_DTE), 'exits': [{'code': c, 'label': lbl} for c, lbl in SWEEP_EXITS],
        'cells': cells, 'n': len(cells), 'best': best,
        'best_dsr': None if d['dsr'] is None else round(d['dsr'] * 100, 1),
        'sharpe_bar': None if d['sr_benchmark'] is None else round(d['sr_benchmark'], 2),
        'current_dte': current_dte,
    }
    if pbo.get('available'):
        out['pbo'] = {k: pbo[k] for k in ('pbo', 'prob_oos_loss', 'n_combos', 'degradation')}
    else:
        out['pbo'] = None
        out['pbo_reason'] = pbo.get('reason')
    out['verdict'] = verdict(out)
    return out


def verdict(s: dict) -> str:
    """One plain paragraph on what the sweep says."""
    b = s['cells'][s['best']]
    head = (f"The best of the {s['n']} settings was {b['dte']} days to expiry, "
            f"{b['exit_label'].lower()}: a Sharpe of {b['sharpe']:.2f}.")
    if s['best_dsr'] is not None:
        head += (f" Tried {s['n']} ways, the luckiest setting of a strategy with no edge would show about "
                 f"{s['sharpe_bar']:.2f}, so the chance this one's edge is real is {s['best_dsr']:.0f}%"
                 f"{' (passes the 95% bar)' if s['best_dsr'] >= 95 else ''}.")
    p = s.get('pbo')
    if p:
        pct = p['pbo'] * 100
        if pct >= 50:
            tail = (f" Splitting the history in two {p['n_combos']} ways, the setting that looked best in one half "
                    f"landed in the bottom half of the other {pct:.0f}% of the time: choosing the setting by its "
                    "backtest is no better than guessing.")
        elif pct >= 25:
            tail = (f" The best-looking setting landed in the bottom half out of sample {pct:.0f}% of the time: "
                    "a real risk that the chosen setting is fitted to this history.")
        else:
            tail = (f" The best-looking setting landed in the bottom half out of sample only {pct:.0f}% of the "
                    "time, so the ranking of settings mostly holds up.")
        head += tail
    elif s.get('pbo_reason'):
        head += f" Overfitting test skipped: {s['pbo_reason']}"
    if s['best_dsr'] is not None and s['best_dsr'] >= 95:
        head += (" These tests check how the setting was chosen, not the prices: the model understates "
                 "sudden losses for option sellers (see the price check), so a pass is a reason to test "
                 "with real quotes, not proof of an edge.")
    return head
