"""
How close are the backtester's model option prices to real quotes?

The options backtester prices every option with a model (backend/options_bt.py), because
no free source has historical option quotes. Tradeski's daily snapshot job has kept the
closing option chains of twelve symbols since 2026-11-06 (data/options-archive/ in the
Tradeski repo). This prices each of those quoted contracts with the backtester's model, on
the same day, and compares it with the real bid/ask midpoint, so the model's error is
measured on real quotes and keeps being measured as the archive grows.

    python tools/options_quotes_check.py                  # read the archive from GitHub
    python tools/options_quotes_check.py --source DIR     # or from a local copy

Writes frontend/data/options_quotes_check.json, which the Options page shows, and
backend/options_iv_calibration.json: an opt-in volatility factor per single stock (see
calibration() below), which the backtester applies when the user asks for it.

What is compared: out-of-the-money calls and puts (in-the-money options are mostly
intrinsic value and early exercise; out of the money is where the volatility model
matters) with a two-sided quote, a spread under half the midpoint, at least 2 days to
expiry and a midpoint of at least $0.05. Errors are in implied-volatility points: the
model's volatility for that strike and expiry minus the volatility the midpoint implies,
both with the same rate and dividend yield, and as a share of the midpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'backend'))

import options_bt as ob  # noqa: E402

ARCHIVE = 'https://raw.githubusercontent.com/Danny-397/Tradeski/main/data/options-archive/'
OUT = ROOT / 'frontend' / 'data' / 'options_quotes_check.json'
MIN_DAYS, MIN_MID, MAX_REL_SPREAD = 2, 0.05, 0.5
DTE_BUCKETS = ((2, 10, '2-10 days'), (11, 35, '11-35 days'), (36, 75, '36-75 days'))
DELTA_BUCKETS = ((0.40, 0.60, 'at the money (40-60 delta)'), (0.15, 0.40, '15-40 delta'),
                 (0.0, 0.15, 'under 15 delta'))
ATM_TARGET_DAYS = 30
CALIBRATION_OUT = ROOT / 'backend' / 'options_iv_calibration.json'
# Tradeski's nightly chains for ~90 symbols (one day, replaced each night): the source for the
# earnings-move check, which needs reports ahead of many stocks, not the archive's twelve.
OPTIONS_DATA = 'https://raw.githubusercontent.com/Danny-397/Tradeski/options-data/options/'
CAL_MIN_CONTRACTS = 20
CAL_CLIP = (0.7, 2.0)


def read(source: str, name: str) -> dict:
    if source.startswith('http'):
        with urllib.request.urlopen(source + name, timeout=30) as resp:
            return json.loads(resp.read().decode('utf-8'))
    return json.loads((Path(source) / name).read_text(encoding='utf-8'))


def quote_ok(bid, ask) -> float | None:
    """The midpoint of a usable two-sided quote, or None."""
    if bid is None or ask is None or bid <= 0 or ask < bid:
        return None
    mid = 0.5 * (bid + ask)
    if mid < MIN_MID or (ask - bid) / mid > MAX_REL_SPREAD:
        return None
    return mid


def compare_day(sym: str, day: str, entry: dict, c: dict) -> list:
    """One row per usable out-of-the-money contract: model vs market."""
    rows = []
    S = c['S']
    d0 = date.fromisoformat(day)
    for exp, sides in entry['expiries'].items():
        dte = (date.fromisoformat(exp) - d0).days
        if dte < MIN_DAYS:
            continue
        T = dte / 365.0
        report = c['next_report'] is not None and c['next_report'] <= date.fromisoformat(exp)
        for kind, key in (('call', 'calls'), ('put', 'puts')):
            for k, bid, ask, vol, oi in sides.get(key, []):
                if (kind == 'call' and k < S) or (kind == 'put' and k > S):
                    continue
                mid = quote_ok(bid, ask)
                if mid is None:
                    continue
                mkt_iv = ob.implied_vol(kind, mid, S, k, T, c['r'], c['q'])
                if mkt_iv is None:
                    continue
                price, model_iv = ob.model_price(kind, c, k, T)
                delta = abs(math.exp(-c['q'] * T) * ob.bs_delta(kind, S * math.exp(-c['q'] * T), k, T, c['r'], mkt_iv))
                rows.append({'symbol': sym, 'day': day, 'expiry': exp, 'dte': dte, 'kind': kind, 'strike': k,
                             'mid': mid, 'bid': bid, 'ask': ask, 'model': price, 'model_iv': model_iv,
                             'market_iv': mkt_iv, 'delta': delta, 'report': report,
                             'iv_err': model_iv - mkt_iv, 'price_err': price / mid - 1,
                             'inside': bid <= price <= ask})
    return rows


def summarize(rows: list) -> dict | None:
    if not rows:
        return None
    iv = np.array([r['iv_err'] for r in rows]) * 100
    pe = np.array([r['price_err'] for r in rows]) * 100
    return {'n': len(rows), 'iv_bias': round(float(np.median(iv)), 2),
            'iv_mae': round(float(np.median(np.abs(iv))), 2),
            'price_bias': round(float(np.median(pe)), 1), 'price_mae': round(float(np.median(np.abs(pe))), 1),
            'inside_spread': round(float(np.mean([r['inside'] for r in rows])) * 100, 1)}


def atm_point(rows: list) -> dict | None:
    """Model vs market at-the-money IV near 30 days: the expiry closest to 30 days, nearest strike."""
    if not rows:
        return None
    best_exp = min({r['expiry'] for r in rows}, key=lambda e: abs(next(r['dte'] for r in rows if r['expiry'] == e)
                                                                    - ATM_TARGET_DAYS))
    near = sorted((r for r in rows if r['expiry'] == best_exp), key=lambda r: abs(r['delta'] - 0.5))[:2]
    return {'expiry': best_exp, 'model': round(float(np.mean([r['model_iv'] for r in near])) * 100, 1),
            'market': round(float(np.mean([r['market_iv'] for r in near])) * 100, 1)}


def calib_row(r: dict) -> bool:
    """The contracts that set a stock's volatility level: near the money (25-60 delta), 11-60
    days out, and not spanning a report (the model prices reports separately)."""
    return 11 <= r['dte'] <= 60 and 0.25 <= r['delta'] <= 0.60 and not r['report'] and r['model_iv'] > 0


def fit_factors(rows: list) -> dict:
    """Per single stock: the median market/model IV ratio over its calibration contracts."""
    out = {}
    for sym in sorted({r['symbol'] for r in rows} - set(ob.INDEX_ETFS)):
        x = [r['market_iv'] / r['model_iv'] for r in rows if r['symbol'] == sym and calib_row(r)]
        if len(x) >= CAL_MIN_CONTRACTS:
            out[sym] = {'factor': round(float(np.clip(np.median(x), *CAL_CLIP)), 3), 'contracts': len(x)}
    return out


def _miss(rows: list, factor: float) -> float:
    return round(float(np.median([abs(r['model_iv'] * factor - r['market_iv']) for r in rows])) * 100, 2)


def calibration(rows: list) -> dict | None:
    """Fit the factors on the first half of the days and score them on the second half (every
    usable contract, not only the calibration set), then refit on every day for use.

    Index funds are left out: SPY's level is fitted to Cboe's indices over 2007-2015, and on
    held-out days a factor made the index funds worse, not better.
    """
    days = sorted({r['day'] for r in rows})
    if len(days) < 4:
        return None
    fit_days = set(days[:(len(days) + 1) // 2])
    fit = fit_factors([r for r in rows if r['day'] in fit_days])
    test = [r for r in rows if r['day'] not in fit_days and r['symbol'] not in ob.INDEX_ETFS]
    by_symbol, before, after = {}, [], []
    for sym, f in fit.items():
        t = [r for r in test if r['symbol'] == sym]
        if not t:
            continue
        by_symbol[sym] = {'fit_factor': f['factor'], 'before': _miss(t, 1.0), 'after': _miss(t, f['factor']),
                          'contracts': len(t)}
        before += [abs(r['model_iv'] - r['market_iv']) for r in t]
        after += [abs(r['model_iv'] * f['factor'] - r['market_iv']) for r in t]
    return {
        'days': [days[0], days[-1]], 'n_days': len(days),
        'tickers': fit_factors(rows),
        'validation': {'fit_days': [min(fit_days), max(fit_days)],
                       'test_days': [min(set(days) - fit_days), max(set(days) - fit_days)],
                       'before': round(float(np.median(before)) * 100, 2) if before else None,
                       'after': round(float(np.median(after)) * 100, 2) if after else None,
                       'improved': sum(1 for v in by_symbol.values() if v['after'] < v['before']),
                       'by_symbol': by_symbol},
        'method': ('Median ratio of market to model implied volatility, near-the-money contracts (25-60 delta) '
                   '11-60 days out that do not span a report; single stocks only.'),
    }


def atm_market_iv(chain: dict, S: float, T: float, r: float, q: float) -> float | None:
    """Mean implied volatility of the call and the put nearest the money, from their midpoints."""
    ivs = []
    for kind, key in (('call', 'calls'), ('put', 'puts')):
        quoted = [x for x in chain.get(key, []) if quote_ok(x.get('bid'), x.get('ask'))]
        if quoted:
            x = min(quoted, key=lambda x: abs(x['strike'] - S))
            iv = ob.implied_vol(kind, 0.5 * (x['bid'] + x['ask']), S, x['strike'], T, r, q)
            if iv:
                ivs.append(iv)
    return float(np.mean(ivs)) if ivs else None


def implied_move(chains: dict, today: date, react: date, S: float, r: float, q: float) -> float | None:
    """The report move the market prices: the variance the first expiry after the reaction day
    carries beyond the last expiry before it, at that earlier expiry's volatility (the usual
    straddle-free estimate). None without an expiry on each side or a positive difference."""
    exps = sorted(chains)
    before = [e for e in exps if (date.fromisoformat(e) - today).days >= 3 and date.fromisoformat(e) < react]
    after = [e for e in exps if date.fromisoformat(e) >= react]
    if not before or not after:
        return None
    e1, e2 = before[-1], after[0]
    T1, T2 = ((date.fromisoformat(e) - today).days / 365.0 for e in (e1, e2))
    s1, s2 = atm_market_iv(chains[e1], S, T1, r, q), atm_market_iv(chains[e2], S, T2, r, q)
    if not s1 or not s2:
        return None
    ev = (s2 * s2 - s1 * s1) * T2
    return math.sqrt(ev) if ev > 0 else None


def earnings_moves(base: str = OPTIONS_DATA) -> dict | None:
    """The report move the market prices tonight against the one the model assumes (the mean
    square of the previous eight reactions), for every single stock with a report between two
    saved expiries. Measures whether the model's earnings premium is right on average."""
    try:
        symbols = sorted(read(base, 'index.json')['symbols'])
    except Exception:
        return None
    rows = []
    for sym in symbols:
        if sym in ob.INDEX_ETFS:
            continue
        events = ob.earnings_events(sym)
        if not events:
            continue
        try:
            d = read(base, f'{sym}.json')
        except Exception:
            continue
        today = datetime.fromisoformat(d['captured_at']).astimezone(timezone.utc).date()
        nxt = [e for e in events if e['date'].date() >= today]
        if not nxt or not d.get('spot'):
            continue
        e = nxt[0]
        react = ob.upcoming_reactions(pd.Series([1.0], index=[pd.Timestamp(today)]), [e])
        if not react:
            continue
        react = react[0].date()
        m = ob.load_market(sym, (today - timedelta(days=3 * 365)).isoformat(), today.isoformat())
        if 'error' in m:
            continue
        ctx = ob.pricing_context(m, sym)
        fut = [x for x in ctx['react'] if x >= react]
        if not fut:
            continue
        model = math.sqrt(ctx['ev_var'][pd.Timestamp(fut[0])])
        r = float(ctx['r'].iloc[-1])
        market = implied_move(d['chains'], today, react, float(d['spot']), r, float(d.get('dividend_yield') or 0.0))
        if market and model > 0:
            rows.append({'symbol': sym, 'report': e['date'].date().isoformat(), 'market': round(market * 100, 2),
                         'model': round(model * 100, 2), 'ratio': round(market / model, 3)})
    if not rows:
        return None
    ratios = np.array([x['ratio'] for x in rows])
    return {'n': len(rows), 'median_ratio': round(float(np.median(ratios)), 2),
            'iqr': [round(float(np.percentile(ratios, 25)), 2), round(float(np.percentile(ratios, 75)), 2)],
            'rows': rows}


def build(source: str) -> dict:
    days = read(source, 'index.json')['days']
    archive = {d: read(source, f'{d}.json') for d in days}
    symbols = sorted({s for a in archive.values() for s in a['symbols']})
    lo = days[0]
    hi = (date.fromisoformat(days[-1]) + timedelta(days=3)).isoformat()
    rows, skipped, atm, notes = [], [], {}, {}
    for sym in symbols:
        m = ob.load_market(sym, lo, hi)
        if 'error' in m:
            skipped.append(f'{sym}: {m["error"]}')
            continue
        ctx = ob.pricing_context(m, sym)
        if m.get('earnings') is None:
            notes[sym] = 'no earnings dates: priced without an earnings premium'
        for d in days:
            entry = archive[d]['symbols'].get(sym)
            if entry is None:
                continue
            c = ob.context_at(ctx, d)
            if c is None:
                skipped.append(f'{sym} {d}: no close in the price data yet')
                continue
            if abs(c['S'] / entry['spot'] - 1) > 0.01:      # the archive's spot and the close disagree
                skipped.append(f'{sym} {d}: archive spot {entry["spot"]:.2f} vs close {c["S"]:.2f}')
                continue
            day_rows = compare_day(sym, d, entry, c)
            rows += day_rows
            pt = atm_point(day_rows)
            if pt:
                atm.setdefault(sym, []).append({'day': d, **pt})

    def bucket(label, pick):
        s = summarize([r for r in rows if pick(r)])
        return {'label': label, **s} if s else None

    groups = []
    for lo_d, hi_d, lab in DELTA_BUCKETS:
        for kind in ('put', 'call'):
            groups.append(bucket(f'{kind}s, {lab}', lambda r, a=lo_d, b=hi_d, k=kind:
                                 r['kind'] == k and a <= r['delta'] < b))
    by_dte = [bucket(lab, lambda r, a=a, b=b: a <= r['dte'] <= b) for a, b, lab in DTE_BUCKETS]
    stocks = [r for r in rows if r['symbol'] not in ob.INDEX_ETFS]
    by_report = [bucket('single stocks, expiring before the next report', lambda r: r in stocks and not r['report']),
                 bucket('single stocks, holding through a report', lambda r: r in stocks and r['report']),
                 bucket('index ETFs (no reports)', lambda r: r['symbol'] in ob.INDEX_ETFS)]
    per_symbol = [{'symbol': s, **summarize([r for r in rows if r['symbol'] == s]),
                   'atm': atm.get(s, [])[-1] if atm.get(s) else None, 'note': notes.get(s)}
                  for s in symbols if any(r['symbol'] == s for r in rows)]
    used_days = sorted({r['day'] for r in rows})
    return {
        'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'source': 'Tradeski data/options-archive (closing chains saved after each US close)',
        'days': used_days, 'symbols': [p['symbol'] for p in per_symbol],
        'overall': summarize(rows),
        'by_moneyness': [g for g in groups if g], 'by_dte': [g for g in by_dte if g],
        'by_report': [g for g in by_report if g], 'by_symbol': per_symbol,
        'atm': atm, 'skipped': skipped,
        'calibration': calibration(rows),
        'earnings_moves': earnings_moves(),
        'report_symbols': sorted({r['symbol'] for r in rows if r['report']}),
        'filters': (f'Out-of-the-money calls and puts within 20% of the price with a two-sided quote, a spread under '
                    f'{MAX_REL_SPREAD * 100:.0f}% of the midpoint, a midpoint of at least ${MIN_MID:.2f} and at '
                    f'least {MIN_DAYS} days to expiry.'),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('--source', default=ARCHIVE, help='archive folder or URL (default: Tradeski on GitHub)')
    args = ap.parse_args()
    src = args.source if args.source.startswith('http') else str(Path(args.source).resolve())
    if src.startswith('http') and not src.endswith('/'):
        src += '/'
    out = build(src)
    if not out['overall']:
        print('No usable quotes; file left unchanged.')
        return 1
    cal = out.get('calibration')
    try:
        old_cal = json.loads(CALIBRATION_OUT.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        old_cal = {}
    if cal and {**old_cal, 'generated': None} != {**cal, 'generated': None}:
        CALIBRATION_OUT.write_text(json.dumps({'generated': out['generated'], **cal}, indent=1) + '\n',
                                   encoding='utf-8', newline='\n')
        v = cal['validation']
        print(f"calibration: {len(cal['tickers'])} stocks; held-out typical miss {v['before']} -> {v['after']} pts, "
              f"{v['improved']} of {len(v['by_symbol'])} improved")
    try:
        old = json.loads(OUT.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        old = {}
    if {**old, 'generated': None} == {**out, 'generated': None}:
        print('No new quotes; file left unchanged.')
        return 0
    OUT.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8', newline='\n')
    o = out['overall']
    print(f"{o['n']} contracts over {len(out['days'])} day(s), {len(out['symbols'])} symbols: model IV minus market "
          f"IV median {o['iv_bias']:+.2f} pts (typical miss {o['iv_mae']:.2f}); price {o['price_bias']:+.1f}% "
          f"(typical miss {o['price_mae']:.1f}%); {o['inside_spread']:.0f}% inside the bid/ask.")
    for s in out['skipped']:
        print('  skipped', s)
    return 0


if __name__ == '__main__':
    sys.exit(main())
