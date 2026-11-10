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

Writes frontend/data/options_quotes_check.json, which the Options page shows.

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
