#!/usr/bin/env python3
"""Check the options backtester's model prices against Cboe's real PutWrite index.

Fits VIX_TO_ATM on 2007-2015, tests on 2016 to today, checks the modelled term
structure against the real VIX curve, and writes frontend/data/options_validation.json,
which the Options page shows. See backend/options_validation.py for the method.

USAGE
    python tools/options_validation.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

import features as feat                  # noqa: E402
import options_bt as ob                  # noqa: E402
import options_validation as ov          # noqa: E402

OUT = ROOT / 'frontend' / 'data' / 'options_validation.json'


def closes(sym: str, start: str, end: str) -> pd.Series:
    df = feat.fetch_ohlcv(sym, start=start, end=end)
    if df is None or df.empty:
        raise SystemExit(f'no data for {sym}')
    s = df['Close'].astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


def main() -> None:
    end = date.today().isoformat()
    market = ob.load_market('SPY', ov.FIT[0], end)
    if market.get('curve') is None:
        raise SystemExit('the VIX curve did not download; try again')
    put = closes('^PUT', '2006-10-01', end)

    fitted = ov.fit_level(market, put)
    print(f'fitted VIX_TO_ATM on {ov.FIT[0]}..{ov.FIT[1]}: {fitted} (code uses {ob.VIX_TO_ATM})')
    if abs(fitted - ob.VIX_TO_ATM) > 1e-9:
        print('  -> update VIX_TO_ATM in backend/options_bt.py and rerun')

    test = ov.replica(ov.TEST_START, end, market)
    real = put.reindex(test.index).ffill().bfill()
    old = ov.replica(ov.TEST_START, end, market, level=0.87, use_curve=False)
    fit_rep = ov.replica(*ov.FIT, market)
    out = {
        'generated': end,
        'index': 'Cboe S&P 500 PutWrite Index (PUT)',
        'replica': 'SPY: sell a 30-day at-the-money put at the close, hold to expiry, keep the strike in '
                   'T-bills, no costs (the index has none)',
        'level': {'fitted': fitted, 'used': ob.VIX_TO_ATM, 'previous': 0.87,
                  'fit_window': list(ov.FIT), 'grid': [ov.GRID[0], ov.GRID[-1]]},
        'fit': {'window': list(ov.FIT), **ov.compare(fit_rep, put)},
        'test': {'window': [ov.TEST_START, test.index[-1].date().isoformat()], **ov.compare(test, put)},
        'previous_model_test': ov.compare(old, put),
        'chart': {'months': [d.strftime('%Y-%m') for d in test.resample('ME').last().index],
                  'model': ov.monthly_index(test), 'real': ov.monthly_index(real)},
        'term_structure': ov.term_check(closes('^VIX', ov.FIT[0], end), market['curve'], ob.VIX_TO_ATM),
    }
    OUT.write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    t = out['test']
    print(f"test {t['window']}: model CAGR {t['model']['cagr']}% vs real {t['real']['cagr']}% "
          f"(gap {t['cagr_gap']:+}), monthly corr {t['monthly_corr']}, tracking error {t['tracking_error']}%")
    p = out['previous_model_test']
    print(f"previous model on the same window: gap {p['cagr_gap']:+}, corr {p['monthly_corr']}")
    for r in out['term_structure']:
        print('term', r)
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
