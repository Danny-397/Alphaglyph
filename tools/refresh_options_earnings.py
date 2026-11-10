"""
Refresh backend/options_earnings.json: earnings report dates, with the time of day,
for the stocks people most often backtest options on.

    python tools/refresh_options_earnings.py            # the default universe
    python tools/refresh_options_earnings.py ROKU SNAP  # add or refresh names

The options backtester asks Yahoo first, but Yahoo refuses datacenter IPs (Render's
and GitHub's runners), so production reads this file. Run it from a home
connection after each earnings season and commit the result. A ticker that fails
keeps its previous rows.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'options_earnings.json')

UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'AMD', 'NFLX', 'ADBE', 'CRM', 'ORCL',
    'INTC', 'QCOM', 'MU', 'TXN', 'AMAT', 'LRCX', 'ARM', 'SMCI', 'PLTR', 'SNOW', 'SHOP', 'UBER', 'ABNB', 'COIN',
    'HOOD', 'SOFI', 'PYPL', 'SQ', 'XYZ', 'ROKU', 'SNAP', 'PINS', 'RBLX', 'DKNG', 'MSTR', 'MARA', 'RIVN', 'LCID',
    'NIO', 'BABA', 'JD', 'PDD', 'DIS', 'CMCSA', 'T', 'VZ', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'SCHW', 'V', 'MA',
    'AXP', 'BRK-B', 'XOM', 'CVX', 'OXY', 'COP', 'BA', 'CAT', 'DE', 'GE', 'F', 'GM', 'UPS', 'FDX', 'WMT', 'COST',
    'TGT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'KO', 'PEP', 'PG', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'UNH', 'CVS',
    'MRNA', 'GILD', 'AMGN', 'IBM', 'CSCO', 'NOW', 'PANW', 'CRWD', 'ZS', 'NET', 'DDOG', 'TTD', 'ENPH', 'FSLR',
]


def fetch(ticker: str) -> list | None:
    import yfinance as yf
    raw = yf.Ticker(ticker).get_earnings_dates(limit=100)
    if raw is None or raw.empty:
        return None
    rows = []
    for ts in raw.index:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert('America/New_York').tz_localize(None)
        when = 'amc' if ts.hour >= 16 else ('bmo' if 0 < ts.hour < 10 else None)
        rows.append([ts.strftime('%Y-%m-%d'), when])
    return sorted(rows)


def main(extra: list[str]) -> int:
    try:
        with open(PATH, encoding='utf-8') as fh:
            snap = json.load(fh)
    except (OSError, ValueError):
        snap = {}
    tickers = dict(snap.get('tickers') or {})
    done, failed = [], []
    for t in sorted(set(UNIVERSE) | {s.upper() for s in extra}):
        try:
            rows = fetch(t)
        except Exception as exc:                   # network, rate limit, delisted
            print(f'{t}: {exc}')
            rows = None
        if rows:
            tickers[t] = rows
            done.append(t)
        else:
            failed.append(t)
        time.sleep(0.5)
    if not done:
        print('Nothing refreshed; file left unchanged.')
        return 1
    out = {'captured_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
           'source': 'yfinance get_earnings_dates(limit=100); [date, "amc"|"bmo"|null]',
           'tickers': dict(sorted(tickers.items()))}
    with open(PATH, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(out, fh, separators=(',', ':'))
        fh.write('\n')
    print(f'Refreshed {len(done)}; kept previous rows for {len(failed)}: {", ".join(failed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
