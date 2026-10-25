"""
Refresh earnings_snapshot.json, the fallback earnings.py serves when Yahoo
refuses the live earnings-dates call (it routinely does from datacenter IPs,
including Render's).

    python refresh_earnings_snapshot.py            # default universe
    python refresh_earnings_snapshot.py AAPL KO    # add or refresh specific names

Run it from a connection Yahoo answers (a home connection always has; the
weekly GitHub Action tries a runner). A ticker that fails keeps its previous
rows, so a blocked run can never empty the snapshot -- it only fails to update
it. The script exits non-zero if no ticker at all refreshed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd

import earnings as earn
import pead

# Everything the PEAD study pools, plus the Predictor's quick-pick names.
UNIVERSE = sorted(set(pead.DEFAULT_UNIVERSE) | {'NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMZN', 'JPM'})


def _rows(frame: pd.DataFrame) -> list[dict]:
    def num(v):
        return None if pd.isna(v) else round(float(v), 4)
    return [{'date': ts.strftime('%Y-%m-%d'),
             'eps_est': num(r['eps_est']),
             'eps_actual': num(r['eps_actual']),
             'surprise_pct': num(r['surprise_pct'])}
            for ts, r in frame.iterrows()]


def main(extra: list[str]) -> int:
    try:
        with open(earn.SNAPSHOT_PATH, encoding='utf-8') as fh:
            snap = json.load(fh)
    except (OSError, ValueError):
        snap = {}
    tickers = dict(snap.get('tickers') or {})

    refreshed, failed = [], []
    for t in sorted(set(UNIVERSE) | {s.upper() for s in extra}):
        frame = earn._fetch_live_frame(t)
        if frame is None or frame['eps_actual'].notna().sum() == 0:
            failed.append(t)
            continue
        tickers[t] = _rows(frame)
        refreshed.append(t)

    if not refreshed:
        print(f'No ticker refreshed (failed: {", ".join(failed)}); snapshot left unchanged.')
        return 1
    if tickers == (snap.get('tickers') or {}):
        # Only the timestamp would change; skip the write so the weekly job
        # doesn't commit (and redeploy) when no earnings data moved.
        print(f'Refreshed {len(refreshed)}, no earnings data changed; snapshot left as is.')
        return 0

    out = {
        'captured_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
        'source': 'yfinance get_earnings_dates(limit=16)',
        'tickers': dict(sorted(tickers.items())),
    }
    with open(earn.SNAPSHOT_PATH, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
        fh.write('\n')
    print(f'Refreshed {len(refreshed)}: {", ".join(refreshed)}')
    if failed:
        print(f'Kept previous rows for {len(failed)}: {", ".join(failed)}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
