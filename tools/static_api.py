#!/usr/bin/env python3
"""Save the backend's answers to a few GET endpoints as static files.

The site is on Vercel and opens in under a second, but the API runs on Render's
free tier, which sleeps after 15 idle minutes and takes 30-60 s to wake. A page
that calls the API as it opens makes a cold visitor wait for the whole boot.
These answers are the same for everyone, so the pages read a saved copy from
frontend/data/api/ first and only ask the live backend when the visitor wants
something the copy can't give (a different ticker list, a refresh).

Two groups, because they change at different rates:

  fixed   /api/ml/info and /api/options/presets. They change only when the code
          or the model changes. backend/tests/test_static_api.py fails if the
          committed copy no longer matches, so a PR that changes either has to
          rerun this script.
  daily   the Screener's regime and default scan, the Predictor's track record
          (calibration) and the PEAD study. They depend on market data, so
          .github/workflows/static-api.yml reruns them after each close, on the
          GitHub runner (not Render, so it costs no free-tier hours).

USAGE
    python tools/static_api.py fixed
    python tools/static_api.py daily
    python tools/static_api.py all
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'frontend' / 'data' / 'api'
sys.path.insert(0, str(ROOT / 'backend'))

# The default Screener list. frontend/app.js uses the same seven.
SCREENER_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'JPM', 'SPY']

FIXED = {
    'ml_info.json': '/api/ml/info',
    'options_presets.json': '/api/options/presets',
}


def client():
    import app as app_module
    app_module.limiter.enabled = False
    return app_module.app.test_client()


def get(c, path: str) -> dict:
    r = c.get(path)
    if r.status_code != 200:
        raise SystemExit(f'{path} answered {r.status_code}: {r.get_data(as_text=True)[:200]}')
    return r.get_json()


def fixed_payloads(c=None) -> dict[str, dict]:
    c = c or client()
    return {name: get(c, path) for name, path in FIXED.items()}


def write(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=1, sort_keys=True) + '\n', encoding='utf-8')
    print('wrote', (OUT / name).relative_to(ROOT))


def daily(c) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    regime = get(c, '/api/regime')
    scan = get(c, '/api/scan?tickers=' + quote(','.join(SCREENER_TICKERS)))
    if not scan.get('scanned'):
        raise SystemExit('scan returned no rows; keeping the old snapshot')
    write('screener.json', {'generated_at': stamp, 'tickers': SCREENER_TICKERS,
                            'regime': regime, 'scan': scan})
    # The two heavy research results. A failed one keeps yesterday's copy.
    for name, path in (('calibration.json', '/api/predict/calibration'),
                       ('pead.json', '/api/research/pead')):
        r = c.get(path)
        data = r.get_json(silent=True) or {}
        if r.status_code == 200 and data.get('available', True):
            data['snapshot_at'] = stamp
            write(name, data)
        else:
            print(f'{path} failed ({r.status_code}); kept the old copy')


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode not in ('fixed', 'daily', 'all'):
        raise SystemExit(__doc__)
    c = client()
    if mode in ('fixed', 'all'):
        for name, data in fixed_payloads(c).items():
            write(name, data)
    if mode in ('daily', 'all'):
        daily(c)


if __name__ == '__main__':
    main()
