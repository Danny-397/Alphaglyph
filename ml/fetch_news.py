"""
Download GDELT news tone for every training ticker, patiently, to disk.

GDELT's free API allows one request every 5 seconds per IP and answers bursts,
or anything it judges a large query, with 429s that can last minutes. Fetching
inside the dataset build lost the whole news block to that. This fetches one
ticker at a time, waits 10 s between requests, backs off for minutes on a 429,
and saves each ticker as ml/data/news/<TICKER>.csv, so it can be stopped and
re-run and only fetches what is missing. ml/dataset.py then reads the files
(ML_NEWS_DIR) instead of calling GDELT.

    python ml/fetch_news.py        # ~15-60 min depending on GDELT's mood
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'backend'))
sys.path.insert(0, _HERE)

import ml_features as mlf                 # noqa: E402
from dataset import TICKERS             # noqa: E402

NEWS_DIR = os.path.join(_HERE, 'data', 'news')
START = '20170101000000'                 # GDELT DOC 2.0 coverage begins in 2017
SPACING = 10
BACKOFF = [60, 120, 240, 480]


def fetch(query: str) -> pd.DataFrame | None:
    params = {'query': query, 'mode': 'timelinetone', 'format': 'json',
              'startdatetime': START,
              'enddatetime': pd.Timestamp.now().strftime('%Y%m%d') + '235959'}
    for wait in [0] + BACKOFF:
        if wait:
            print(f'    rate-limited; waiting {wait}s', flush=True)
            time.sleep(wait)
        resp = requests.get(mlf._GDELT_URL, params=params, timeout=60,
                            headers={'User-Agent': 'AlphaGlyph research (educational)'})
        if resp.status_code == 429 or 'limit requests' in resp.text[:200]:
            continue
        resp.raise_for_status()
        timeline = resp.json().get('timeline') or []
        points = timeline[0].get('data') if timeline else None
        if not points:
            return pd.DataFrame(columns=['f_news_tone'])
        idx = pd.to_datetime([p['date'] for p in points]).tz_localize(None)
        return pd.DataFrame({'f_news_tone': [float(p['value']) for p in points]}, index=idx)
    return None


def main():
    os.makedirs(NEWS_DIR, exist_ok=True)
    for i, ticker in enumerate(TICKERS, 1):
        path = os.path.join(NEWS_DIR, f'{ticker}.csv')
        if os.path.exists(path):
            continue
        query = mlf.news_query(ticker)
        if query is None:
            print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} no unambiguous query, skipped')
            continue
        try:
            df = fetch(query)
        except Exception as exc:            # one bad ticker never stops the run
            print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} FAILED ({exc})', flush=True)
            df = None
        if df is None:
            print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} gave up after backoff; re-run later',
                  flush=True)
        else:
            df.to_csv(path, index_label='date')
            print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} {len(df):>5} days  ({query})', flush=True)
        time.sleep(SPACING)
    have = [t for t in TICKERS if os.path.exists(os.path.join(NEWS_DIR, f'{t}.csv'))]
    print(f'\n{len(have)}/{len(TICKERS)} tickers have news on disk in {NEWS_DIR}')


if __name__ == '__main__':
    main()
