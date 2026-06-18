"""
Training dataset builder for the ML transformer.

Run from the repo root (Colab or locally):

    TIINGO_API_KEY=...  python ml/dataset.py

Downloads ~12 years of daily data for ~80 liquid large-caps, builds the
multi-modal feature frames with the SAME code the live server uses
(backend/ml_features.py — zero train/serve skew), attaches forward-looking
labels, and writes everything to ml/data/dataset.npz.

Anti-overfitting decisions made HERE, before the model ever sees data:

  Cross-sectional training   One model trained across ~80 tickers, not 7 —
                             multiplies the sample count ~12x and forces the
                             model to learn market behaviour, not ticker quirks.
                             (Features are price-scale invariant to allow this.)

  Date-based 60/20/20        Train = oldest 60% of trading days, validation =
                             next 20%, test = most recent 20% (cutoffs computed
                             from the data's own date range). Splitting by date
                             (never randomly!) means validation and test are
                             always strictly in the model's future.

  Purge gaps                 10 trading days are dropped at each boundary so
                             a label window can never straddle two splits —
                             the standard leakage control from López de Prado's
                             "Advances in Financial Machine Learning".

  Train-only normalisation   Feature means/stds are computed on the training
                             rows only and saved for the server to reuse.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Reuse the backend's feature pipeline — the whole point is one implementation.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'backend'))

import features as feat              # noqa: E402
import ml_features as mlf            # noqa: E402

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUT_FILE = os.path.join(OUT_DIR, 'dataset.npz')

START_DATE = '2012-01-01'

# Chronological 60 / 20 / 20 split (train / validation / test), computed in
# build() from the data's OWN date range: train is the oldest 60% of trading
# days, validation the next 20%, test the most recent 20%. Splitting by DATE
# (never randomly) keeps validation and test strictly in the model's future.
# PURGE_DAYS trading days are dropped after each boundary so no label horizon
# crosses a split (leakage control, López de Prado).
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20      # test is the remaining 0.20
PURGE_DAYS = 10

# ~80 liquid large-caps across sectors. Liquid names have the cleanest data
# and the least bid-ask noise; sector spread stops the model learning "tech
# go up" as its only idea.
TICKERS = [
    # Tech / communication
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'ORCL', 'CRM',
    'ADBE', 'AMD', 'INTC', 'QCOM', 'TXN', 'CSCO', 'IBM', 'NFLX', 'DIS', 'TMUS', 'VZ',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA', 'COF',
    # Healthcare
    'UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'BMY', 'AMGN', 'GILD', 'CVS',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT', 'HD', 'LOW', 'TSLA',
    # Industrials / energy / materials
    'CAT', 'DE', 'BA', 'GE', 'HON', 'UPS', 'UNP', 'LMT', 'RTX', 'MMM',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'LIN', 'FCX', 'NEM',
    # Utilities / REITs / staples breadth
    'NEE', 'DUK', 'SO', 'AMT', 'PLD', 'SPG', 'CL', 'KMB', 'GIS', 'MO',
    # Broad market (the regime detector's own proxy)
    'SPY',
]


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    arrays: dict[str, np.ndarray] = {}
    kept, skipped = [], []
    t0 = time.time()

    # Warm the macro cache once — every ticker then slices it for free.
    # This single chunked FRED fetch takes ~90s; if it fails, macro is
    # zero-filled and the build still completes (modality dropout handles it).
    print('Fetching macro data from FRED (one-time, ~90s)…', flush=True)
    macro = mlf.fetch_macro(START_DATE, end_date)
    print('  macro:', 'OK' if macro is not None else 'unavailable — zero-filled',
          flush=True)
    if not os.getenv('ML_SKIP_SENTIMENT'):
        print('News sentiment (GDELT) enabled — it auto-disables if rate-limited. '
              'Set ML_SKIP_SENTIMENT=1 to skip it for a faster run.', flush=True)

    for i, ticker in enumerate(TICKERS, 1):
        try:
            ohlcv = feat.fetch_ohlcv(ticker, start=START_DATE, end=end_date)
            if ohlcv is None or len(ohlcv) < 500:
                skipped.append(ticker)
                print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} SKIPPED (insufficient data)',
                      flush=True)
                continue

            frame  = mlf.build_feature_frame(ohlcv, ticker)
            labels = mlf.build_labels(ohlcv)
        except Exception as exc:                      # never let one ticker kill the build
            skipped.append(ticker)
            print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} SKIPPED ({type(exc).__name__})',
                  flush=True)
            continue
        if frame is None:
            skipped.append(ticker)
            print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} SKIPPED (feature build failed)',
                  flush=True)
            continue

        labels = labels.reindex(frame.index)
        valid  = labels.notna().all(axis=1)
        # Keep feature rows even where labels are NaN (window context),
        # but remember which prediction dates are usable.
        arrays[f'f_{ticker}'] = frame.values.astype(np.float32)
        arrays[f'l_{ticker}'] = labels.values.astype(np.float32)
        arrays[f'v_{ticker}'] = valid.values
        arrays[f'd_{ticker}'] = np.array([d.strftime('%Y-%m-%d') for d in frame.index])
        kept.append(ticker)
        print(f'[{i:>2}/{len(TICKERS)}] {ticker:<6} {len(frame):>5} days  '
              f'({time.time() - t0:5.0f}s elapsed)', flush=True)

    if not kept:
        raise SystemExit('No tickers succeeded — check TIINGO_API_KEY and connectivity.')

    # ── Chronological 60/20/20 cutoffs from the data's own date range ──────
    # Union of every prediction date, then the 60th and 80th percentiles. This
    # makes the split exactly 60% train / 20% validation / 20% test by trading
    # day, adapting automatically as the dataset grows.
    all_dates = sorted({d for t in kept for d in arrays[f'd_{t}'].tolist()})
    n_dates   = len(all_dates)
    TRAIN_END = all_dates[int(n_dates * TRAIN_FRAC)]
    VAL_END   = all_dates[int(n_dates * (TRAIN_FRAC + VAL_FRAC))]
    print(f'\nsplit (60/20/20 by date, {n_dates} trading days): '
          f'train ≤ {TRAIN_END} | val ≤ {VAL_END} | test ≤ {all_dates[-1]}', flush=True)

    # ── Train-split normalisation stats (train rows only, all tickers) ────
    train_rows = []
    for t in kept:
        dates = arrays[f'd_{t}']
        mask  = dates <= TRAIN_END
        train_rows.append(arrays[f'f_{t}'][mask])
    stacked = np.concatenate(train_rows, axis=0)
    means   = stacked.mean(axis=0).tolist()
    stds    = stacked.std(axis=0).tolist()

    arrays['tickers'] = np.array(kept)
    np.savez_compressed(OUT_FILE, **arrays)

    stats = {
        'features':   mlf.ALL_FEATURES,
        'seq_len':    mlf.SEQ_LEN,
        'horizon':    mlf.HORIZON,
        'means':      means,
        'stds':       stds,
        'train_end':  TRAIN_END,
        'val_end':    VAL_END,
        'purge_days': PURGE_DAYS,
        'start_date': START_DATE,
        'built_at':   pd.Timestamp.now().isoformat(),
        'n_tickers':  len(kept),
        'skipped':    skipped,
    }
    with open(os.path.join(OUT_DIR, 'stats.json'), 'w') as fh:
        json.dump(stats, fh, indent=2)

    n_days = sum(len(arrays[f'd_{t}']) for t in kept)
    print(f'\nDone: {len(kept)} tickers, {n_days:,} ticker-days '
          f'→ {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1e6:.1f} MB)')
    if skipped:
        print(f'Skipped: {", ".join(skipped)}')


if __name__ == '__main__':
    build()
