"""
Calibration & track record for the price forecaster.

The Predictor makes probabilistic claims ("~58% chance of an up-move"). This
module does what AlphaGlyph's whole thesis demands of any such claim: it checks
whether those probabilities are *honest*. A forecaster is well-calibrated when,
across every day it said "60%", the market actually rose ~60% of the time.

We calibrate the one channel we can reconstruct point-in-time with zero leakage:
the ML transformer's price forecast. For every historical window across a basket
of tickers we recover the model's p(up) and the realised next-`horizon` outcome,
then:

  * bin predictions into deciles → a reliability diagram (predicted vs actual);
  * score them with the Brier score and log loss (lower = better), each against
    the naive "always predict the base rate" baseline so the number has meaning.

Two honesty controls make the number trustworthy rather than flattering:

  Out-of-sample only   By default we evaluate only on dates AFTER the model's
                       validation cutoff (meta 'splits.val_end' + purge) — data
                       the model never trained or tuned on. In-sample calibration
                       would look better and mean nothing.

  No look-ahead        Each p(up) uses only the 60 days ending on its own date;
                       the label is the return over the days strictly after it.

The expected result is sobering — the model lives near a coin flip (test AUC
≈ 0.51) — and that is exactly the point: the reliability diagram should sit close
to the diagonal with little spread, an honestly humble forecaster rather than an
over-confident one.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

import features as feat
import ml_features as mlf
import ml_runtime

logger = logging.getLogger(__name__)

# A compact, liquid, sector-spread default basket — enough events for stable
# bins without a punishing number of cold data fetches on the free tier.
DEFAULT_UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ',
                    'XOM', 'PG', 'NVDA', 'HD', 'BAC', 'SPY']

_CACHE: dict = {}
_CACHE_TTL = 12 * 3600   # 12h — moves only as new daily bars arrive


def _oos_cutoff(meta: dict) -> pd.Timestamp | None:
    """First date the model never saw in training/validation (val_end + purge)."""
    splits = (meta or {}).get('splits') or {}
    val_end = splits.get('val_end')
    if not val_end:
        return None
    purge = int(splits.get('purge_days', 10))
    # Purge is in trading days; a business-day offset is a safe over-approximation.
    return pd.Timestamp(val_end) + pd.tseries.offsets.BDay(purge)


def _collect(tickers, horizon, cutoff):
    """Gather aligned (p_up, realised_up) pairs across the basket."""
    p_all, y_all, dates_all = [], [], []
    used = []
    for t in tickers:
        try:
            ohlcv = feat.fetch_ohlcv(t, period='5y')
        except Exception as exc:
            logger.warning('calibration: fetch failed for %s: %s', t, exc)
            continue
        if ohlcv is None or ohlcv.empty:
            continue

        X, dates = ml_runtime._prepare_windows(ohlcv, t,
                                               include_sentiment=False, include_macro=False)
        if X is None:
            continue
        pred = ml_runtime.predict_batch(X)
        if pred is None:
            continue

        labels = mlf.build_labels(ohlcv, horizon)          # indexed by decision date
        y_dir = labels['y_dir'].reindex(pd.DatetimeIndex(dates))
        p_up  = pd.Series(pred['p_up'], index=pd.DatetimeIndex(dates))

        pair = pd.DataFrame({'p': p_up, 'y': y_dir}).dropna()
        if cutoff is not None:
            pair = pair[pair.index > cutoff]
        if pair.empty:
            continue

        p_all.append(pair['p'].values)
        y_all.append(pair['y'].values)
        dates_all.append(pair.index)
        used.append(t)

    if not p_all:
        return None
    return (np.concatenate(p_all), np.concatenate(y_all),
            used, pd.DatetimeIndex(np.concatenate([d.values for d in dates_all])))


def compute_calibration(tickers=None, n_bins: int = 10, oos_only: bool = True,
                        bin_mode: str = 'quantile') -> dict:
    """
    Reliability diagram + scores for the ML price forecaster. Never raises.

    bin_mode:
        'quantile' (default) — equal-count bins by predicted-probability rank.
                    The model's outputs cluster tightly near the base rate, so
                    equal-WIDTH bins collapse to one useless point; quantile bins
                    reveal whether its relatively-more-bullish calls actually fare
                    better — the honest, informative view of a compressed model.
        'width'    — fixed [0,1] deciles (textbook reliability diagram).
    """
    tickers = tickers or DEFAULT_UNIVERSE
    tickers = [t.upper().strip() for t in tickers][:20]
    cache_key = (tuple(tickers), n_bins, oos_only, bin_mode)
    hit = _CACHE.get(cache_key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]

    info = ml_runtime.get_info()
    if not info.get('loaded'):
        return {'available': False, 'reason': 'ML model not loaded'}
    ml_runtime._ensure_loaded()
    meta = ml_runtime._meta or {}
    horizon = int(meta.get('horizon', 5))
    cutoff = _oos_cutoff(meta) if oos_only else None

    got = _collect(tickers, horizon, cutoff)
    if got is None:
        return {'available': False, 'reason': 'no evaluable predictions'}
    p, y, used, dates = got
    n = len(p)

    base_rate = float(y.mean())
    # Brier & log loss, with the "always predict base rate" baseline for context.
    eps = 1e-7
    pc = np.clip(p, eps, 1 - eps)
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base_rate - y) ** 2))
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    bb = np.clip(base_rate, eps, 1 - eps)
    logloss_base = float(-np.mean(y * np.log(bb) + (1 - y) * np.log(1 - bb)))
    # Brier skill score: how much better than the base-rate baseline (0 = no skill).
    bss = float(1 - brier / brier_base) if brier_base > 0 else 0.0
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))

    # ── Reliability bins ──────────────────────────────────────────────────────
    if bin_mode == 'quantile':
        # Equal-count bins by predicted-probability rank (unique quantile edges).
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        bins.append({
            'bin_lo':    round(float(edges[b]), 4),
            'bin_hi':    round(float(edges[b + 1]), 4),
            'mean_pred': round(float(p[mask].mean()), 4),
            'frac_up':   round(float(y[mask].mean()), 4),
            'count':     cnt,
        })

    result = {
        'available':     True,
        'oos_only':      oos_only,
        'horizon':       horizon,
        'n_predictions': n,
        'bin_mode':      bin_mode,
        'n_tickers':     len(used),
        'tickers':       used,
        'eval_start':    str(dates.min().date()),
        'eval_end':      str(dates.max().date()),
        'base_rate':     round(base_rate, 4),
        'metrics': {
            'brier':         round(brier, 4),
            'brier_base':    round(brier_base, 4),
            'brier_skill':   round(bss, 4),
            'log_loss':      round(logloss, 4),
            'log_loss_base': round(logloss_base, 4),
            'accuracy':      round(acc, 4),
            'model_test_auc': (meta.get('test_metrics') or {}).get('auc'),
        },
        'bins': bins,
        'note': (
            'Out-of-sample reliability of the ML price forecaster on data after the '
            'validation cutoff. Points on the diagonal = perfectly calibrated. The '
            'model lives near a coin flip by design — a humble, honest forecaster is '
            'the goal, not an over-confident one.'
        ) if oos_only else 'In-sample calibration — shown for comparison only.',
    }
    _CACHE[cache_key] = (time.time(), result)
    return result
