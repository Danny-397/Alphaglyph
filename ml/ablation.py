"""
Which inputs does the transformer actually use? A modality ablation.

The model is built for three input blocks (price, macro, news), and the shipped
checkpoint had been trained with the macro and news blocks zero-filled. This
retrains the same architecture with the same splits and early stopping on
three input sets, each from several seeds:

    price               macro and news held at zero
    price + macro       news held at zero
    price + macro + news

plus a logistic regression on the last day's features, the baseline any deep
model has to beat. Every number reported is on the TEST split (mid-2023 on),
which nothing here is tuned on: early stopping and the choice of what to ship
use validation only.

Uncertainty comes from a moving-block bootstrap over test DATES (20-trading-
day blocks), not over samples: the 85 tickers on one day share a market, and
the 5-day labels overlap, so resampling rows would make every interval far too
narrow. Arms are compared on the same resampled dates (paired).

    python ml/dataset.py          # first, if ml/data/dataset.npz is missing
    python ml/ablation.py         # ~1-3 h on a laptop CPU; writes ml/ablation.json

Optional: ABLATION_SEEDS=1,2,3 (default), ABLATION_BOOT=1000.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import train as tr                       # noqa: E402
import ml_features as mlf                # noqa: E402  (on sys.path via train)

OUT = os.path.join(_HERE, 'ablation.json')
SEEDS = [int(s) for s in os.getenv('ABLATION_SEEDS', '1,2,3').split(',')]
N_BOOT = int(os.getenv('ABLATION_BOOT', '1000'))
BLOCK = 20

ARMS = {
    'price':              (mlf.MACRO_SLICE, mlf.SENTI_SLICE),
    'price + macro':      (mlf.SENTI_SLICE,),
    'price + macro + news': (),
}


def test_dates(idx, stats):
    data = np.load(tr.DATA_FILE, allow_pickle=False)
    return np.array([data[f'd_{t}'][end] for t, end in idx['test']])


def block_bootstrap(dates, n_boot, rng):
    """Row-index resamples built from contiguous blocks of unique test dates."""
    days = np.unique(dates)
    rows_of = {d: np.flatnonzero(dates == d) for d in days}
    n_blocks = int(np.ceil(len(days) / BLOCK))
    for _ in range(n_boot):
        starts = rng.integers(0, len(days) - BLOCK + 1, n_blocks)
        picked = np.concatenate([days[s:s + BLOCK] for s in starts])
        yield np.concatenate([rows_of[d] for d in picked])


def last_day(feats, idx, split):
    return np.stack([feats[t][end] for t, end in idx[split]])


def logistic_baseline(feats, labels, idx, seed=0):
    """L2 logistic regression on the 13 features of the window's last day."""
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(last_day(feats, idx, 'train'))
    ytr = torch.tensor([labels[t][end][1] for t, end in idx['train']], dtype=torch.float32)
    w = torch.nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.LBFGS(w.parameters(), max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            w(Xtr).squeeze(1), ytr) + 1e-3 * (w.weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return {split: torch.sigmoid(w(torch.from_numpy(last_day(feats, idx, split)))
                                     .squeeze(1)).numpy()
                for split in ('val', 'test')}


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    started = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feats, labels, idx, stats = tr.load_splits()
    loaders = tr.make_loaders(feats, labels, idx, num_workers=0)
    n_features = len(stats['features'])

    y_val = np.array([labels[t][end][1] for t, end in idx['val']])
    y_test = np.array([labels[t][end][1] for t, end in idx['test']])
    dates = test_dates(idx, stats)

    # Probabilities per arm: the mean over seeds (a seed ensemble), plus each
    # seed's own AUCs so seed-to-seed spread is visible.
    probs_test, report = {}, {}
    for arm, zero in ARMS.items():
        runs, seed_probs = [], []
        for seed in SEEDS:
            t0 = time.time()
            model = tr.fit(loaders, n_features, device, seed=seed, zero_blocks=zero,
                           verbose=False)
            val_m, _, _ = tr.evaluate(model, loaders['val'], device, zero)
            test_m, p_test, _ = tr.evaluate(model, loaders['test'], device, zero)
            seed_probs.append(p_test)
            runs.append({'seed': seed, 'val_auc': val_m['auc'], 'test_auc': test_m['auc']})
            print(f'{arm:<22} seed {seed}  val AUC {val_m["auc"]:.4f}  '
                  f'test AUC {test_m["auc"]:.4f}  ({time.time() - t0:.0f}s)', flush=True)
        probs_test[arm] = np.mean(seed_probs, axis=0)
        report[arm] = {
            'runs': runs,
            'mean_val_auc': round(float(np.mean([r['val_auc'] for r in runs])), 4),
            'mean_test_auc': round(float(np.mean([r['test_auc'] for r in runs])), 4),
            'ensemble_test_auc': round(tr.auc_score(y_test, probs_test[arm]), 4),
        }

    base = logistic_baseline(feats, labels, idx)
    probs_test['logistic regression'] = base['test']
    report['logistic regression'] = {
        'val_auc': round(tr.auc_score(y_val, base['val']), 4),
        'ensemble_test_auc': round(tr.auc_score(y_test, base['test']), 4),
    }
    print(f"{'logistic regression':<22} val AUC {report['logistic regression']['val_auc']:.4f}  "
          f"test AUC {report['logistic regression']['ensemble_test_auc']:.4f}", flush=True)

    # Paired moving-block bootstrap over test dates.
    rng = np.random.default_rng(0)
    names = list(probs_test)
    boot = {n: [] for n in names}
    for rows in block_bootstrap(dates, N_BOOT, rng):
        for n in names:
            boot[n].append(tr.auc_score(y_test[rows], probs_test[n][rows]))
    boot = {n: np.array(v) for n, v in boot.items()}
    for n in names:
        report[n]['test_auc_ci95'] = interval(boot[n])
        report[n]['p_auc_le_half'] = round(float(np.mean(boot[n] <= 0.5)), 4)

    full = 'price + macro + news'
    differences = {}
    for a, b in [('price + macro', 'price'), (full, 'price + macro'), (full, 'price'),
                 (full, 'logistic regression')]:
        d = boot[a] - boot[b]
        differences[f'{a} minus {b}'] = {
            'auc_difference': round(float(report[a]['ensemble_test_auc']
                                          - report[b]['ensemble_test_auc']), 4),
            'ci95': interval(d),
        }

    chosen = max(ARMS, key=lambda a: report[a]['mean_val_auc'])
    news_cols = [stats['features'].index(c) for c in mlf.SENTI_FEATURES]
    news_cover = float(np.mean([np.any(feats[t][end][news_cols] != 0)
                                for t, end in idx['test']]))
    out = {
        'generated': time.strftime('%Y-%m-%d'),
        'seeds': SEEDS,
        'bootstrap': {'resamples': N_BOOT, 'block_days': BLOCK, 'unit': 'test dates'},
        'splits': {'train_end': stats['train_end'], 'val_end': stats['val_end'],
                   'purge_days': stats['purge_days'],
                   'windows': {k: len(v) for k, v in idx.items()}},
        'test_base_rate_up': round(float(y_test.mean()), 4),
        'test_news_coverage': round(news_cover, 4),
        'arms': report,
        'paired_differences': differences,
        'chosen_by_validation': chosen,
        'minutes': round((time.time() - started) / 60, 1),
    }
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ('chosen_by_validation', 'paired_differences')}, indent=2))
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
