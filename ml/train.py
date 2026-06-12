"""
Train the multi-modal transformer and export it for the live bot.

Run AFTER ml/dataset.py, ideally on a GPU (Google Colab free T4 is plenty —
the model is ~150k params and trains in well under an hour):

    python ml/train.py

What this script does, in order:
  1. Loads ml/data/dataset.npz, normalises with TRAIN-split stats only
  2. Trains with early stopping on the validation split (2022 → mid-2023),
     applying modality dropout so the model survives missing macro/sentiment
  3. Calibrates BUY/SELL probability thresholds on the VALIDATION split
     (never on test — that would be selection bias, the exact sin the
     Deflated Sharpe Ratio exists to catch)
  4. Reports honest TEST-split metrics (mid-2023 → today): these numbers were
     never used for any training or selection decision
  5. Exports backend/ml_model.onnx + backend/ml_model_meta.json and verifies
     ONNX output matches PyTorch to 1e-4

Commit the two exported backend/ files and push — Render redeploys and the
ML strategy goes live automatically.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, 'backend'))

import ml_features as mlf                                   # noqa: E402
from model import MultiModalTransformer, multitask_loss     # noqa: E402

DATA_FILE  = os.path.join(_HERE, 'data', 'dataset.npz')
STATS_FILE = os.path.join(_HERE, 'data', 'stats.json')
ONNX_OUT   = os.path.join(_REPO, 'backend', 'ml_model.onnx')
META_OUT   = os.path.join(_REPO, 'backend', 'ml_model_meta.json')

BATCH_SIZE     = 256
MAX_EPOCHS     = 60
PATIENCE       = 8       # early-stopping patience (epochs without val improvement)
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
MODALITY_DROP  = 0.15    # P(zero the macro block) and P(zero the sentiment block)
CLIP_SIGMA     = 5.0
SEED           = 42


# ── Data ───────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Lazily slices 60-day windows out of per-ticker day matrices, so the
    full window tensor (which would be ~650 MB) never materialises.
    """
    def __init__(self, feats: dict, labels: dict, index: list):
        self.feats, self.labels, self.index = feats, labels, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        t, end = self.index[i]
        X = self.feats[t][end - mlf.SEQ_LEN + 1: end + 1]
        y = self.labels[t][end]
        return torch.from_numpy(X), torch.tensor(y[1]), torch.tensor(y[0]), torch.tensor(y[2])


def load_splits():
    data  = np.load(DATA_FILE, allow_pickle=False)
    stats = json.load(open(STATS_FILE))
    mu    = np.asarray(stats['means'], dtype=np.float32)
    sd    = np.asarray(stats['stds'],  dtype=np.float32)
    sd    = np.where(sd <= 0, 1.0, sd)

    feats, labels = {}, {}
    idx = {'train': [], 'val': [], 'test': []}
    purge = int(stats['purge_days'])

    for t in data['tickers']:
        f = np.clip((data[f'f_{t}'] - mu) / sd, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)
        feats[t], labels[t] = f, data[f'l_{t}']
        valid, dates = data[f'v_{t}'], data[f'd_{t}']

        i_train = int(np.searchsorted(dates, stats['train_end'], side='right'))
        i_val   = int(np.searchsorted(dates, stats['val_end'],   side='right'))

        for end in range(mlf.SEQ_LEN - 1, len(dates)):
            if not valid[end]:
                continue
            if end < i_train:
                idx['train'].append((t, end))
            elif i_train + purge <= end < i_val:
                idx['val'].append((t, end))
            elif end >= i_val + purge:
                idx['test'].append((t, end))

    print(f"windows — train {len(idx['train']):,} | val {len(idx['val']):,} "
          f"| test {len(idx['test']):,}")
    return feats, labels, idx, stats


# ── Metrics ────────────────────────────────────────────────────────────────────

def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Rank-based AUC (no sklearn dependency)."""
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_prob) + 1)
    pos = y_true > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, rets, dirs, losses = [], [], [], []
    for X, y_dir, y_ret, y_vol in loader:
        X, y_dir, y_ret, y_vol = (v.to(device) for v in (X, y_dir, y_ret, y_vol))
        d, q, v = model(X)
        loss, _ = multitask_loss(d, q, v, y_dir, y_ret, y_vol)
        losses.append(loss.item())
        probs.append(torch.sigmoid(d.squeeze(1)).cpu().numpy())
        rets.append(y_ret.cpu().numpy())
        dirs.append(y_dir.cpu().numpy())
    probs, rets, dirs = map(np.concatenate, (probs, rets, dirs))
    return {
        'loss': float(np.mean(losses)),
        'auc':  round(auc_score(dirs, probs), 4),
        'acc':  round(float(((probs > 0.5) == (dirs > 0.5)).mean()), 4),
    }, probs, rets


def calibrate_thresholds(probs: np.ndarray, rets: np.ndarray) -> dict:
    """
    Sweep BUY/SELL probability cutoffs on the VALIDATION split and keep the
    ones with the best t-statistic of realised forward returns, requiring
    ≥1% coverage so we never select a handful of lucky samples.
    """
    def best(side: str):
        best_th, best_t = (0.55 if side == 'buy' else 0.45), 0.0
        for th in np.arange(0.52, 0.71, 0.01):
            mask = probs >= th if side == 'buy' else probs <= 1 - th
            if mask.mean() < 0.01:
                continue
            r = rets[mask] if side == 'buy' else -rets[mask]
            t = r.mean() / (r.std() / np.sqrt(len(r)) + 1e-12)
            if t > best_t:
                best_t, best_th = t, (th if side == 'buy' else round(1 - th, 2))
        return round(float(best_th), 2)

    return {'buy_prob': best('buy'), 'sell_prob': best('sell')}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    feats, labels, idx, stats = load_splits()
    n_features = len(stats['features'])

    loaders = {
        k: DataLoader(WindowDataset(feats, labels, v), batch_size=BATCH_SIZE,
                      shuffle=(k == 'train'), num_workers=2, drop_last=(k == 'train'))
        for k, v in idx.items()
    }

    model = MultiModalTransformer(n_features=n_features, seq_len=mlf.SEQ_LEN).to(device)
    print(f'parameters: {model.count_params():,}')
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3)

    macro_sl, senti_sl = mlf.MACRO_SLICE, mlf.SENTI_SLICE
    best_val, best_state, stale = float('inf'), None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for X, y_dir, y_ret, y_vol in loaders['train']:
            X, y_dir, y_ret, y_vol = (v.to(device) for v in (X, y_dir, y_ret, y_vol))

            # Modality dropout: per-sample, zero an entire block so the model
            # learns to function when a live data source is down.
            for sl in (macro_sl, senti_sl):
                drop = torch.rand(X.shape[0], device=device) < MODALITY_DROP
                X[drop, :, sl] = 0.0

            d, q, v = model(X)
            loss, _ = multitask_loss(d, q, v, y_dir, y_ret, y_vol)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        val, _, _ = evaluate(model, loaders['val'], device)
        sched.step(val['loss'])
        marker = ''
        if val['loss'] < best_val - 1e-4:
            best_val, stale = val['loss'], 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = '  ← best'
        else:
            stale += 1
        print(f"epoch {epoch:>2}  val loss {val['loss']:.4f}  AUC {val['auc']:.3f}  "
              f"acc {val['acc']:.3f}{marker}")
        if stale >= PATIENCE:
            print(f'early stop (no improvement for {PATIENCE} epochs)')
            break

    model.load_state_dict(best_state)

    # Persist the trained weights immediately, BEFORE the (occasionally fragile)
    # ONNX export — so a missing export dependency never wastes a training run.
    torch.save(best_state, os.path.join(_HERE, 'data', 'ml_model.pt'))

    # ── Threshold calibration on VALIDATION, honest report on TEST ────────
    val_m,  val_probs,  val_rets  = evaluate(model, loaders['val'],  device)
    thresholds = calibrate_thresholds(val_probs, val_rets)
    test_m, test_probs, test_rets = evaluate(model, loaders['test'], device)

    buy_mask = test_probs >= thresholds['buy_prob']
    edge = float(test_rets[buy_mask].mean()) if buy_mask.any() else 0.0
    print(f"\nVAL  — AUC {val_m['auc']}  acc {val_m['acc']}  thresholds {thresholds}")
    print(f"TEST — AUC {test_m['auc']}  acc {test_m['acc']}  "
          f"buy-signal coverage {buy_mask.mean():.1%}  "
          f"avg 5-day fwd log-ret on signals {edge:+.4f}")

    # ── ONNX export + parity check ─────────────────────────────────────────
    # dynamo=False forces the legacy TorchScript exporter. PyTorch's newer
    # dynamo exporter mishandles the TransformerEncoder's attention reshape with
    # a dynamic batch axis (produces a broken Reshape), whereas the legacy path
    # exports this model correctly and is exactly what dynamic_axes targets.
    model.eval().cpu()
    dummy = torch.randn(2, mlf.SEQ_LEN, n_features)
    torch.onnx.export(
        model, dummy, ONNX_OUT, opset_version=17,
        input_names=['input'], output_names=['direction', 'quantiles', 'volatility'],
        dynamic_axes={'input': {0: 'batch'}, 'direction': {0: 'batch'},
                      'quantiles': {0: 'batch'}, 'volatility': {0: 'batch'}},
        dynamo=False,
    )
    import onnxruntime as ort
    sess  = ort.InferenceSession(ONNX_OUT, providers=['CPUExecutionProvider'])
    check = torch.randn(4, mlf.SEQ_LEN, n_features)
    with torch.no_grad():
        td, tq, tv = model(check)
    od, oq, ov = sess.run(None, {'input': check.numpy()})
    for a, b in ((td, od), (tq, oq), (tv, ov)):
        assert np.allclose(a.numpy(), b, atol=1e-4), 'ONNX/PyTorch outputs diverge'
    print(f'ONNX parity check passed → {ONNX_OUT} '
          f'({os.path.getsize(ONNX_OUT) / 1e6:.2f} MB)')

    meta = {
        'version':       datetime.now().strftime('%Y%m%d-%H%M'),
        'created':       datetime.now().isoformat(),
        'architecture':  'multi-modal transformer encoder '
                         '(3 layers, d_model 64, 4 heads, multi-task heads)',
        'n_params':      model.count_params(),
        'seq_len':       mlf.SEQ_LEN,
        'horizon':       mlf.HORIZON,
        'features':      stats['features'],
        'feature_means': stats['means'],
        'feature_stds':  stats['stds'],
        'clip':          CLIP_SIGMA,
        'thresholds':    thresholds,
        'splits':        {'train_end': stats['train_end'], 'val_end': stats['val_end'],
                          'purge_days': stats['purge_days']},
        'val_metrics':   val_m,
        'test_metrics':  {**test_m, 'buy_coverage': round(float(buy_mask.mean()), 4),
                          'buy_avg_fwd_logret': round(edge, 5)},
        'n_tickers':     stats['n_tickers'],
        'modality_dropout': MODALITY_DROP,
    }
    with open(META_OUT, 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'meta → {META_OUT}\n\nCommit backend/ml_model.onnx + '
          f'backend/ml_model_meta.json and push to deploy.')


if __name__ == '__main__':
    main()
