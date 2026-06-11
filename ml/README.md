# ML Transformer — Training Pipeline

This directory trains TradeBot's multi-modal transformer and exports it for
the live bot. Training happens **offline** (Google Colab free GPU is plenty);
the server only ever runs the exported ONNX model through `onnxruntime`,
which is what keeps the deployed bot inside Render's free tier.

## What the model is

A deliberately small (~150k parameter) transformer encoder that reads 60-day
sequences of multi-modal market data and outputs three things per ticker:

| Head | Output | Why it matters |
|---|---|---|
| Direction | P(positive return over next 5 days) | The trade trigger |
| Return distribution | q10 / q25 / q50 / q75 / q90 quantiles | Forecasts a *distribution*, not a point — quantiles are architecturally prevented from crossing |
| Volatility | Annualised vol forecast | Risk context |

**Input modalities** (per day, 13 features): price/indicator block (log
returns, range, volume z-score, RSI, MACD, SMA ratio), macro block (VIX,
10y−2y Treasury spread, Fed Funds — FRED, free), news sentiment block
(GDELT tone — free). During training, whole modality blocks are randomly
zeroed ("modality dropout") so the live model keeps working when a source is
down — and so future paid modalities (options flow, social sentiment) can be
added as new blocks without redesigning anything.

## Anti-overfitting discipline (read this before changing anything)

1. **Small model.** ~150k params against ~150k training windows. Scaling the
   model up will *improve training loss and destroy test performance*.
2. **Cross-sectional training** over ~80 liquid tickers — the model learns
   market behaviour, not one stock's quirks. All features are price-scale
   invariant to make this valid.
3. **Date-based splits, never random**: train ≤ 2021, validate 2022 → mid-2023,
   test mid-2023 → today, with 10-day purge gaps so no 5-day label window
   straddles a boundary (López de Prado's leakage control).
4. **Thresholds calibrated on validation only.** Test metrics are reported
   once, at the end, untouched by any decision.
5. The deployed strategy still runs through the platform's existing
   walk-forward backtest, Monte Carlo, and Deflated Sharpe machinery —
   the model gets no special treatment.

## Training on Google Colab (free GPU)

1. Open [colab.research.google.com](https://colab.research.google.com) →
   New Notebook → Runtime → Change runtime type → **T4 GPU**.
2. Run, cell by cell:

```python
!git clone https://github.com/Danny-397/trading-bot-.git
%cd trading-bot-
!pip -q install -r ml/requirements-train.txt

import os
os.environ['TIINGO_API_KEY'] = 'your-key-here'   # same free key as the server

# ~10–20 min: downloads 12 years × ~80 tickers + macro + sentiment
!python ml/dataset.py

# ~15–40 min on a T4: trains, calibrates, exports ONNX, prints honest test metrics
!python ml/train.py
```

3. Download the two exported artefacts:

```python
from google.colab import files
files.download('backend/ml_model.onnx')
files.download('backend/ml_model_meta.json')
```

4. Drop both files into `backend/` in your local repo, commit, push.
   Render redeploys → the ML strategy lights up automatically (the frontend
   enables the "ML Transformer" option when `/api/ml/info` reports a loaded
   model).

## Honest expectations

Daily equity returns are ~99% noise. A *good* result here is direction AUC
of 0.52–0.56 out of sample — tiny but real edges are what actual quant funds
trade. If you see AUC > 0.60, **suspect leakage before celebrating.** Run
the deployed strategy through the Backtest page's Research tab and let the
Deflated Sharpe Ratio tell you whether the edge survives multiple-testing
correction. Reporting a negative result honestly is a feature of this
project, not a failure.

## Files

| File | Role |
|---|---|
| `dataset.py` | Downloads data, builds features/labels via `backend/ml_features.py` (the same code the server uses — zero train/serve skew), writes `ml/data/dataset.npz` |
| `model.py` | PyTorch architecture + multi-task loss |
| `train.py` | Training loop, early stopping, threshold calibration, ONNX export + parity check |
| `requirements-train.txt` | Training-only dependencies (torch never touches the server) |
