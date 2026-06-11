"""
Multi-modal transformer for short-horizon return forecasting.

Design philosophy: financial daily data is SMALL data (a few hundred
thousand samples) drowning in noise, so the model is deliberately tiny
(~150k parameters — about 1/700,000th of GPT-3) with aggressive
regularisation.  A bigger model would just memorise the training set;
the skill here is right-sizing, not scaling.

Architecture
------------
    input [B, 60 days, 13 features]
      → linear projection to d_model + learned positional embedding
      → 3 pre-norm transformer encoder layers (4 heads, FF 128, dropout 0.25)
      → mean-pool over time
      → three task heads:
          direction  — logit for P(return > 0 over next 5 days)
          quantiles  — q10/q25/q50/q75/q90 of that return, built as a base
                       quantile plus cumulative softplus increments so the
                       quantiles can NEVER cross (a distribution that's
                       always valid)
          volatility — softplus-positive annualised vol forecast

Multi-task learning is itself a regulariser: one shared representation must
explain direction, distribution AND risk, which crowds out memorisation.

Modality dropout (applied by the training loop, not here): entire macro or
sentiment blocks are zeroed at random during training, forcing the price
block to carry the core signal and making the deployed model robust to a
missing data source.
"""

from __future__ import annotations

import torch
import torch.nn as nn

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


class MultiModalTransformer(nn.Module):
    def __init__(self, n_features: int = 13, seq_len: int = 60,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 d_ff: int = 128, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed  = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(d_model)
        self.drop    = nn.Dropout(dropout)

        self.head_dir  = nn.Linear(d_model, 1)
        self.head_qnt  = nn.Linear(d_model, len(QUANTILES))
        self.head_vol  = nn.Linear(d_model, 1)
        self.softplus  = nn.Softplus()

    def forward(self, x: torch.Tensor):
        """x: [B, seq_len, n_features] → (dir_logit [B,1], quantiles [B,5], vol [B,1])"""
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)
        h = self.norm(h.mean(dim=1))        # mean-pool over time
        h = self.drop(h)

        dir_logit = self.head_dir(h)

        # Monotonic quantiles: q10 is free, each later quantile adds a
        # softplus-positive increment — crossing quantiles are impossible.
        raw  = self.head_qnt(h)
        base = raw[:, :1]
        incs = self.softplus(raw[:, 1:])
        quantiles = torch.cat([base, base + torch.cumsum(incs, dim=1)], dim=1)

        vol = self.softplus(self.head_vol(h))
        return dir_logit, quantiles, vol

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Quantile-regression loss, averaged over the QUANTILES levels.
    pred: [B, 5], target: [B] (realised forward return).
    """
    target = target.unsqueeze(1)
    qs     = torch.tensor(QUANTILES, device=pred.device).unsqueeze(0)
    err    = target - pred
    return torch.maximum(qs * err, (qs - 1) * err).mean()


def multitask_loss(dir_logit, quantiles, vol, y_dir, y_ret, y_vol,
                   w_dir: float = 1.0, w_qnt: float = 1.0, w_vol: float = 0.5):
    """Weighted sum of the three heads' losses (returns total + parts)."""
    l_dir = nn.functional.binary_cross_entropy_with_logits(dir_logit.squeeze(1), y_dir)
    l_qnt = pinball_loss(quantiles, y_ret)
    l_vol = nn.functional.mse_loss(torch.log1p(vol.squeeze(1)), torch.log1p(y_vol))
    total = w_dir * l_dir + w_qnt * l_qnt + w_vol * l_vol
    return total, {'dir': l_dir.item(), 'qnt': l_qnt.item(), 'vol': l_vol.item()}
