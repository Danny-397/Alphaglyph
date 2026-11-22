"""Draw frontend/og-image.png, the 1200x630 link preview, in the site's style.

Midnight background, the Newsreader headline and AlphaGlyph green, with the
validation numbers the home page shows. Run from the repo root:

    python tools/make_og.py

Needs matplotlib, plus fontTools with brotli to read the self-hosted woff2;
without those it falls back to Georgia.
"""
from __future__ import annotations

import io
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BG = "#121821"
TEXT = "#e4e8ec"
MUTED = "#a5afba"
FAINT = "#838e9a"
LINE = "#2a3440"
GREEN = "#30d158"
ORANGE = "#ff9f0a"

# The same three figures as the home page's validation report (index.html).
ROWS = [("72%", "Overfitting risk"), ("1.26", "In-sample Sharpe"), ("0.65", "Out-of-sample Sharpe")]


def _serif() -> font_manager.FontProperties:
    woff2 = os.path.join("frontend", "fonts", "newsreader-latin-normal.woff2")
    try:
        from fontTools.ttLib import TTFont

        font = TTFont(woff2)
        font.flavor = None
        buf = io.BytesIO()
        font.save(buf)
        path = os.path.join(tempfile.gettempdir(), "alphaglyph-newsreader.ttf")
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path)
    except Exception:  # noqa: BLE001 - any failure means "use the fallback"
        return font_manager.FontProperties(family=["Georgia", "DejaVu Serif"])


def main() -> None:
    serif = _serif()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (n for n in ("Inter", "Segoe UI", "Helvetica Neue", "Arial") if n in installed), "DejaVu Sans")
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)

    fig.text(0.066, 0.87, "AlphaGlyph", color=MUTED, fontsize=22, fontproperties=serif)
    fig.text(0.066, 0.70, "Backtest any strategy.", color=TEXT, fontsize=58, fontproperties=serif)
    fig.text(0.066, 0.575, "Then find out if the edge is real.", color=TEXT, fontsize=58,
             fontproperties=serif)

    fig.text(0.066, 0.43, "●", color=ORANGE, fontsize=15, va="center")
    fig.text(0.087, 0.43, "Likely overfit", color=TEXT, fontsize=27, va="center",
             fontproperties=serif, style="italic")
    fig.text(0.29, 0.43, "AlphaGlyph's own 42-variant strategy grid, validated",
             color=FAINT, fontsize=14, va="center")

    xs = [0.066, 0.36, 0.654]
    for x, (val, lbl) in zip(xs, ROWS):
        fig.add_artist(plt.Line2D([x, x + 0.26], [0.35, 0.35], color=LINE, lw=1.2,
                                  transform=fig.transFigure))
        fig.text(x, 0.215, val, color=TEXT, fontsize=48, fontproperties=serif)
        fig.text(x, 0.155, lbl, color=MUTED, fontsize=15)

    fig.text(0.066, 0.06, "Monte Carlo, deflated Sharpe and Fama-French on every backtest.",
             color=FAINT, fontsize=14)
    fig.text(0.934, 0.06, "alphaglyph.org", color=GREEN, fontsize=15, ha="right")

    out = os.path.join("frontend", "og-image.png")
    fig.savefig(out, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
