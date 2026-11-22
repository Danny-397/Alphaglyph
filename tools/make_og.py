"""Draw frontend/og-image.png, the 1200x630 link preview, in the site's style.

Evergreen-black background, a Geist headline and AlphaGlyph green, with the
validation numbers the home page shows. Run from the repo root:

    python tools/make_og.py

Needs matplotlib, plus fontTools with brotli to read the self-hosted woff2;
without those it falls back to a system sans.
"""
from __future__ import annotations

import io
import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

BG = "#0e1411"
TEXT = "#e6ece8"
MUTED = "#a3b0a8"
FAINT = "#7f8d85"
LINE = "#22302a"
GREEN = "#30d158"
ORANGE = "#ff9f0a"

# The same three figures as the home page's validation report (index.html).
ROWS = [("72%", "Overfitting risk"), ("1.26", "In-sample Sharpe"), ("0.65", "Out-of-sample Sharpe")]


def _display() -> font_manager.FontProperties:
    woff2 = os.path.join("frontend", "fonts", "geist-latin.woff2")
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer

        # Pin the variable font to the site's headline weight; matplotlib
        # cannot pick an axis value itself.
        font = instancer.instantiateVariableFont(TTFont(woff2), {"wght": 600})
        font.flavor = None
        buf = io.BytesIO()
        font.save(buf)
        path = os.path.join(tempfile.gettempdir(), "alphaglyph-geist.ttf")
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())
        font_manager.fontManager.addfont(path)
        return font_manager.FontProperties(fname=path)
    except Exception:  # noqa: BLE001 - any failure means "use the fallback"
        return font_manager.FontProperties(family=["Segoe UI", "DejaVu Sans"], weight="semibold")


def main() -> None:
    disp = _display()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (n for n in ("Inter", "Segoe UI", "Helvetica Neue", "Arial") if n in installed), "DejaVu Sans")
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)

    fig.text(0.066, 0.87, "AlphaGlyph", color=MUTED, fontsize=22, fontproperties=disp)
    fig.text(0.066, 0.70, "Backtest any strategy.", color=TEXT, fontsize=50, fontproperties=disp)
    fig.text(0.066, 0.575, "Then find out if the edge is real.", color=TEXT, fontsize=50,
             fontproperties=disp)

    fig.text(0.066, 0.43, "●", color=ORANGE, fontsize=15, va="center")
    fig.text(0.087, 0.43, "Likely overfit", color=TEXT, fontsize=27, va="center",
             fontproperties=disp)
    fig.text(0.315, 0.43, "AlphaGlyph's own 42-variant strategy grid, validated",
             color=FAINT, fontsize=14, va="center")

    xs = [0.066, 0.36, 0.654]
    for x, (val, lbl) in zip(xs, ROWS):
        fig.add_artist(plt.Line2D([x, x + 0.26], [0.35, 0.35], color=LINE, lw=1.2,
                                  transform=fig.transFigure))
        fig.text(x, 0.215, val, color=TEXT, fontsize=48, fontproperties=disp)
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
