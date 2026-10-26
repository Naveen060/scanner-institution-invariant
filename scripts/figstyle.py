"""Figure style switch for the manuscripts.

  FIG_PROFILE=ieee python scripts/final_report.py     # figures for paper/ieee: written to paper/ieee/figures/
  FIG_PROFILE=ieee python scripts/round2_tables.py

IEEE profile (IEEE Transactions author instructions): Arial-compatible sans-serif font (Liberation Sans, metric-
compatible with Arial), 8 pt text at final size, no title inside the graphic (the caption carries it), subfigure
labels "(a)", "(b)" centred below each part, marker shapes (not only colour) to separate groups, vector PDF with
embedded TrueType fonts plus a 600 dpi PNG. Page width 7.16 in, column width 3.5 in: figure sizes below equal the
printed sizes, so text is not rescaled.
Without FIG_PROFILE the original sizes and fonts are kept (used by the single-column paper).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IEEE = os.environ.get("FIG_PROFILE", "") == "ieee"
FIGDIR = "paper/ieee/figures" if IEEE else "paper/figures"
DPI = 600 if IEEE else 200
PAGE_W, COL_W = 7.16, 3.5  # inches

plt.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts (IEEE PDF eXpress rejects Type 3)
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
if IEEE:
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "savefig.bbox": "standard",
    })
os.makedirs(FIGDIR, exist_ok=True)


def FS(n):
    """Font size: the original value, or 8 pt in the IEEE profile."""
    return 8 if IEEE else n


def size(orig, ieee):
    """Figure size (inches): the original, or the IEEE print size."""
    return ieee if IEEE else orig


def title(ax, text, **kw):
    """Panel title; omitted in the IEEE profile (the caption describes the figure)."""
    if not IEEE:
        ax.set_title(text, **kw)


def finish(fig, name, axes=None, labels=None):
    """Tight layout, optional (a)/(b) labels below each part (IEEE profile), save PDF + PNG, close."""
    if IEEE and axes is not None and labels:
        fig.tight_layout(rect=[0, 0.075, 1, 1])
        for ax, lab in zip(axes, labels):
            p = ax.get_position()
            fig.text((p.x0 + p.x1) / 2, 0.012, lab, ha="center", va="bottom", fontsize=8)
    else:
        fig.tight_layout(pad=0.4 if IEEE else 1.08)
    fig.savefig(f"{FIGDIR}/{name}.pdf")
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=DPI)
    plt.close(fig)
