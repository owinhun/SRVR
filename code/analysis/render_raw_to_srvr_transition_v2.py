#!/usr/bin/env python3
"""Render the frozen video-SALMONN-2 Raw-to-SRVR transition counts."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures/raw_to_srvr_transitions_v2.pdf"
PNG = ROOT / "paper/figures/raw_to_srvr_transitions_v2.png"


def box(ax, xy, text, face, edge):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), 0.28, 0.19,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face, edgecolor=edge, linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x + 0.14, y + 0.095, text, ha="center", va="center", fontsize=8.2, weight="bold")


def arrow(ax, start, end, color, width, bend, label, label_xy):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        connectionstyle=f"arc3,rad={bend}", linewidth=width,
        color=color, alpha=0.88, shrinkA=3, shrinkB=3,
    )
    ax.add_patch(patch)
    ax.text(*label_xy, label, color=color, fontsize=7.3, ha="center", va="center", weight="bold")


def main():
    fig, ax = plt.subplots(figsize=(3.25, 2.10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.18, 0.93, "Raw AV", ha="center", fontsize=9.0, weight="bold")
    ax.text(0.82, 0.93, "SRVR", ha="center", fontsize=9.0, weight="bold")

    box(ax, (0.04, 0.64), "Correct 133", "#edf8f1", "#2b7a4b")
    box(ax, (0.04, 0.16), "Wrong 158", "#fbeeee", "#b54a4a")
    box(ax, (0.68, 0.64), "Correct 237", "#edf8f1", "#2b7a4b")
    box(ax, (0.68, 0.16), "Wrong 54", "#fbeeee", "#b54a4a")

    arrow(ax, (0.32, 0.75), (0.68, 0.75), "#297a4a", 5.8, -0.08, "132 stay correct", (0.50, 0.86))
    arrow(ax, (0.32, 0.27), (0.68, 0.70), "#297a4a", 5.0, -0.06, "105 corrected", (0.48, 0.51))
    arrow(ax, (0.32, 0.23), (0.68, 0.23), "#b54a4a", 3.1, 0.10, "53 remain wrong", (0.50, 0.09))
    arrow(ax, (0.32, 0.69), (0.68, 0.31), "#b54a4a", 0.9, 0.04, "1 harmed", (0.57, 0.46))

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(PNG, dpi=220, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    main()
