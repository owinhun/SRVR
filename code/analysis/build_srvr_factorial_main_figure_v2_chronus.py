#!/usr/bin/env python3
"""Build the mechanism + three-model full-N factorial main figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path("/workspace/svhalluc")
SOURCE = ROOT / "outputs/audio_pointer_factorial_full291_v1.json"
CHRONUS = ROOT / "outputs/chronusomni_cmtb291_factorial_analysis_v1.json"
CHRONUS_SHARDS = [
    ROOT / f"outputs/chronusomni_cmtb291_factorial_v1_shard{index}.jsonl"
    for index in range(4)
]
PDF = ROOT / "paper/figures/srvr_factorial_main_v2_chronus.pdf"
PNG = ROOT / "paper/figures/srvr_factorial_main_v2_chronus.png"
BLUE = "#1764AB"
RED = "#D1495B"
GREEN = "#2A7F62"
CHRONUS_CONDITIONS = {
    "P0A0": "no_pointer_audio_off",
    "P0A1": "no_pointer_audio_on",
    "P1A0": "correct_pointer_audio_off",
    "P1A1": "correct_pointer_audio_on",
}


def box(axis, xy, width, height, text, edge, face="#FFFFFF", size=7.4) -> None:
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor=edge, facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text,
              ha="center", va="center", fontsize=size, color="#222222")


def mechanism(axis) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("(a) Why occurrence-only routing?", fontsize=9.3,
                   fontweight="bold", pad=4)
    box(axis, (0.08, 0.78), 0.84, 0.15,
        "Speech-bearing audio at $\\tau$", "#555555", "#F5F5F5", 8.2)
    box(axis, (0.02, 0.56), 0.46, 0.14,
        "Content $c$\nnarrated A", RED, "#FFF2F3", 6.9)
    box(axis, (0.52, 0.56), 0.46, 0.14,
        "Occurrence $\\tau$\n1.2–9.5 s", BLUE, "#EFF6FD", 6.9)
    axis.annotate("", xy=(0.25, 0.71), xytext=(0.38, 0.78),
                  arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.2})
    axis.annotate("", xy=(0.75, 0.71), xytext=(0.62, 0.78),
                  arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2})
    box(axis, (0.02, 0.31), 0.46, 0.15,
        "Standard AV\nNarrated A ✗", RED, "#FFF2F3", 6.7)
    box(axis, (0.52, 0.31), 0.46, 0.15,
        "SRVR\nAccess at $\\tau$", BLUE, "#EFF6FD", 6.7)
    axis.annotate("", xy=(0.25, 0.47), xytext=(0.25, 0.55),
                  arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.2})
    axis.annotate("", xy=(0.75, 0.47), xytext=(0.75, 0.55),
                  arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2})
    box(axis, (0.52, 0.08), 0.46, 0.13,
        "Visual C ✓\n(audio off)", GREEN, "#F0F9F5", 6.9)
    axis.annotate("", xy=(0.75, 0.22), xytext=(0.75, 0.30),
                  arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.2})
    axis.text(0.10, 0.02, "Occurrence→vision; audio muted.",
              fontsize=6.8, color="#444444")


def interaction(axis, model: dict, title: str, panel: str) -> None:
    x = np.asarray([0.0, 1.0])
    styles = {
        "off": {"color": BLUE, "marker": "o", "label": "Audio off"},
        "on": {"color": RED, "marker": "s", "label": "Audio on"},
    }
    for audio_key, cells in (("off", ("P0A0", "P1A0")), ("on", ("P0A1", "P1A1"))):
        values = np.asarray([model["cell_accuracy"][cell] * 100 for cell in cells])
        ci = np.asarray([model["cell_bootstrap_95ci"][cell] for cell in cells]) * 100
        error = np.vstack((values - ci[:, 0], ci[:, 1] - values))
        style = styles[audio_key]
        axis.errorbar(
            x, values, yerr=error, color=style["color"], marker=style["marker"],
            markersize=5, linewidth=1.8, capsize=2.5, label=style["label"], zorder=3,
        )
    off = np.asarray([model["cell_accuracy"]["P0A0"], model["cell_accuracy"]["P1A0"]]) * 100
    on = np.asarray([model["cell_accuracy"]["P0A1"], model["cell_accuracy"]["P1A1"]]) * 100
    for index, xx in enumerate(x):
        for key, yy, other in (("off", off[index], on[index]), ("on", on[index], off[index])):
            axis.annotate(
                f"{yy:.1f}", (xx, yy), xytext=(0, 6 if yy >= other else -12),
                textcoords="offset points", ha="center", fontsize=6.8,
                color=styles[key]["color"],
            )
    ci = np.asarray(model["interaction_bootstrap_95ci"]) * 100
    axis.text(
        0.5, 0.035,
        f"$\\Delta_{{int}}$={model['probability_scale_interaction'] * 100:+.1f} pp\n"
        f"[{ci[0]:+.1f}, {ci[1]:+.1f}]",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB"},
    )
    axis.set_title(f"({panel}) {title}", fontsize=8.8, fontweight="bold")
    axis.set_xticks(x, ["No\npointer", "Correct\npointer"])
    axis.set_xlim(-0.18, 1.18)
    axis.set_ylim(35, 100)
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=7.3)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chronus_panel() -> dict:
    analysis = json.loads(CHRONUS.read_text())
    if analysis.get("status") != "PASS_COMPLETE_FULL_POPULATION_ANALYSIS":
        raise RuntimeError("Chronus analysis did not pass")
    rows = [row for path in CHRONUS_SHARDS for row in read_jsonl(path)]
    ids = sorted({row["sample_id"] for row in rows})
    by_key = {(row["sample_id"], row["condition"]): row for row in rows}
    rng = np.random.default_rng(20260814)
    cell_accuracy = {}
    cell_bootstrap = {}
    for cell, condition in CHRONUS_CONDITIONS.items():
        values = np.array([int(bool(by_key[(sample_id, condition)]["correct"])) for sample_id in ids])
        cell_accuracy[cell] = float(values.mean())
        boot = np.empty(20_000)
        for iteration in range(len(boot)):
            boot[iteration] = values[rng.integers(0, len(values), size=len(values))].mean()
        cell_bootstrap[cell] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    interaction_effect = analysis["effects"]["audio_x_pointer_interaction"]
    return {
        "cell_accuracy": cell_accuracy,
        "cell_bootstrap_95ci": cell_bootstrap,
        "probability_scale_interaction": interaction_effect["point_estimate_probability_points"] / 100,
        "interaction_bootstrap_95ci": [
            value / 100 for value in interaction_effect["ci95_probability_points"]
        ],
    }


def main() -> None:
    data = json.loads(SOURCE.read_text())
    if data["status"] != "PASS":
        raise RuntimeError("factorial artifact did not pass")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 4, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1.12, 1, 1, 1]},
        sharey=False,
    )
    mechanism(axes[0])
    interaction(axes[1], data["models"]["qwen25"], "Qwen2.5", "b")
    interaction(axes[2], data["models"]["qwen3"], "Qwen3", "c")
    interaction(axes[3], chronus_panel(), "Chronus", "d")
    axes[2].set_yticklabels([])
    axes[3].set_yticklabels([])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.71, 1.02),
               ncol=2, frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.84, bottom=0.12, wspace=0.35)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(PDF)
    print(PNG)


if __name__ == "__main__":
    main()
