#!/usr/bin/env python3
"""Render a compact, self-contained CMTB example for the first page."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path("/workspace/svhalluc")
FRAME = ROOT / "paper/figures/cmtb_example_frame_v66.png"
VERSION = os.environ.get("SRVR_FIGURE_VERSION", "v66")
if VERSION not in {"v66", "v90", "v94", "v96"}:
    raise ValueError(f"unsupported figure version: {VERSION}")
PDF = ROOT / f"paper/figures/cmtb_example_{VERSION}.pdf"
PNG = ROOT / f"paper/figures/cmtb_example_{VERSION}.png"
AUDIT = ROOT / f"outputs/cmtb_example_figure_{VERSION}_audit.json"

INK = "#17202A"
TEAL = "#16827A"
TEAL_PALE = "#E8F5F2"
RED = "#C2413B"
RED_PALE = "#FCEFED"
BLUE = "#236AA1"
BLUE_PALE = "#EDF5FA"
GREY = "#65717C"
GRID = "#DDE3E8"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rounded_box(axis, xy, width, height, face, edge, radius=.02, lw=.8):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=.012,rounding_size={radius}",
        transform=axis.transAxes, facecolor=face, edgecolor=edge, linewidth=lw,
    )
    axis.add_patch(box)
    return box


def main() -> None:
    if not FRAME.exists():
        raise FileNotFoundError(FRAME)
    figure_font = "STIXGeneral" if VERSION == "v96" else "DejaVu Sans"
    math_font = "stix" if VERSION == "v96" else "dejavusans"
    plt.rcParams.update({
        "font.family": figure_font, "font.size": 9.5,
        "mathtext.fontset": math_font,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.08, 2.05), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.03, 1.42], wspace=.13)

    # (a) Current visual evidence.
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(mpimg.imread(FRAME))
    ax.set_anchor("N")
    ax.set_axis_off()
    ax.set_title("(a) Visual evidence", loc="left", fontsize=10.2, fontweight="bold", color=INK, pad=2)
    ax.text(
        .03, .06, "Current $y$\nadd cubed bread",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
        color="white", fontweight="bold",
        bbox={"boxstyle": "round,pad=.25", "facecolor": TEAL, "edgecolor": "white", "alpha": .94},
    )

    # (b) Speech carries the time and an off-time description.
    ax = fig.add_subplot(grid[0, 1])
    ax.set_axis_off()
    ax.set_title("(b) Speech", loc="left", fontsize=10.2, fontweight="bold", color=INK, pad=2)
    rounded_box(ax, (.02, .56), .96, .28, RED_PALE, "#E7A8A2")
    compact = VERSION in {"v90", "v94", "v96"}
    if VERSION in {"v94", "v96"}:
        # Separate the heading, quote, and d_n tag at final paper scale.
        narration_y, quote_y, dn_x, dn_y = .80, .64, .97, .80
    else:
        narration_y, quote_y = .75, .655
        dn_x, dn_y = .91, (.75 if VERSION == "v90" else .63)
    ax.text(.07, narration_y, "Narration", transform=ax.transAxes, color=RED,
            fontweight="bold", fontsize=9.5)
    ax.text(
        .07, quote_y, '“half a cup of milk and\none egg and mix”',
        transform=ax.transAxes, color=INK, fontsize=9.5, va="center",
    )
    ax.text(dn_x, dn_y, "$d_n$", transform=ax.transAxes, color=RED, fontsize=11,
            fontweight="bold", ha="right")

    # Timeline is normalized to the 15.08 s audited media duration.
    x0, x1, y = .06, .94, .36
    ax.plot([x0, x1], [y, y], transform=ax.transAxes, color=GREY, linewidth=1.1)
    start = x0 + (1.122 / 15.084) * (x1 - x0)
    end = x0 + (9.054 / 15.084) * (x1 - x0)
    frame = x0 + (4.5 / 15.084) * (x1 - x0)
    ax.add_patch(Rectangle((start, y-.045), end-start, .09, transform=ax.transAxes,
                           facecolor=BLUE, edgecolor="none", alpha=.9))
    # stop the frame marker below the speech-interval label so it does not
    # strike through the text
    ax.plot([frame, frame], [y-.11, y+.055], transform=ax.transAxes, color=TEAL, linewidth=1.7)
    ax.text((start+end)/2, y+.09, "speech 1.12–9.05 s", transform=ax.transAxes,
            ha="center", color=BLUE, fontsize=9.5, fontweight="bold")
    # single line: the two-line form reached down into the panel caption
    ax.text(frame, y-.14, "frame 4.5 s", transform=ax.transAxes,
            ha="center", va="top", color=TEAL, fontsize=9.5, fontweight="bold")
    ax.text(x0, y-.06, "0", transform=ax.transAxes, ha="center", va="top", color=GREY, fontsize=9.5)
    ax.text(x1, y-.06, "15.1 s", transform=ax.transAxes, ha="center", va="top", color=GREY, fontsize=9.5)
    if VERSION == "v66":
        ax.text(.5, .02, "Occurrence points; content distracts.", transform=ax.transAxes,
                ha="center", va="bottom", color=INK, fontsize=9.5, fontweight="bold")

    # (c) Actual question and model transition.
    ax = fig.add_subplot(grid[0, 2])
    ax.set_axis_off()
    ax.set_title("(c) Question and answers", loc="left", fontsize=10.2,
                 fontweight="bold", color=INK, pad=2)
    # the question sits above the option stack; keep a clear gap so it never
    # collides with the first box (whose top edge is at `top` + height)
    ax.text(.01, .97, "When speech is heard,\nwhat visual action occurs?", transform=ax.transAxes,
            fontsize=9.5, color=INK, fontweight="bold", va="top")
    options = [
        ("A", "add cubed bread", TEAL_PALE, TEAL, "correct $y$"),
        ("B", "pour and spread sauce", "#F5F7F8", "#AAB2BD", "unrelated $d_u$"),
        ("C", "pour milk and egg", RED_PALE, RED, "narrated $d_n$"),
    ]
    top = .58 if VERSION in {"v94", "v96"} else .59
    for index, (letter, text, face, edge, role) in enumerate(options):
        step = .19 if VERSION in {"v94", "v96"} else .18
        box_height = .16 if VERSION in {"v94", "v96"} else .13
        yy = top - index * step
        rounded_box(ax, (.01, yy), .98, box_height, face, edge, radius=.018, lw=.75)
        ax.text(.05, yy+box_height/2, letter, transform=ax.transAxes, ha="center", va="center",
                color=edge, fontweight="bold", fontsize=9.5)
        if VERSION in {"v94", "v96"}:
            main_y, role_y = yy + .125, yy + .030
        else:
            main_y = yy + (.092 if VERSION == "v90" else .085)
            role_y = yy + (.026 if VERSION == "v90" else .035)
        ax.text(.11, main_y, text, transform=ax.transAxes, ha="left", va="center",
                color=INK, fontsize=9.5)
        ax.text(.11, role_y, role, transform=ax.transAxes, ha="left", va="center",
                color=edge if index != 1 else GREY, fontsize=9.5)
    if compact:
        raw_label = "Raw AV: C" if VERSION == "v96" else "Raw AV: C  ✗"
        srvr_label = "SRVR: A" if VERSION == "v96" else "SRVR: A  ✓"
        ax.text(.02, .045, raw_label, transform=ax.transAxes, color=RED,
                fontweight="bold", fontsize=9.5, ha="left")
        ax.annotate("", xy=(.66, .060), xytext=(.39, .060), xycoords=ax.transAxes,
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": GREY})
        ax.text(.70, .045, srvr_label, transform=ax.transAxes, color=TEAL,
                fontweight="bold", fontsize=9.5, ha="left")
    else:
        ax.text(.02, .055, "Raw AV", transform=ax.transAxes, color=RED, fontweight="bold", fontsize=9.5)
        ax.text(.22, .055, "C  ✗", transform=ax.transAxes, color=RED, fontweight="bold", fontsize=9.5)
        ax.annotate("", xy=(.62, .070), xytext=(.35, .070), xycoords=ax.transAxes,
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": GREY})
        ax.text(.66, .055, "SRVR", transform=ax.transAxes, color=TEAL, fontweight="bold", fontsize=9.5)
        ax.text(.88, .055, "A  ✓", transform=ax.transAxes, color=TEAL, fontweight="bold", fontsize=9.5)

    fig.subplots_adjust(left=.006, right=.997, top=.92, bottom=.015)
    fig.savefig(PDF, bbox_inches="tight", pad_inches=.02)
    fig.savefig(PNG, dpi=260, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)

    audit = {
        "status": "PASS_ILLUSTRATIVE_LOCKED_ITEM",
        "version": VERSION,
        "sample_id": "cmtb_001423",
        "audited_interval_sec": [1.122, 9.054],
        "frame_time_sec": 4.5,
        "roles": {"current": "A", "narrated": "C", "unrelated": "B"},
        "predictions": {"qwen3_raw_av": "C", "qwen3_srvr": "A"},
        "frame_sha256": sha(FRAME),
        "pdf_sha256": sha(PDF),
        "png_sha256": sha(PNG),
        "script_sha256": sha(Path(__file__)),
        "boundary": "Illustrative locked CMTB item; not a prevalence estimate.",
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
