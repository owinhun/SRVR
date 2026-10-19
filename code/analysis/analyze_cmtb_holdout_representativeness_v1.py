#!/usr/bin/env python3
"""Audit that held-out CMTB-291 is the full video-disjoint remainder, not an accuracy-selected subset."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from scipy.stats import chi2_contingency


ROOT = Path("/workspace/svhalluc")
OFFICIAL = ROOT / "data/SVHalluc/samples.json"
EVAL = ROOT / "data/ter_large_eval_v1.jsonl"
DEV = ROOT / "data/phase0_cmtb.jsonl"
O = ROOT / "outputs"
T = ROOT / "paper/tables"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def category(video_id: str) -> str:
    parts = video_id.split("/")
    # Accept both repository-relative IDs (``temporal/218/...``) and
    # absolute paths (``.../SVHalluc/temporal/218/...``).
    try:
        return parts[parts.index("temporal") + 1]
    except (ValueError, IndexError):
        return "unknown"


def distribution(rows: list[dict], key) -> dict[str, int]:
    return dict(sorted(collections.Counter(key(row) for row in rows).items()))


def main() -> None:
    official = [r for r in json.loads(OFFICIAL.read_text()) if r["task_type_abbr"] == "CMTB"]
    heldout = [r for r in read_jsonl(EVAL) if r["task"] == "CMTB"]
    dev = read_jsonl(DEV)
    official_videos = {r["video_path"] for r in official}
    heldout_videos = {r["video_id"] for r in heldout}
    dev_videos = {r.get("video_id", r.get("video_path")) for r in dev}
    if dev_videos and not dev_videos <= official_videos:
        # phase0 rows store video_path rather than video_id in some revisions.
        dev_videos = {r["video_path"] for r in dev}

    answer_labels = ["A", "B", "C"]
    held_counts = collections.Counter(r["answer"] for r in heldout)
    dev_counts = collections.Counter(r["answer"] for r in dev)
    chi2, p, _, _ = chi2_contingency([
        [held_counts[x] for x in answer_labels],
        [dev_counts[x] for x in answer_labels],
    ])
    held_categories = collections.Counter(category(r["video_id"]) for r in heldout)
    dev_categories = collections.Counter(category(r["video_path"]) for r in dev)
    all_categories = set(held_categories) | set(dev_categories)
    tv = 0.5 * sum(
        abs(held_categories[x] / len(heldout) - dev_categories[x] / len(dev))
        for x in all_categories
    )
    checks = {
        "official_cmtb_339": len(official) == 339 and len(official_videos) == 339,
        "heldout_cmtb_291": len(heldout) == 291 and len(heldout_videos) == 291,
        "dev_cmtb_48": len(dev) == 48 and len(dev_videos) == 48,
        "video_disjoint": not (heldout_videos & dev_videos),
        "exact_partition": heldout_videos | dev_videos == official_videos,
        "no_prediction_fields_in_selection": not any(
            any(key in row for key in ("prediction", "correct", "accuracy"))
            for row in heldout + dev
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "official_n": len(official), "heldout_n": len(heldout), "development_n": len(dev),
        "heldout_fraction": len(heldout) / len(official),
        "answer_distribution": {
            "official": distribution(official, lambda r: r["answer"]),
            "heldout": dict(sorted(held_counts.items())),
            "development": dict(sorted(dev_counts.items())),
            "heldout_vs_development_chi2": float(chi2),
            "heldout_vs_development_p": float(p),
        },
        "recipe_category_distribution": {
            "heldout": dict(sorted(held_categories.items())),
            "development": dict(sorted(dev_categories.items())),
            "total_variation_distance": tv,
        },
        "selection_rule": "Frozen Dev-48 videos for method development; every remaining official CMTB video enters held-out CMTB-291.",
        "limitation": "Distribution summaries demonstrate lineage and absence of outcome selection; they are not a proof of distributional equivalence.",
        "checks": checks,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (OFFICIAL, EVAL, DEV)},
    }
    (O / "cmtb_holdout_representativeness_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# CMTB holdout representativeness v1", "", f"Status: **{result['status']}**", "",
        f"Held-out CMTB-291 is {result['heldout_fraction']:.1%} of official CMTB-339. It is the complete video-disjoint remainder after frozen Dev-48, not a QC-, prediction-, or accuracy-selected subset.", "",
        "| Split | N | A | B | C |", "|---|---:|---:|---:|---:|",
        f"| Official | {len(official)} | {sum(r['answer']=='A' for r in official)} | {sum(r['answer']=='B' for r in official)} | {sum(r['answer']=='C' for r in official)} |",
        f"| Held-out | {len(heldout)} | {held_counts['A']} | {held_counts['B']} | {held_counts['C']} |",
        f"| Development | {len(dev)} | {dev_counts['A']} | {dev_counts['B']} | {dev_counts['C']} |",
        "", f"Held-out versus development answer-distribution chi-square p={p:.4g}. Recipe-category total-variation distance={tv:.3f}.", "",
        result["limitation"],
    ]
    (O / "cmtb_holdout_representativeness_v1.md").write_text("\n".join(lines) + "\n")
    T.mkdir(parents=True, exist_ok=True)
    (T / "cmtb_split_lineage_v1.csv").write_text(
        "split,n,fraction_of_official,role,selection_rule\n"
        "Official CMTB,339,1.000,benchmark population,all official CMTB rows\n"
        "Development,48,0.142,method development,frozen video-disjoint Dev-48\n"
        "Held-out evaluation,291,0.858,primary evaluation,all remaining official CMTB rows\n"
    )
    (T / "cmtb_split_lineage_v1.tex").write_text(
        "\\begin{tabular}{lrrl}\n"
        "\\toprule\n"
        "Split & $N$ & Official share & Role \\\\\n"
        "\\midrule\n"
        "Official CMTB & 339 & 100.0\\% & benchmark population \\\\\n"
        "Development & 48 & 14.2\\% & frozen method development \\\\\n"
        "Held-out evaluation & 291 & 85.8\\% & complete remaining set \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
