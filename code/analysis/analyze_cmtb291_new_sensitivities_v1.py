#!/usr/bin/env python3
"""Analyze the frozen 2026-08-20 CMTB-291 machine-only sensitivities."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path("/workspace/svhalluc")
SEED = 20260820
BOOTSTRAPS = 20_000


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def keyed(rows: list[dict], condition: str | None = None) -> dict[str, dict]:
    if condition is not None:
        rows = [row for row in rows if row.get("condition") == condition]
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate sample IDs")
    return result


def paired(new: dict[str, dict], baseline: dict[str, dict]) -> dict:
    ids = sorted(set(new) & set(baseline))
    if len(ids) != 291:
        raise RuntimeError(f"expected 291 paired rows, got {len(ids)}")
    x = np.asarray([bool(new[sid]["correct"]) for sid in ids], dtype=np.int8)
    y = np.asarray([bool(baseline[sid]["correct"]) for sid in ids], dtype=np.int8)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(ids), size=(BOOTSTRAPS, len(ids)))
    distribution = 100.0 * (x[indices] - y[indices]).mean(axis=1)
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    p_value = float(binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {
        "n": len(ids), "new_correct": int(x.sum()), "new_accuracy_pct": float(100 * x.mean()),
        "baseline_correct": int(y.sum()), "baseline_accuracy_pct": float(100 * y.mean()),
        "difference_pp": float(100 * (x - y).mean()),
        "bootstrap_95_ci_pp": [float(v) for v in np.quantile(distribution, [0.025, 0.975])],
        "new_correct_baseline_wrong": n10, "new_wrong_baseline_correct": n01,
        "exact_mcnemar_p": p_value,
    }


def interval_iou(a: float, b: float, c: float, d: float) -> float:
    overlap = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return overlap / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs/cmtb291_new_sensitivities_20260820_analysis_v1.json")
    parser.add_argument("--md-output", type=Path, default=ROOT / "outputs/cmtb291_new_sensitivities_20260820_report_v1.md")
    args = parser.parse_args()
    paths = {
        "manifest": ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl",
        "qwen25_auto": ROOT / "outputs/qwen25_ter_replication_v1.jsonl",
        "qwen3_auto": ROOT / "outputs/qwen3_ter_large_v1.jsonl",
        "qwen25_black": ROOT / "outputs/qwen25_cmtb291_black_video_shortcut_v1.jsonl",
        "qwen3_black": ROOT / "outputs/qwen3_cmtb291_black_video_shortcut_v1.jsonl",
        "whisper_intervals": ROOT / "outputs/cmtb291_faster_whisper_small_en_intervals_v1.jsonl",
        "qwen25_whisper": ROOT / "outputs/qwen25_cmtb291_whisper_reference_v1.jsonl",
        "qwen25_self": ROOT / "outputs/qwen25_self_timestamp_cmtb291_v1.jsonl",
    }
    manifest = keyed(read(paths["manifest"]))
    q25_auto = keyed(read(paths["qwen25_auto"]), "TER_V_auto")
    q3_auto = keyed(read(paths["qwen3_auto"]), "TER_V_auto")
    q25_black = keyed(read(paths["qwen25_black"]))
    q3_black = keyed(read(paths["qwen3_black"]))
    whisper = keyed(read(paths["whisper_intervals"]))
    q25_whisper = keyed(read(paths["qwen25_whisper"]))
    q25_self = keyed(read(paths["qwen25_self"]))
    if any(len(rows) != 291 for rows in (manifest, q25_auto, q3_auto, q25_black, q3_black, whisper, q25_whisper, q25_self)):
        raise RuntimeError("every full-population input must contain 291 unique rows")

    whisper_ious = []
    for sid, row in whisper.items():
        base = manifest[sid]
        whisper_ious.append(interval_iou(
            float(row["whisper_start_sec"]), float(row["whisper_end_sec"]),
            float(base["speech_start_sec"]), float(base["speech_end_sec"]),
        ))
    valid_self = {sid: row for sid, row in q25_self.items() if row["self_timestamp_status"] in {"json", "labeled_regex"}}
    self_ious = []
    for sid, row in valid_self.items():
        base = manifest[sid]
        self_ious.append(interval_iou(
            float(row["self_timestamp_start_sec"]), float(row["self_timestamp_end_sec"]),
            float(base["speech_start_sec"]), float(base["speech_end_sec"]),
        ))
    valid_ids = sorted(valid_self)
    valid_correct = sum(bool(valid_self[sid]["correct"]) for sid in valid_ids)
    valid_auto_correct = sum(bool(q25_auto[sid]["correct"]) for sid in valid_ids)

    result = {
        "analysis": {"seed": SEED, "paired_video_bootstraps": BOOTSTRAPS},
        "qwen25_black_vs_auto": paired(q25_black, q25_auto),
        "qwen3_black_vs_auto": paired(q3_black, q3_auto),
        "qwen25_whisper_vs_silero": paired(q25_whisper, q25_auto),
        "qwen25_self_vs_silero_all_rows": paired(q25_self, q25_auto),
        "qwen25_self": {
            "status_counts": dict(collections.Counter(row["self_timestamp_status"] for row in q25_self.values())),
            "valid_n": len(valid_self), "valid_rate_pct": 100 * len(valid_self) / 291,
            "valid_correct": valid_correct, "valid_accuracy_pct": 100 * valid_correct / len(valid_self),
            "silero_accuracy_on_self_valid_subset_pct": 100 * valid_auto_correct / len(valid_self),
            "median_iou_to_silero": float(np.median(self_ious)),
            "iou_iqr": [float(v) for v in np.quantile(self_ious, [0.25, 0.75])],
        },
        "whisper_intervals": {
            "valid_n": sum(row["status"] == "valid" for row in whisper.values()),
            "median_iou_to_silero": float(np.median(whisper_ious)),
            "iou_iqr": [float(v) for v in np.quantile(whisper_ious, [0.25, 0.75])],
            "mean_iou_to_silero": float(np.mean(whisper_ious)),
        },
        "option_counts": {
            "qwen25_black": dict(collections.Counter(str(row.get("prediction")) for row in q25_black.values())),
            "qwen3_black": dict(collections.Counter(str(row.get("prediction")) for row in q3_black.values())),
        },
        "source_sha256": {name: digest(path) for name, path in paths.items()},
    }
    args.json_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    q25b, q3b = result["qwen25_black_vs_auto"], result["qwen3_black_vs_auto"]
    wh, self_all = result["qwen25_whisper_vs_silero"], result["qwen25_self_vs_silero_all_rows"]
    lines = [
        "# CMTB-291 new sensitivities (2026-08-20)\n",
        "All designs were frozen before successful predictions. Differences are new condition minus the existing automatic-Silero reference.\n",
        "| Condition | Accuracy | Automatic reference | Difference (95% paired bootstrap CI) | Discordant new+/new- | McNemar |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Qwen2.5 black video | {q25b['new_accuracy_pct']:.1f}% | {q25b['baseline_accuracy_pct']:.1f}% | {q25b['difference_pp']:+.1f} [{q25b['bootstrap_95_ci_pp'][0]:+.1f}, {q25b['bootstrap_95_ci_pp'][1]:+.1f}] | {q25b['new_correct_baseline_wrong']}/{q25b['new_wrong_baseline_correct']} | {q25b['exact_mcnemar_p']:.3g} |",
        f"| Qwen3 black video | {q3b['new_accuracy_pct']:.1f}% | {q3b['baseline_accuracy_pct']:.1f}% | {q3b['difference_pp']:+.1f} [{q3b['bootstrap_95_ci_pp'][0]:+.1f}, {q3b['bootstrap_95_ci_pp'][1]:+.1f}] | {q3b['new_correct_baseline_wrong']}/{q3b['new_wrong_baseline_correct']} | {q3b['exact_mcnemar_p']:.3g} |",
        f"| Qwen2.5 faster-Whisper reference | {wh['new_accuracy_pct']:.1f}% | {wh['baseline_accuracy_pct']:.1f}% | {wh['difference_pp']:+.1f} [{wh['bootstrap_95_ci_pp'][0]:+.1f}, {wh['bootstrap_95_ci_pp'][1]:+.1f}] | {wh['new_correct_baseline_wrong']}/{wh['new_wrong_baseline_correct']} | {wh['exact_mcnemar_p']:.3g} |",
        f"| Qwen2.5 self reference (all rows) | {self_all['new_accuracy_pct']:.1f}% | {self_all['baseline_accuracy_pct']:.1f}% | {self_all['difference_pp']:+.1f} [{self_all['bootstrap_95_ci_pp'][0]:+.1f}, {self_all['bootstrap_95_ci_pp'][1]:+.1f}] | {self_all['new_correct_baseline_wrong']}/{self_all['new_wrong_baseline_correct']} | {self_all['exact_mcnemar_p']:.3g} |\n",
        f"Qwen2.5 self timing is valid on {len(valid_self)}/291 ({100*len(valid_self)/291:.1f}%), with median IoU {np.median(self_ious):.3f}; valid-only downstream accuracy is {100*valid_correct/len(valid_self):.1f}%. Faster-Whisper is valid on all 291 rows with median IoU {np.median(whisper_ious):.3f}.\n",
        "Interpretation: black-video collapse rejects a timestamp/question/option-only shortcut. Faster-Whisper gives partial detector robustness but incurs a small paired cost. Qwen2.5 does not reproduce Qwen3's reusable self-timing result, so self-reference is model-dependent.\n",
    ]
    args.md_output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
