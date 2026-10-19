#!/usr/bin/env python3
"""Analyze frozen Qwen2.5 CMTB-291 Silero threshold answer sensitivity."""

from __future__ import annotations

import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/svhalluc")
OUT = ROOT / "outputs"
MANIFEST = OUT / "videollama21_standard_cmtb291_manifest_v1.jsonl"
GEOMETRY = OUT / "cmtb291_vad_threshold_sensitivity_geometry_v1.json"
BASELINE = OUT / "qwen25_ter_replication_v1.jsonl"
PRED_GLOB = str(OUT / "qwen25_cmtb291_vad_threshold_sensitivity_v1_shard*.jsonl")
JSON_OUT = OUT / "qwen25_cmtb291_vad_threshold_sensitivity_analysis_v1.json"
REPORT = OUT / "qwen25_cmtb291_vad_threshold_sensitivity_report_v1.md"
THRESHOLDS = (0.3, 0.7)
BOOTSTRAP = 20_000
SEED = 20260814


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def paired(candidate: dict[str, dict], baseline: dict[str, dict], seed_offset: int) -> dict:
    ids = sorted(candidate)
    if set(ids) != set(baseline):
        raise RuntimeError("paired IDs differ")
    candidate_correct = np.asarray([bool(candidate[sample_id]["correct"]) for sample_id in ids], dtype=int)
    baseline_correct = np.asarray([bool(baseline[sample_id]["correct"]) for sample_id in ids], dtype=int)
    difference = candidate_correct - baseline_correct
    gain = int(np.sum((baseline_correct == 0) & (candidate_correct == 1)))
    loss = int(np.sum((baseline_correct == 1) & (candidate_correct == 0)))
    rng = np.random.default_rng(SEED + seed_offset)
    draws = difference[rng.integers(0, len(ids), size=(BOOTSTRAP, len(ids)))].mean(axis=1)
    return {
        "n": len(ids),
        "threshold_accuracy": float(candidate_correct.mean()),
        "threshold_0.5_accuracy": float(baseline_correct.mean()),
        "difference_vs_0.5": float(difference.mean()),
        "paired_bootstrap_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "baseline_wrong_to_threshold_correct": gain,
        "baseline_correct_to_threshold_wrong": loss,
        "exact_mcnemar_p": exact_mcnemar(gain, loss),
    }


def main() -> None:
    manifest = read(MANIFEST)
    expected = {row["sample_id"] for row in manifest}
    if len(expected) != 291 or len(manifest) != 291:
        raise RuntimeError("expected exact CMTB-291 manifest")
    baseline = {
        row["sample_id"]: row for row in read(BASELINE)
        if row.get("task") == "CMTB" and row.get("condition") == "TER_V_auto"
    }
    if set(baseline) != expected:
        raise RuntimeError("threshold-0.5 baseline is incomplete or mismatched")
    paths = [Path(path) for path in sorted(glob.glob(PRED_GLOB))]
    if len(paths) != 2:
        raise RuntimeError("expected exactly two threshold output shards")
    predictions = [row for path in paths for row in read(path)]
    keys = {(row.get("sample_id"), float(row.get("threshold", -1))) for row in predictions}
    if len(predictions) != len(keys) or len(keys) != 291 * len(THRESHOLDS):
        raise RuntimeError(f"threshold rows incomplete or duplicate: {len(predictions)}/{len(keys)}")
    by_threshold = {
        threshold: {
            row["sample_id"]: row for row in predictions
            if float(row["threshold"]) == threshold
        }
        for threshold in THRESHOLDS
    }
    if any(set(rows) != expected for rows in by_threshold.values()):
        raise RuntimeError("threshold population mismatch")
    if any(row.get("protocol") != "cmtb291_vad_threshold_sensitivity_protocol_v1" for row in predictions):
        raise RuntimeError("protocol provenance mismatch")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    if geometry.get("status") != "PASS_READY_FOR_FROZEN_QWEN25_INFERENCE":
        raise RuntimeError("geometry audit not pass")
    results = {
        str(threshold): {
            "answer_sensitivity": paired(by_threshold[threshold], baseline, index),
            "interval_geometry": geometry["thresholds"][str(threshold)],
            "parse_failures_counted_wrong": sum(
                row.get("parse_status") != "parsed" for row in by_threshold[threshold].values()
            ),
        }
        for index, threshold in enumerate(THRESHOLDS)
    }
    result = {
        "status": "PASS_COMPLETE_CMTB291_VAD_THRESHOLD_SENSITIVITY",
        "scope": "Frozen detector-threshold robustness sensitivity on CMTB-291; not external generalization.",
        "n": 291,
        "thresholds": results,
        "bootstrap": {"repetitions": BOOTSTRAP, "seed": SEED, "unit": "CMTB video"},
        "hashes": {
            "manifest": sha(MANIFEST), "geometry": sha(GEOMETRY), "baseline": sha(BASELINE),
            **{str(path.relative_to(ROOT)): sha(path) for path in paths},
            "analysis": sha(Path(__file__)),
        },
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# Qwen2.5 CMTB-291 VAD threshold sensitivity", "", "Status: **PASS complete**", "", "| Threshold | Accuracy | Δ vs .5 | 95% CI | McNemar p | Median IoU to .5 | Median coverage |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for threshold in THRESHOLDS:
        answer = results[str(threshold)]["answer_sensitivity"]
        geo = results[str(threshold)]["interval_geometry"]
        lines.append(
            f"| {threshold:.1f} | {100 * answer['threshold_accuracy']:.1f}% | "
            f"{100 * answer['difference_vs_0.5']:+.1f} pp | "
            f"[{100 * answer['paired_bootstrap_95ci'][0]:+.1f}, {100 * answer['paired_bootstrap_95ci'][1]:+.1f}] | "
            f"{answer['exact_mcnemar_p']:.4g} | {geo['median_iou_to_0.5']:.3f} | {100 * geo['median_coverage']:.1f}% |"
        )
    lines.extend(["", result["scope"], ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
