#!/usr/bin/env python3
"""Analyze the frozen Qwen3 self-timestamp CMTB-291 secondary sensitivity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path("/workspace/svhalluc")
O = ROOT / "outputs"
PRED = O / "qwen3_self_timestamp_cmtb291_v1.jsonl"
AUTO = O / "qwen3_ter_large_v1.jsonl"
MANIFEST = O / "videollama21_standard_cmtb291_manifest_v1.jsonl"
OUT = O / "qwen3_self_timestamp_cmtb291_v1_analysis.json"
REPORT = O / "qwen3_self_timestamp_cmtb291_v1_report.md"
SEED = 20260814
BOOT = 20_000


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_iou(a: float, b: float, c: float, d: float) -> float:
    inter = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return inter / union if union > 0 else 0.0


def paired(method: dict[str, dict], baseline: dict[str, dict]) -> dict:
    ids = sorted(set(method) & set(baseline))
    ma = np.array([bool(method[s]["correct"]) for s in ids], dtype=int)
    ba = np.array([bool(baseline[s]["correct"]) for s in ids], dtype=int)
    diff = ma - ba
    wc = int(np.sum((ba == 0) & (ma == 1)))
    cw = int(np.sum((ba == 1) & (ma == 0)))
    rng = np.random.default_rng(SEED)
    draws = rng.choice(diff, size=(BOOT, len(diff)), replace=True).mean(axis=1)
    return {
        "n": len(ids),
        "method_accuracy": float(ma.mean()),
        "baseline_accuracy": float(ba.mean()),
        "difference": float(diff.mean()),
        "bootstrap_95ci": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "baseline_wrong_to_method_correct": wc,
        "baseline_correct_to_method_wrong": cw,
        "exact_mcnemar_p": (
            float(binomtest(min(wc, cw), wc + cw, 0.5).pvalue)
            if wc + cw else 1.0
        ),
    }


def main() -> None:
    rows = read(PRED)
    manifest = read(MANIFEST)
    expected = {r["sample_id"] for r in manifest}
    auto_rows = [
        r for r in read(AUTO)
        if r["task"] == "CMTB" and r["condition"] == "TER_V_auto"
    ]
    ids = [r["sample_id"] for r in rows]
    integrity = {
        "rows_291": len(rows) == 291,
        "unique_291": len(set(ids)) == 291,
        "exact_manifest": set(ids) == expected,
        "auto_reference_291": (
            len(auto_rows) == 291
            and {r["sample_id"] for r in auto_rows} == expected
        ),
        "prediction_blind_timing_prompt": all(
            r["provenance"] == "frozen_secondary_self_timestamp_cmtb291_v1"
            and not r["audio_handling"]["answer_query_use_audio_in_video"]
            for r in rows
        ),
        "single_revision": len({r["model_revision"] for r in rows + auto_rows}) == 1,
    }
    if not all(integrity.values()):
        raise RuntimeError(integrity)

    valid = [r for r in rows if r["self_timestamp_status"] in {"json", "labeled_regex"}]
    iou_by_id = {
        r["sample_id"]: interval_iou(
            r["self_timestamp_start_sec"], r["self_timestamp_end_sec"],
            r["frozen_vad_start_sec_analysis_only"],
            r["frozen_vad_end_sec_analysis_only"],
        )
        for r in valid
    }
    ious = list(iou_by_id.values())
    by_iou = {}
    for label, lo, hi in [("lt_0.5", 0.0, 0.5), ("0.5_to_0.8", 0.5, 0.8), ("ge_0.8", 0.8, 1.000001)]:
        subset = [r for r in valid if lo <= iou_by_id[r["sample_id"]] < hi]
        by_iou[label] = {
            "n": len(subset),
            "accuracy": (sum(r["correct"] for r in subset) / len(subset)) if subset else None,
        }

    method = {r["sample_id"]: r for r in rows}
    auto = {r["sample_id"]: r for r in auto_rows}
    result = {
        "status": "PASS_SECONDARY_CMTB291_SELF_REFERENCE_SENSITIVITY",
        "scope": (
            "Full-CMTB secondary sensitivity on an already analyzed benchmark; "
            "not independent dataset generalization or untouched confirmation."
        ),
        "integrity": integrity,
        "timestamp_parse": {
            "valid_n": len(valid),
            "failure_n": 291 - len(valid),
            "status_counts": {
                s: sum(r["self_timestamp_status"] == s for r in rows)
                for s in sorted({r["self_timestamp_status"] for r in rows})
            },
        },
        "timestamp_alignment_to_frozen_vad": {
            "n": len(ious),
            "mean_iou": float(np.mean(ious)) if ious else None,
            "median_iou": float(np.median(ious)) if ious else None,
            "iou_ge_0.5": int(sum(x >= 0.5 for x in ious)),
        },
        "condition_accuracy": {
            "self_timestamp_all291": sum(r["correct"] for r in rows) / 291,
            "self_timestamp_valid_only": (
                sum(r["correct"] for r in valid) / len(valid) if valid else None
            ),
            "auto_vad_timestamp": sum(r["correct"] for r in auto_rows) / 291,
        },
        "accuracy_by_self_vad_iou": by_iou,
        "self_vs_auto_vad_timestamp": paired(method, auto),
        "bootstrap": {"repetitions": BOOT, "seed": SEED, "unit": "video"},
        "hashes": {
            "predictions": sha(PRED),
            "auto_reference_source": sha(AUTO),
            "manifest": sha(MANIFEST),
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    p = result["self_vs_auto_vad_timestamp"]
    a = result["condition_accuracy"]
    t = result["timestamp_alignment_to_frozen_vad"]
    REPORT.write_text(
        "\n".join([
            "# Qwen3 self-timestamp reuse on CMTB-291", "",
            "Status: **PASS secondary sensitivity**", "",
            f"Valid bounds: {len(valid)}/291; median self/VAD IoU: {t['median_iou']:.3f}.",
            f"Accuracy: all-row self {a['self_timestamp_all291']:.1%}, valid-only self {a['self_timestamp_valid_only']:.1%}, automatic VAD {a['auto_vad_timestamp']:.1%}.",
            f"Self minus automatic: {p['difference']:+.1%} [{p['bootstrap_95ci'][0]:+.1%}, {p['bootstrap_95ci'][1]:+.1%}], W→C={p['baseline_wrong_to_method_correct']}, C→W={p['baseline_correct_to_method_wrong']}, p={p['exact_mcnemar_p']:.4g}.",
            "", result["scope"],
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

