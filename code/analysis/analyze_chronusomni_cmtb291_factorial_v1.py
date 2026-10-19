#!/usr/bin/env python3
"""Analyze the frozen full-population ChronusOmni CMTB factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
ANNOTATIONS = ROOT / "outputs/nor_annotations_annotator_2.jsonl"
QWEN3 = ROOT / "outputs/qwen3_ter_large_v1.jsonl"
ANALYSIS = ROOT / "outputs/chronusomni_cmtb291_factorial_analysis_v1.json"
REPORT = ROOT / "outputs/chronusomni_cmtb291_factorial_report_v1.md"
CONDITIONS = (
    "no_pointer_audio_off",
    "no_pointer_audio_on",
    "correct_pointer_audio_off",
    "correct_pointer_audio_on",
)
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_REPLICATES = 20000
ANNOTATIONS_SHA256 = "e588a72bcb6c9d7b64fd852cf22608f73cb1741665191bb4f56cf858568bd064"
QWEN3_SHA256 = "149d50489bc87557d766dc64c45cc2cf64bfa37368fa7e8da28cd18707401973"
MATERIAL_RAW_MARGIN_POINTS = 5.0


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", type=Path, nargs="+",
        default=[ROOT / f"outputs/chronusomni_cmtb291_factorial_v1_shard{i}.jsonl" for i in range(4)],
    )
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--report", type=Path, default=REPORT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def paired_summary(a: np.ndarray, b: np.ndarray) -> dict:
    wrong_to_correct = int(np.sum((a == 0) & (b == 1)))
    correct_to_wrong = int(np.sum((a == 1) & (b == 0)))
    discordant = wrong_to_correct + correct_to_wrong
    p_value = 1.0 if discordant == 0 else binomtest(
        min(wrong_to_correct, correct_to_wrong), discordant, 0.5, alternative="two-sided"
    ).pvalue
    return {
        "delta_probability_points": float(100 * (b.mean() - a.mean())),
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "both_correct": int(np.sum((a == 1) & (b == 1))),
        "both_wrong": int(np.sum((a == 0) & (b == 0))),
        "exact_mcnemar_p": float(p_value),
    }


def binary_transition_summary(a: np.ndarray, b: np.ndarray) -> dict:
    off_to_on = int(np.sum((a == 0) & (b == 1)))
    on_to_off = int(np.sum((a == 1) & (b == 0)))
    discordant = off_to_on + on_to_off
    p_value = 1.0 if discordant == 0 else binomtest(
        min(off_to_on, on_to_off), discordant, 0.5, alternative="two-sided"
    ).pvalue
    return {
        "non_narrated_to_narrated": off_to_on,
        "narrated_to_non_narrated": on_to_off,
        "both_narrated": int(np.sum((a == 1) & (b == 1))),
        "neither_narrated": int(np.sum((a == 0) & (b == 0))),
        "exact_paired_p": float(p_value),
    }


def bootstrap(matrix: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = matrix.shape[0]
    metrics = {
        "practical_srvr": np.empty(BOOTSTRAP_REPLICATES),
        "pointer_audio_off": np.empty(BOOTSTRAP_REPLICATES),
        "pointer_audio_on": np.empty(BOOTSTRAP_REPLICATES),
        "audio_no_pointer": np.empty(BOOTSTRAP_REPLICATES),
        "audio_correct_pointer": np.empty(BOOTSTRAP_REPLICATES),
        "audio_x_pointer_interaction": np.empty(BOOTSTRAP_REPLICATES),
    }
    for iteration in range(BOOTSTRAP_REPLICATES):
        sample = matrix[rng.integers(0, n, size=n)].mean(axis=0)
        c00, c01, c10, c11 = sample
        metrics["practical_srvr"][iteration] = c10 - c01
        metrics["pointer_audio_off"][iteration] = c10 - c00
        metrics["pointer_audio_on"][iteration] = c11 - c01
        metrics["audio_no_pointer"][iteration] = c01 - c00
        metrics["audio_correct_pointer"][iteration] = c11 - c10
        metrics["audio_x_pointer_interaction"][iteration] = (c11 - c10) - (c01 - c00)
    return {
        name: {
            "ci95_probability_points": [
                float(100 * np.percentile(values, 2.5)),
                float(100 * np.percentile(values, 97.5)),
            ],
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        }
        for name, values in metrics.items()
    }


def bootstrap_paired_delta(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    n = len(a)
    values = np.empty(BOOTSTRAP_REPLICATES)
    for iteration in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, n, size=n)
        values[iteration] = b[indices].mean() - a[indices].mean()
    return {
        "point_estimate_probability_points": float(100 * (b.mean() - a.mean())),
        "ci95_probability_points": [
            float(100 * np.percentile(values, 2.5)),
            float(100 * np.percentile(values, 97.5)),
        ],
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED + 1,
    }


def main() -> None:
    args = cli()
    manifest = read_jsonl(MANIFEST)
    expected = {row["sample_id"] for row in manifest}
    rows = [row for path in args.predictions for row in read_jsonl(path)]
    keys = [(row["sample_id"], row["condition"]) for row in rows]
    if len(expected) != 291 or len(rows) != 291 * 4 or len(keys) != len(set(keys)):
        raise RuntimeError("full-population count or uniqueness failure")
    by_key = {(row["sample_id"], row["condition"]): row for row in rows}
    expected_keys = {(sample_id, condition) for sample_id in expected for condition in CONDITIONS}
    if set(by_key) != expected_keys:
        raise RuntimeError("sample-condition coverage mismatch")
    sample_ids = sorted(expected)
    matrix = np.array(
        [[int(bool(by_key[(sample_id, condition)].get("correct"))) for condition in CONDITIONS] for sample_id in sample_ids],
        dtype=np.int8,
    )
    cells = {}
    for index, condition in enumerate(CONDITIONS):
        successes = int(matrix[:, index].sum())
        condition_rows = [by_key[(sample_id, condition)] for sample_id in sample_ids]
        cells[condition] = {
            "correct": successes,
            "total": 291,
            "accuracy_percent": 100 * successes / 291,
            "wilson_ci95_percent": [100 * value for value in wilson(successes, 291)],
            "parse_failures": sum(row.get("prediction") not in {"A", "B", "C"} for row in condition_rows),
            "runtime_failures": sum(row.get("parse_status") == "runtime_failure" for row in condition_rows),
            "prediction_distribution": Counter(
                row.get("prediction") if row.get("prediction") in {"A", "B", "C"} else "PARSE_FAILURE"
                for row in condition_rows
            ),
        }
    c00, c01, c10, c11 = (matrix[:, index] for index in range(4))
    point = {
        "practical_srvr": 100 * float(c10.mean() - c01.mean()),
        "pointer_audio_off": 100 * float(c10.mean() - c00.mean()),
        "pointer_audio_on": 100 * float(c11.mean() - c01.mean()),
        "audio_no_pointer": 100 * float(c01.mean() - c00.mean()),
        "audio_correct_pointer": 100 * float(c11.mean() - c10.mean()),
        "audio_x_pointer_interaction": 100 * float((c11.mean() - c10.mean()) - (c01.mean() - c00.mean())),
    }
    comparisons = {
        "practical_srvr_no_pointer_audio_on_to_correct_pointer_audio_off": paired_summary(c01, c10),
        "pointer_effect_audio_off": paired_summary(c00, c10),
        "pointer_effect_audio_on": paired_summary(c01, c11),
        "audio_effect_no_pointer": paired_summary(c00, c01),
        "audio_effect_correct_pointer": paired_summary(c10, c11),
    }
    if sha256(QWEN3) != QWEN3_SHA256:
        raise RuntimeError("frozen Qwen3 stronger-raw comparator hash mismatch")
    qwen3_rows = [
        row
        for row in read_jsonl(QWEN3)
        if row.get("condition") == "O_raw" and row.get("sample_id") in expected
    ]
    qwen3_by_id = {row["sample_id"]: row for row in qwen3_rows}
    if len(qwen3_rows) != 291 or set(qwen3_by_id) != expected:
        raise RuntimeError("Qwen3 O_raw comparator population mismatch")
    qwen3_raw = np.array(
        [int(bool(qwen3_by_id[sample_id].get("correct"))) for sample_id in sample_ids], dtype=np.int8
    )
    raw_vs_qwen3 = bootstrap_paired_delta(qwen3_raw, c01)
    raw_vs_qwen3["paired_transition"] = paired_summary(qwen3_raw, c01)
    if sha256(ANNOTATIONS) != ANNOTATIONS_SHA256:
        raise RuntimeError("frozen single-human role map hash mismatch")
    annotations = read_jsonl(ANNOTATIONS)
    roles = {
        row["sample_id"]: row["offtime_narrated_option"]
        for row in annotations
        if row.get("offtime_narrated_option") in {"A", "B", "C"}
        and row.get("unrelated_option") in {"A", "B", "C"}
    }
    if len(roles) != 49 or not set(roles).issubset(expected):
        raise RuntimeError(f"frozen determinate role population mismatch: {len(roles)}")
    role_ids = sorted(roles)
    narrated = {
        condition: np.array(
            [int(by_key[(sample_id, condition)].get("prediction") == roles[sample_id]) for sample_id in role_ids],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }
    role_diagnostic = {
        "scope": "post-hoc directional diagnostic on 49 determinate roles from one genuine prediction-blind human; not prevalence or inter-human reliability",
        "n_determinate": len(role_ids),
        "narrated_selection_count": {
            condition: int(narrated[condition].sum()) for condition in CONDITIONS
        },
        "transitions": {
            "practical_srvr_no_pointer_audio_on_to_correct_pointer_audio_off": binary_transition_summary(
                narrated["no_pointer_audio_on"], narrated["correct_pointer_audio_off"]
            ),
            "pointer_effect_audio_off": binary_transition_summary(
                narrated["no_pointer_audio_off"], narrated["correct_pointer_audio_off"]
            ),
            "pointer_effect_audio_on": binary_transition_summary(
                narrated["no_pointer_audio_on"], narrated["correct_pointer_audio_on"]
            ),
            "audio_effect_no_pointer": binary_transition_summary(
                narrated["no_pointer_audio_off"], narrated["no_pointer_audio_on"]
            ),
            "audio_effect_correct_pointer": binary_transition_summary(
                narrated["correct_pointer_audio_off"], narrated["correct_pointer_audio_on"]
            ),
        },
    }
    bootstrap_results = bootstrap(matrix)
    for name, estimate in point.items():
        bootstrap_results[name]["point_estimate_probability_points"] = estimate
    raw_low, raw_high = raw_vs_qwen3["ci95_probability_points"]
    practical_low, practical_high = bootstrap_results["practical_srvr"]["ci95_probability_points"]
    interaction_low, interaction_high = bootstrap_results["audio_x_pointer_interaction"][
        "ci95_probability_points"
    ]
    promotion_checks = {
        "raw_materially_exceeds_existing_strongest_raw": (
            raw_vs_qwen3["point_estimate_probability_points"] >= MATERIAL_RAW_MARGIN_POINTS
            and raw_low > 0
        ),
        "practical_srvr_positive_ci_excludes_zero": practical_low > 0,
        "interaction_ci_excludes_zero": interaction_low > 0 or interaction_high < 0,
    }
    promotion = {
        "decision": "PROMOTE_AT_LEAST_ONE_FROZEN_GATE" if any(promotion_checks.values()) else "SUPPLEMENT_ONLY",
        "checks": promotion_checks,
        "material_raw_margin_points_frozen_before_full_result": MATERIAL_RAW_MARGIN_POINTS,
    }
    payload = {
        "status": "PASS_COMPLETE_FULL_POPULATION_ANALYSIS",
        "population_n": 291,
        "inference_rows": len(rows),
        "conditions": list(CONDITIONS),
        "cells": cells,
        "effects": bootstrap_results,
        "paired_comparisons": comparisons,
        "chronus_raw_vs_existing_qwen3_raw": raw_vs_qwen3,
        "promotion": promotion,
        "single_human_narrated_option_directional_diagnostic": role_diagnostic,
        "accuracy_not_used_for_inclusion_or_execution_gate": True,
        "hashes": {
            "manifest": sha256(MANIFEST),
            "single_human_annotations": sha256(ANNOTATIONS),
            "qwen3_raw_comparator": sha256(QWEN3),
            "prediction_shards": {str(path): sha256(path) for path in args.predictions},
            "script": sha256(Path(__file__)),
        },
    }
    args.analysis.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# ChronusOmni CMTB-291 factorial report v1", "",
        "Status: **PASS complete full-population analysis**", "",
        "## Cells", "",
        "| Condition | Correct / 291 | Accuracy | Parse failures |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        cell = cells[condition]
        lines.append(
            f"| `{condition}` | {cell['correct']} / 291 | {cell['accuracy_percent']:.1f}% | {cell['parse_failures']} |"
        )
    lines += ["", "## Frozen effects", "", "| Estimand | Difference | 95% video-bootstrap CI |", "|---|---:|---:|"]
    for name in (
        "practical_srvr", "pointer_audio_off", "pointer_audio_on", "audio_no_pointer",
        "audio_correct_pointer", "audio_x_pointer_interaction",
    ):
        effect = bootstrap_results[name]
        low, high = effect["ci95_probability_points"]
        lines.append(f"| `{name}` | {effect['point_estimate_probability_points']:+.1f}%p | [{low:+.1f}, {high:+.1f}]%p |")
    lines += [
        "", "## Existing strongest raw comparator", "",
        f"Chronus raw AV minus Qwen3 raw AV: {raw_vs_qwen3['point_estimate_probability_points']:+.1f}%p "
        f"(95% paired video-bootstrap CI [{raw_low:+.1f}, {raw_high:+.1f}]%p).", "",
        f"Frozen promotion decision: **{promotion['decision']}**.", "",
        "", "## Single-human narrated-option direction", "",
        "This is a post-hoc directional diagnostic on 49 determinate roles, not a prevalence or inter-human result.", "",
        "| Condition | Narrated selections / 49 |", "|---|---:|",
    ]
    for condition in CONDITIONS:
        lines.append(
            f"| `{condition}` | {role_diagnostic['narrated_selection_count'][condition]} / 49 |"
        )
    lines += ["", "| Directional contrast | Non-narrated->narrated | Narrated->non-narrated |", "|---|---:|---:|"]
    for name in (
        "practical_srvr_no_pointer_audio_on_to_correct_pointer_audio_off",
        "audio_effect_correct_pointer",
    ):
        transition = role_diagnostic["transitions"][name]
        lines.append(
            f"| `{name}` | {transition['non_narrated_to_narrated']} | "
            f"{transition['narrated_to_non_narrated']} |"
        )
    lines += [
        "", "The result is scoped to frozen CMTB current-time binding. It is not an official ChronusAV reproduction.", "",
    ]
    args.report.write_text("\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
