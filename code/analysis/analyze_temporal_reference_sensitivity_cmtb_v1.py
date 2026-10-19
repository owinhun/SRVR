#!/usr/bin/env python3
"""Post-hoc CMTB headroom audit for temporal-reference evidence sensitivity.

This script discovers no new evaluation condition.  It reuses the frozen
alpha-0 answer and the four prospectively generated maximal-displacement
answers.  Its result is explicitly developmental; an external, unseen
counterfactual condition is required for confirmation.
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/svhalluc")
OUT = ROOT / "outputs/temporal_reference_sensitivity_cmtb_v1.json"
REPORT = ROOT / "outputs/temporal_reference_sensitivity_cmtb_v1.md"
POINTER = ROOT / "outputs/cmtb291_maximal_displacement_pointer_v1.jsonl"
BOOTSTRAP = 20_000
SEED = 20260810


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alpha0(path: Path) -> dict[str, dict]:
    rows = [
        row for row in jsonl(path)
        if row.get("task") == "CMTB" and row.get("condition") == "TER_V_auto"
    ]
    if len(rows) != 291 or len({row["sample_id"] for row in rows}) != 291:
        raise RuntimeError(f"invalid alpha0 population: {path}")
    return {row["sample_id"]: row for row in rows}


def displaced(pattern: str) -> dict[str, dict[float, dict]]:
    grouped: dict[str, dict[float, dict]] = defaultdict(dict)
    for name in sorted(glob.glob(str(ROOT / pattern))):
        for row in jsonl(Path(name)):
            grouped[row["sample_id"]][float(row["pointer_alpha"])] = row
    if set(map(len, grouped.values())) != {4}:
        raise RuntimeError(f"invalid displaced population: {pattern}")
    return grouped


def auc(scores: list[float], labels: list[bool]) -> float:
    positive = [score for score, label in zip(scores, labels) if label]
    negative = [score for score, label in zip(scores, labels) if not label]
    if not positive or not negative:
        return float("nan")
    total = 0.0
    for pos in positive:
        for neg in negative:
            total += float(pos > neg) + 0.5 * float(pos == neg)
    return total / (len(positive) * len(negative))


def bootstrap_delta(correct: np.ndarray, accept: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(correct), size=len(correct))
        sampled_correct = correct[idx]
        sampled_accept = accept[idx]
        if sampled_accept.sum() == 0:
            continue
        values.append(float(sampled_correct[sampled_accept].mean() - sampled_correct.mean()))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def analyze(name: str, zero_path: Path, dose_pattern: str, strict_ids: set[str], seed: int) -> dict:
    zero = alpha0(zero_path)
    dose = displaced(dose_pattern)
    if set(zero) != set(dose):
        raise RuntimeError(f"ID mismatch for {name}")

    records = []
    for sid in sorted(zero):
        base = zero[sid]
        predictions = [base["prediction"]] + [dose[sid][a]["prediction"] for a in (.25, .5, .75, 1.0)]
        agreement_count = sum(pred == predictions[0] for pred in predictions[1:])
        records.append({
            "sample_id": sid,
            "correct": bool(base["correct"]),
            "strict_zero_overlap": sid in strict_ids,
            "alpha0_prediction": predictions[0],
            "alpha1_prediction": predictions[-1],
            "far_change": predictions[0] != predictions[-1],
            "sensitivity_score": 1.0 - agreement_count / 4.0,
            "prediction_path": predictions,
        })

    def population(rows: list[dict], offset: int) -> dict:
        correct = np.asarray([row["correct"] for row in rows], dtype=bool)
        accept = np.asarray([row["far_change"] for row in rows], dtype=bool)
        scores = [row["sensitivity_score"] for row in rows]
        labels = [row["correct"] for row in rows]
        base_accuracy = float(correct.mean())
        accepted_accuracy = float(correct[accept].mean()) if accept.any() else float("nan")
        base_error = 1.0 - base_accuracy
        accepted_error = 1.0 - accepted_accuracy
        return {
            "n": len(rows),
            "base_accuracy": base_accuracy,
            "far_change_coverage": float(accept.mean()),
            "far_change_n": int(accept.sum()),
            "far_change_accuracy": accepted_accuracy,
            "far_change_accuracy_minus_base": accepted_accuracy - base_accuracy,
            "delta_bootstrap_95ci": bootstrap_delta(correct, accept, seed + offset),
            "error_rate_relative_reduction": (base_error - accepted_error) / base_error if base_error else 0.0,
            "continuous_sensitivity_correctness_auc": auc(scores, labels),
            "invariant_n": int((~accept).sum()),
            "invariant_accuracy": float(correct[~accept].mean()) if (~accept).any() else float("nan"),
        }

    return {
        "model": name,
        "all": population(records, 0),
        "strict_zero_overlap": population([row for row in records if row["strict_zero_overlap"]], 1),
        "score_definition": "1 - mean_k 1[y(alpha_k)=y(alpha0)] for alpha_k in {.25,.5,.75,1}; far-change uses alpha1 only",
    }


def main() -> None:
    pointers = {row["sample_id"]: row for row in jsonl(POINTER)}
    if len(pointers) != 291:
        raise RuntimeError("pointer population is not 291")
    strict_ids = {sid for sid, row in pointers.items() if row["strict_zero_overlap"]}
    if len(strict_ids) != 194:
        raise RuntimeError("strict population is not 194")

    models = [
        analyze(
            "Qwen2.5-Omni-7B",
            ROOT / "outputs/qwen25_ter_replication_v1.jsonl",
            "outputs/qwen25_maximal_pointer_dose_v1_shard*.jsonl",
            strict_ids,
            SEED,
        ),
        analyze(
            "Qwen3-Omni-30B-A3B",
            ROOT / "outputs/qwen3_ter_large_v1.jsonl",
            "outputs/qwen3_maximal_pointer_dose_v1_shard*.jsonl",
            strict_ids,
            SEED + 100,
        ),
    ]
    result = {
        "status": "POST_HOC_HEADROOM_REQUIRES_PROSPECTIVE_EXTERNAL_CONFIRMATION",
        "name": "Speech-Time Evidence Sensitivity (STES)",
        "models": models,
        "bootstrap": {"repetitions": BOOTSTRAP, "seed": SEED},
        "input_sha256": {
            "pointer": sha(POINTER),
            "qwen25_alpha0": sha(ROOT / "outputs/qwen25_ter_replication_v1.jsonl"),
            "qwen3_alpha0": sha(ROOT / "outputs/qwen3_ter_large_v1.jsonl"),
        },
        "claim_boundary": (
            "This is a post-hoc reuse of frozen CMTB dose outputs. It establishes headroom only; "
            "the score is not promoted unless an outcome-unseen external far-reference condition passes a frozen gate."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Speech-Time Evidence Sensitivity: CMTB post-hoc headroom",
        "",
        f"Status: **{result['status']}**",
        "",
        "STES accepts the automatic-reference answer only when a farthest feasible equal-duration reference changes the free-decoded option. The continuous diagnostic is one minus agreement with the alpha-0 answer over four displaced references.",
        "",
        "| Model | Base acc. | Far-change coverage | Accepted acc. | Gain | Error reduction | Continuous AUC | Invariant acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        row = model["all"]
        lines.append(
            f"| {model['model']} | {row['base_accuracy']:.1%} | {row['far_change_n']}/{row['n']} ({row['far_change_coverage']:.1%}) | "
            f"{row['far_change_accuracy']:.1%} | {row['far_change_accuracy_minus_base']:+.1%} "
            f"[{row['delta_bootstrap_95ci'][0]:+.1%}, {row['delta_bootstrap_95ci'][1]:+.1%}] | "
            f"{row['error_rate_relative_reduction']:.1%} | {row['continuous_sensitivity_correctness_auc']:.3f} | {row['invariant_accuracy']:.1%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Correct alpha-0 answers are usually temporally responsive, whereas many errors remain unchanged even when the reference is moved to the far boundary. This supports a per-instance evidence-sensitivity certificate rather than another answer-fusion rule.",
        "",
        "## Boundary",
        "",
        result["claim_boundary"],
        "",
    ]
    REPORT.write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "models": models}, indent=2))


if __name__ == "__main__":
    main()

