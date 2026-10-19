#!/usr/bin/env python3
"""Normalize and analyze the frozen video-SALMONN-2 CMTB-291 experiment."""

from __future__ import annotations

import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
NORMALIZED = ROOT / "outputs/video_salmonn2_cmtb291_normalized_v1.jsonl"
ANALYSIS = ROOT / "outputs/video_salmonn2_cmtb291_analysis_v1.json"
REPORT = ROOT / "outputs/video_salmonn2_cmtb291_report_v1.md"
SHARDS = 4


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    # Match the already-frozen Dev-44 analysis convention: indices 249 and
    # 9749 for 10,000 bootstrap replicates.
    return values[min(len(values) - 1, max(0, int(q * len(values)) - 1))]


def bootstrap_delta(
    left: dict[str, float], right: dict[str, float], *, seed: int
) -> list[float]:
    ids = sorted(left)
    if set(ids) != set(right):
        raise RuntimeError("Bootstrap ID mismatch")
    rng = random.Random(seed)
    deltas = []
    for _ in range(10000):
        sampled = [rng.choice(ids) for _ in ids]
        deltas.append(sum(left[sid] - right[sid] for sid in sampled) / len(sampled))
    return [percentile(deltas, 0.025), percentile(deltas, 0.975)]


def paired(left: dict[str, int], right: dict[str, int]) -> dict:
    ids = sorted(left)
    if set(ids) != set(right):
        raise RuntimeError("Paired ID mismatch")
    left_only = sum(left[sid] and not right[sid] for sid in ids)
    right_only = sum(right[sid] and not left[sid] for sid in ids)
    return {
        "n": len(ids),
        "left_accuracy": sum(left.values()) / len(ids),
        "right_accuracy": sum(right.values()) / len(ids),
        "left_minus_right": sum(left[sid] - right[sid] for sid in ids) / len(ids),
        "right_wrong_to_left_correct": left_only,
        "right_correct_to_left_wrong": right_only,
        "exact_mcnemar_p": exact_p(left_only, right_only),
    }


def main() -> None:
    normalized = []
    for shard_id in range(SHARDS):
        mapping = jsonl(ROOT / f"outputs/video_salmonn2_cmtb291_v1_shard{shard_id}_map.jsonl")
        result_path = ROOT / f"outputs/video_salmonn2_cmtb291_output_v1_shard{shard_id}/test_results_rank0.json"
        raw = json.loads(result_path.read_text())
        if len(mapping) != len(raw):
            raise RuntimeError(f"Shard {shard_id} mapping/results mismatch: {len(mapping)} vs {len(raw)}")
        for meta, result in zip(mapping, raw):
            if result["ref"] != meta["answer"]:
                raise RuntimeError(f"Shard {shard_id} reference/order validation failed")
            match = re.search(r"\b([ABC])\b", result["pred"].upper())
            prediction = match.group(1) if match else None
            normalized.append(
                {
                    **meta,
                    "prediction": prediction,
                    "correct": prediction == meta["answer"],
                    "parse_status": "parsed" if prediction else "parse_failure",
                    "raw_output": result["pred"],
                    "shard_id": shard_id,
                    "model": "tsinghua-ee/video-SALMONN-2",
                    "decoding": {"do_sample": False, "seed": 2024},
                }
            )
    NORMALIZED.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized)
    )

    by_condition: dict[str, dict[str, int]] = defaultdict(dict)
    parse_failures = defaultdict(int)
    for row in normalized:
        if row["analysis_subset"] == "primary":
            by_condition[row["condition"]][row["sample_id"]] = int(row["correct"])
        parse_failures[row["condition"]] += row["parse_status"] == "parse_failure"
    raw = by_condition["Raw_AV"]
    video = by_condition["Video_only"]
    vad = by_condition["VAD_localized_clip"]
    strict_ids = sorted(set.intersection(*(set(by_condition[f"Matched_random_{i}"]) for i in range(3))))
    vad_strict = {sid: vad[sid] for sid in strict_ids}
    random_mean = {
        sid: sum(by_condition[f"Matched_random_{i}"][sid] for i in range(3)) / 3
        for sid in strict_ids
    }
    sensitivity = {
        row["sample_id"]: int(row["correct"])
        for row in normalized
        if row["condition"] == "Matched_random_sensitivity_frozen"
    }
    vad_sensitivity = {sid: vad[sid] for sid in sensitivity}
    vad_vs_raw = paired(vad, raw)
    vad_vs_video = paired(vad, video)
    vad_minus_random = sum(vad_strict[sid] - random_mean[sid] for sid in strict_ids) / len(strict_ids)
    result = {
        "scope": "Frozen video-SALMONN-2 CMTB-291 closing evaluation",
        "sample_count": len(raw),
        "strict_random_sample_count": len(strict_ids),
        "accuracy": {
            condition: sum(values.values()) / len(values)
            for condition, values in sorted(by_condition.items())
        },
        "parse_failures": dict(sorted(parse_failures.items())),
        "vad_vs_raw": {
            **vad_vs_raw,
            "video_cluster_bootstrap_95ci": bootstrap_delta(vad, raw, seed=20260820),
            "descriptive_threshold_10pp": vad_vs_raw["left_minus_right"] >= 0.10,
            "ci_above_zero": bootstrap_delta(vad, raw, seed=20260820)[0] > 0,
        },
        "vad_vs_video": {
            **vad_vs_video,
            "video_cluster_bootstrap_95ci": bootstrap_delta(vad, video, seed=20260821),
        },
        "vad_vs_matched_random_primary": {
            "n": len(strict_ids),
            "vad_accuracy": sum(vad_strict.values()) / len(strict_ids),
            "matched_random_mean_accuracy": sum(random_mean.values()) / len(strict_ids),
            "vad_minus_matched_random_mean": vad_minus_random,
            "video_cluster_bootstrap_95ci": bootstrap_delta(vad_strict, random_mean, seed=20260822),
            "descriptive_threshold_15pp": vad_minus_random >= 0.15,
            "ci_above_zero": bootstrap_delta(vad_strict, random_mean, seed=20260822)[0] > 0,
        },
        "vad_vs_random_sensitivity_nonprimary": {
            **paired(vad_sensitivity, sensitivity),
            "video_cluster_bootstrap_95ci": bootstrap_delta(
                vad_sensitivity, sensitivity, seed=20260824
            ),
            "warning": "Strict same-duration non-overlap was infeasible; descriptive sensitivity only",
        },
        "reporting_thresholds": {
            "vad_minus_raw_at_least_10pp": vad_vs_raw["left_minus_right"] >= 0.10,
            "vad_minus_random_at_least_15pp": vad_minus_random >= 0.15,
            "paired_cluster_cis_above_zero": (
                bootstrap_delta(vad, raw, seed=20260820)[0] > 0
                and bootstrap_delta(vad_strict, random_mean, seed=20260822)[0] > 0
            ),
        },
        "reporting_threshold_provenance": "RETROSPECTIVE_NOT_PREREGISTERED",
        "sensitivity_rows": len(sensitivity),
        "analysis_policy": "Full 291 for Raw/Video/VAD; strict non-overlap subset only for primary random control",
    }
    ANALYSIS.write_text(json.dumps(result, indent=2) + "\n")
    thresholds = result["reporting_thresholds"]
    REPORT.write_text(
        "\n".join(
            [
                "# video-SALMONN-2 CMTB-291 Closing Evaluation",
                "",
                f"- Raw AV: {result['accuracy']['Raw_AV']:.1%} (n=291)",
                f"- Video-only: {result['accuracy']['Video_only']:.1%} (n=291)",
                f"- Speech-timed video-only interval: {result['accuracy']['VAD_localized_clip']:.1%} (n=291)",
                f"- VAD - Raw: {vad_vs_raw['left_minus_right']:+.1%}; 95% cluster CI {result['vad_vs_raw']['video_cluster_bootstrap_95ci']}",
                f"- Strict matched-random mean: {result['vad_vs_matched_random_primary']['matched_random_mean_accuracy']:.1%} (n={len(strict_ids)}, 3 replicates)",
                f"- VAD - matched-random: {vad_minus_random:+.1%}; 95% cluster CI {result['vad_vs_matched_random_primary']['video_cluster_bootstrap_95ci']}",
                "",
                "## Retrospective reporting thresholds (not preregistered)",
                "",
                f"- VAD - Raw >= 10%p: {thresholds['vad_minus_raw_at_least_10pp']}",
                f"- VAD - random >= 15%p: {thresholds['vad_minus_random_at_least_15pp']}",
                f"- Both cluster CIs above zero: {thresholds['paired_cluster_cis_above_zero']}",
                "",
                "Evaluation inputs and inference settings were frozen before predictions. The reporting thresholds are retrospective and do not affect estimates, intervals, or tests.",
            ]
        )
        + "\n"
    )
    print(ANALYSIS.read_text())


if __name__ == "__main__":
    main()
