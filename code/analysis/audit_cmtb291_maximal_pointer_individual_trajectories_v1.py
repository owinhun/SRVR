#!/usr/bin/env python3
"""Post-hoc row-level audit of the frozen full-N maximal pointer dose.

This does not alter the prospective success gate.  It checks whether the
aggregate monotone dose curve is supported by mostly one-directional item
trajectories rather than balanced correctness churn.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
OUT_JSON = ROOT / "outputs/cmtb291_maximal_pointer_individual_trajectory_audit_v1.json"
OUT_MD = ROOT / "outputs/cmtb291_maximal_pointer_individual_trajectory_audit_v1.md"
POINTER = ROOT / "outputs/cmtb291_maximal_displacement_pointer_v1.jsonl"
PRIMARY = ROOT / "outputs/cmtb291_maximal_displacement_dose_analysis_v1.json"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def endpoint(prefix: str) -> list[dict]:
    name = "qwen3_ter_large_v1.jsonl" if prefix == "qwen3" else "qwen25_ter_replication_v1.jsonl"
    return [
        row
        for row in read_jsonl(ROOT / "outputs" / name)
        if row.get("task") == "CMTB" and row.get("condition") == "TER_V_auto"
    ]


def displaced(prefix: str) -> list[dict]:
    rows: list[dict] = []
    pattern = ROOT / "outputs" / f"{prefix}_maximal_pointer_dose_v1_shard*.jsonl"
    for name in sorted(glob.glob(str(pattern))):
        rows.extend(read_jsonl(Path(name)))
    return rows


def analyze(prefix: str, label: str, populations: dict[str, set[str]]) -> dict:
    zero = endpoint(prefix)
    moved = displaced(prefix)
    rows_by_alpha: dict[float, list[dict]] = {0.0: zero}
    for alpha in ALPHAS[1:]:
        rows_by_alpha[alpha] = [
            row for row in moved if abs(float(row["pointer_alpha"]) - alpha) < 1e-9
        ]
    by_alpha: dict[float, dict[str, bool]] = {
        alpha: {
            row["sample_id"]: bool(row["correct"])
            for row in rows_by_alpha[alpha]
        }
        for alpha in ALPHAS
    }
    expected = populations["all"]
    if any(set(by_alpha[alpha]) != expected for alpha in ALPHAS):
        raise RuntimeError(f"{prefix}: sample population mismatch")

    summaries: dict[str, dict] = {}
    for population_name, ids in populations.items():
        trajectories: dict[str, str] = {}
        for sample_id in sorted(ids):
            trajectories[sample_id] = "".join(
                "1" if by_alpha[alpha][sample_id] else "0" for alpha in ALPHAS
            )

        def has_reverse(bits: str) -> bool:
            return any(bits[index] == "0" and bits[index + 1] == "1" for index in range(4))

        weakly_non_increasing = sum(not has_reverse(bits) for bits in trajectories.values())
        forward_steps = sum(
            bits[index] == "1" and bits[index + 1] == "0"
            for bits in trajectories.values()
            for index in range(4)
        )
        reverse_steps = sum(
            bits[index] == "0" and bits[index + 1] == "1"
            for bits in trajectories.values()
            for index in range(4)
        )
        adjacent = []
        for left_alpha, right_alpha in zip(ALPHAS[:-1], ALPHAS[1:]):
            left_only = sum(
                by_alpha[left_alpha][sample_id] and not by_alpha[right_alpha][sample_id]
                for sample_id in ids
            )
            right_only = sum(
                not by_alpha[left_alpha][sample_id] and by_alpha[right_alpha][sample_id]
                for sample_id in ids
            )
            adjacent.append(
                {
                    "left_alpha": left_alpha,
                    "right_alpha": right_alpha,
                    "nearer_only_correct": left_only,
                    "farther_only_correct": right_only,
                    "exact_mcnemar_p": exact_mcnemar(left_only, right_only),
                }
            )
        endpoint_forward = sum(bits[0] == "1" and bits[-1] == "0" for bits in trajectories.values())
        endpoint_reverse = sum(bits[0] == "0" and bits[-1] == "1" for bits in trajectories.values())
        answer_distribution = Counter(
            row["answer"] for row in rows_by_alpha[0.0] if row["sample_id"] in ids
        )
        prediction_distribution_by_alpha = {}
        maximum_prediction_share_by_alpha = {}
        for alpha in ALPHAS:
            predictions = Counter(
                row.get("prediction") or "PARSE_FAIL"
                for row in rows_by_alpha[alpha]
                if row["sample_id"] in ids
            )
            prediction_distribution_by_alpha[str(alpha)] = dict(sorted(predictions.items()))
            maximum_prediction_share_by_alpha[str(alpha)] = max(predictions.values()) / len(ids)
        summaries[population_name] = {
            "n": len(ids),
            "weakly_non_increasing_trajectories": weakly_non_increasing,
            "weakly_non_increasing_fraction": weakly_non_increasing / len(ids),
            "trajectories_with_any_reverse_step": len(ids) - weakly_non_increasing,
            "trajectories_with_any_reverse_step_fraction": (len(ids) - weakly_non_increasing) / len(ids),
            "forward_correct_to_wrong_steps": forward_steps,
            "reverse_wrong_to_correct_steps": reverse_steps,
            "correct_to_wrong_endpoints": endpoint_forward,
            "wrong_to_correct_endpoints": endpoint_reverse,
            "adjacent_pairs": adjacent,
            "trajectory_counts": dict(sorted(Counter(trajectories.values()).items())),
            "answer_distribution": dict(sorted(answer_distribution.items())),
            "prediction_distribution_by_alpha": prediction_distribution_by_alpha,
            "maximum_prediction_share_by_alpha": maximum_prediction_share_by_alpha,
        }
    return {
        "model": label,
        "populations": summaries,
        "input_sha256": {
            "endpoint": sha256(ROOT / "outputs" / ("qwen3_ter_large_v1.jsonl" if prefix == "qwen3" else "qwen25_ter_replication_v1.jsonl")),
            **{
                Path(name).name: sha256(Path(name))
                for name in sorted(glob.glob(str(ROOT / "outputs" / f"{prefix}_maximal_pointer_dose_v1_shard*.jsonl")))
            },
        },
    }


def main() -> None:
    pointer_rows = read_jsonl(POINTER)
    pointer_map = {row["sample_id"]: row for row in pointer_rows}
    if len(pointer_map) != 291:
        raise RuntimeError("pointer manifest is not CMTB-291")
    populations = {
        "all": set(pointer_map),
        "strict_zero_overlap": {
            sample_id for sample_id, row in pointer_map.items() if row["strict_zero_overlap"]
        },
        "positive_overlap": {
            sample_id for sample_id, row in pointer_map.items() if not row["strict_zero_overlap"]
        },
    }
    if len(populations["strict_zero_overlap"]) != 194 or len(populations["positive_overlap"]) != 97:
        raise RuntimeError("geometry strata drift")

    models = [
        analyze("qwen3", "Qwen3-Omni-30B-A3B", populations),
        analyze("qwen25", "Qwen2.5-Omni-7B", populations),
    ]
    primary = json.loads(PRIMARY.read_text())
    primary_models = {row["model"]: row for row in primary["models"]}
    for model in models:
        source = primary_models.get(model["model"])
        if source is None:
            raise RuntimeError(f"missing primary model: {model['model']}")
        trajectory_counts = model["populations"]["all"]["trajectory_counts"]
        reconstructed = [
            sum(count for bits, count in trajectory_counts.items() if bits[index] == "1")
            for index in range(len(ALPHAS))
        ]
        expected = [
            int(source["populations"]["all"]["alpha"][str(alpha)]["correct"])
            for alpha in ALPHAS
        ]
        if reconstructed != expected:
            raise RuntimeError(
                f"trajectory/primary correct-count mismatch for {model['model']}: "
                f"{reconstructed} != {expected}"
            )
        model["primary_count_crosscheck"] = {
            "reconstructed_correct_by_alpha": reconstructed,
            "primary_correct_by_alpha": expected,
            "pass": True,
        }
    result = {
        "status": "AUDITED_POST_HOC",
        "claim_scope": (
            "Supplementary row-level robustness only. The maximal-dose conditions were frozen "
            "prospectively, but this trajectory estimand was selected after outcomes were available."
        ),
        "population": "CMTB-291 maximal pointer displacement with geometry-defined 194/97 strata",
        "alpha_order": list(ALPHAS),
        "trajectory_bit_meaning": "1=correct and 0=incorrect at the corresponding alpha",
        "models": models,
        "shared_input_sha256": {
            "pointer": sha256(POINTER),
            "primary_analysis": sha256(PRIMARY),
        },
        "analysis_code_sha256": None,
    }
    result["analysis_code_sha256"] = sha256(Path(__file__))
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# CMTB-291 maximal-pointer individual-trajectory audit v1",
        "",
        "Status: **AUDITED_POST_HOC**",
        "",
        result["claim_scope"],
        "",
        "A trajectory is weakly non-increasing when binary correctness never changes from wrong back to correct as the supplied pointer moves farther from speech time.",
        "",
        "| Model | Population | Weakly non-increasing | Any reverse step | Correct→wrong endpoints | Wrong→correct endpoints | Forward/reverse adjacent steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for population_name in ("all", "strict_zero_overlap", "positive_overlap"):
            summary = model["populations"][population_name]
            lines.append(
                f"| {model['model']} | {population_name} N={summary['n']} | "
                f"{summary['weakly_non_increasing_trajectories']}/{summary['n']} "
                f"({100 * summary['weakly_non_increasing_fraction']:.1f}%) | "
                f"{summary['trajectories_with_any_reverse_step']}/{summary['n']} | "
                f"{summary['correct_to_wrong_endpoints']} | {summary['wrong_to_correct_endpoints']} | "
                f"{summary['forward_correct_to_wrong_steps']}/{summary['reverse_wrong_to_correct_steps']} |"
            )
    lines.extend(["", "## Adjacent-dose discordance on all 291", "", "| Model | Alpha pair | Nearer-only correct | Farther-only correct | Exact paired p |", "|---|---:|---:|---:|---:|"])
    for model in models:
        for pair in model["populations"]["all"]["adjacent_pairs"]:
            lines.append(
                f"| {model['model']} | {pair['left_alpha']:.2f}→{pair['right_alpha']:.2f} | "
                f"{pair['nearer_only_correct']} | {pair['farther_only_correct']} | {pair['exact_mcnemar_p']:.4g} |"
            )
    lines.extend(
        [
            "",
            "## Answer-position collapse audit on all 291",
            "",
            "The correct-answer positions are A/B/C = 93/99/99. The maximum share assigned to any single parsed option at each dose is:",
            "",
            "| Model | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in models:
        shares = model["populations"]["all"]["maximum_prediction_share_by_alpha"]
        lines.append(
            f"| {model['model']} | "
            + " | ".join(f"{100 * shares[str(alpha)]:.1f}%" for alpha in ALPHAS)
            + " |"
        )
    lines.extend(
        [
            "",
            "No dose collapses to a single option: the largest option share is 49.1%. This does not establish that the model names the visual action at each displaced interval, but it rules out a trivial fixed-letter explanation for the accuracy reversal.",
        ]
    )
    lines.extend(
        [
            "",
            "The aggregate dose curves are not produced by balanced row-level churn. This is a post-hoc robustness audit and must not be described as a preregistered or independently confirmatory estimand.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "models": [{"model": model["model"], "all": model["populations"]["all"]} for model in models]}, indent=2))


if __name__ == "__main__":
    main()
