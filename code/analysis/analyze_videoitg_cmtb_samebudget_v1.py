#!/usr/bin/env python3
"""Audit and analyze frozen same-budget InternVL CMTB predictions."""

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path


CONDITIONS = ("uniform_32", "videoitg_32", "srvr_32")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap(correct_a, correct_b, seed=20260812, repetitions=20000):
    rng = random.Random(seed)
    n = len(correct_a)
    effects = []
    for _ in range(repetitions):
        indices = [rng.randrange(n) for _ in range(n)]
        effects.append(
            sum(correct_b[i] - correct_a[i] for i in indices) / n
        )
    effects.sort()
    return [effects[int(0.025 * repetitions)], effects[int(0.975 * repetitions)]]


def compare(rows, first, second):
    a = [int(row[first]["correct"]) for row in rows]
    b = [int(row[second]["correct"]) for row in rows]
    wrong_correct = sum(x == 0 and y == 1 for x, y in zip(a, b))
    correct_wrong = sum(x == 1 and y == 0 for x, y in zip(a, b))
    return {
        "first": first,
        "second": second,
        "delta_accuracy": sum(y - x for x, y in zip(a, b)) / len(a),
        "delta_correct_count": sum(b) - sum(a),
        "wrong_to_correct": wrong_correct,
        "correct_to_wrong": correct_wrong,
        "mcnemar_exact_p": exact_mcnemar(wrong_correct, correct_wrong),
        "paired_bootstrap_95_ci": paired_bootstrap(a, b),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument("--dev-gate", action="store_true")
    parser.add_argument("--model-label", default="InternVL2.5-8B")
    parser.add_argument("--sequential-secondary", action="store_true")
    args = parser.parse_args()

    manifest_rows = [json.loads(line) for line in open(args.manifest) if line.strip()]
    if len(manifest_rows) != args.population:
        raise ValueError(f"manifest has {len(manifest_rows)}, expected {args.population}")
    answers = {row["sample_id"]: row["answer_evaluation_only"] for row in manifest_rows}
    expected_ids = set(answers)
    raw = [json.loads(line) for line in open(args.predictions) if line.strip()]
    selector = [json.loads(line) for line in open(args.selector) if line.strip()]
    selector_by_id = {row["sample_id"]: row for row in selector}

    grouped = {sample_id: {} for sample_id in expected_ids}
    duplicate_keys = []
    for row in raw:
        key = (row["sample_id"], row["condition"])
        if row["condition"] in grouped.get(row["sample_id"], {}):
            duplicate_keys.append(key)
        if row["sample_id"] in grouped:
            grouped[row["sample_id"]][row["condition"]] = row

    complete = all(set(grouped[sample_id]) == set(CONDITIONS) for sample_id in expected_ids)
    parse_ok = all(
        row.get("parse_status") == "parsed" and row.get("prediction") in "ABC"
        for conditions in grouped.values()
        for row in conditions.values()
    )
    answer_blind = all(
        row.get("answer_used") is False
        for conditions in grouped.values()
        for row in conditions.values()
    )
    selector_answer_blind = all(row.get("answer_used") is False for row in selector)
    budget_match = True
    valid_indices = True
    for sample_id, conditions in grouped.items():
        if set(conditions) != set(CONDITIONS):
            budget_match = False
            continue
        lengths = [len(conditions[name]["selected_source_indices"]) for name in CONDITIONS]
        budgets = [conditions[name]["actual_common_frame_budget"] for name in CONDITIONS]
        budget_match &= len(set(lengths)) == 1 and len(set(budgets)) == 1 and lengths[0] == budgets[0]
        valid_indices &= all(
            all(0 <= index < row["source_total_frames"] for index in row["selected_source_indices"])
            for row in conditions.values()
        )

    enriched = []
    if complete:
        for sample_id in sorted(expected_ids):
            item = {}
            for condition in CONDITIONS:
                row = dict(grouped[sample_id][condition])
                row["answer"] = answers[sample_id]
                row["correct"] = row["prediction"] == answers[sample_id]
                item[condition] = row
            enriched.append(item)

    metrics = {}
    collapse_ok = False
    if enriched:
        for condition in CONDITIONS:
            predictions = [row[condition]["prediction"] for row in enriched]
            counts = collections.Counter(predictions)
            correct = sum(row[condition]["correct"] for row in enriched)
            metrics[condition] = {
                "correct": correct,
                "n": len(enriched),
                "accuracy": correct / len(enriched),
                "prediction_counts": dict(sorted(counts.items())),
                "max_prediction_fraction": max(counts.values()) / len(enriched),
            }
        collapse_ok = all(
            len(metrics[c]["prediction_counts"]) >= 2
            and metrics[c]["max_prediction_fraction"] < 0.90
            for c in CONDITIONS
        )

    hard_checks = {
        "exact_prediction_rows": len(raw) == args.population * 3,
        "exact_selector_rows": len(selector) == args.population,
        "selector_population_matches": set(selector_by_id) == expected_ids,
        "no_duplicate_prediction_keys": not duplicate_keys,
        "complete_three_conditions": complete,
        "all_predictions_parsed": parse_ok,
        "prediction_runner_answer_blind": answer_blind,
        "selector_answer_blind": selector_answer_blind,
        "within_item_budget_matched": budget_match,
        "all_frame_indices_valid": valid_indices,
        "no_option_position_collapse": collapse_ok,
    }
    technical_pass = all(hard_checks.values())
    comparisons = {}
    if enriched:
        comparisons = {
            "videoitg_vs_uniform": compare(enriched, "uniform_32", "videoitg_32"),
            "srvr_vs_uniform": compare(enriched, "uniform_32", "srvr_32"),
            "srvr_vs_videoitg": compare(enriched, "videoitg_32", "srvr_32"),
        }
    promote = None
    if args.dev_gate:
        promote = technical_pass and comparisons["videoitg_vs_uniform"]["delta_correct_count"] >= 3

    report = {
        "status": "PASS" if technical_pass else "FAIL_TECHNICAL_AUDIT",
        "population": args.population,
        "model_label": args.model_label,
        "sequential_secondary": args.sequential_secondary,
        "dev_gate": args.dev_gate,
        "promote_to_full": promote,
        "promotion_rule": "technical audit PASS and VideoITG minus Uniform >= 3 correct items"
        if args.dev_gate
        else None,
        "hard_checks": hard_checks,
        "metrics": metrics,
        "comparisons": comparisons,
        "input_hashes": {
            "manifest": sha256(args.manifest),
            "predictions": sha256(args.predictions),
            "selector": sha256(args.selector),
        },
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"# VideoITG same-budget CMTB report — {args.model_label}",
        "",
        f"- Population: {args.population}",
        f"- Technical audit: **{report['status']}**",
    ]
    if args.sequential_secondary:
        lines.append("- Scope: **sequential secondary Dev diagnostic; cannot override the primary gate**")
    if args.dev_gate:
        lines.append(f"- Promote to CMTB-291: **{promote}**")
    lines += ["", "## Accuracy", "", "| Condition | Correct | Accuracy | Predictions |", "|---|---:|---:|---|"]
    for condition in CONDITIONS:
        value = metrics.get(condition, {})
        lines.append(
            f"| {condition} | {value.get('correct', 'NA')}/{value.get('n', 'NA')} | "
            f"{100*value.get('accuracy', float('nan')):.1f}% | {value.get('prediction_counts', {})} |"
        )
    lines += ["", "## Paired comparisons", ""]
    for name, value in comparisons.items():
        lines.append(
            f"- {name}: {100*value['delta_accuracy']:+.1f} pp; "
            f"W→C={value['wrong_to_correct']}, C→W={value['correct_to_wrong']}, "
            f"exact McNemar p={value['mcnemar_exact_p']:.6g}, "
            f"95% CI=[{100*value['paired_bootstrap_95_ci'][0]:.1f}, "
            f"{100*value['paired_bootstrap_95_ci'][1]:.1f}] pp."
        )
    lines += ["", "## Hard checks", ""]
    lines.extend(f"- {key}: {value}" for key, value in hard_checks.items())
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
