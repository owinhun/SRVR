#!/usr/bin/env python3
"""Merge, audit, and gate the frozen S2FS InternVL CMTB Dev-48 run."""

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path


CONDITIONS = (
    "full_uniform_8",
    "srvr_uniform_8",
    "visual_coverage_8",
    "semantic_attract_8",
    "signed_repulsion_8",
    "shuffled_repulsion_8",
)
MODEL = "OpenGVLab/InternVL2_5-8B"
REVISION = "e9e4c0dc1db56bfab10458671519b7fa3dd29463"
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPETITIONS = 20000


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap(first, second):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(first)
    effects = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        indices = [rng.randrange(n) for _ in range(n)]
        effects.append(sum(second[i] - first[i] for i in indices) / n)
    effects.sort()
    return [
        effects[int(0.025 * BOOTSTRAP_REPETITIONS)],
        effects[int(0.975 * BOOTSTRAP_REPETITIONS)],
    ]


def comparison(enriched, first, second):
    a = [int(row[first]["correct"]) for row in enriched]
    b = [int(row[second]["correct"]) for row in enriched]
    wrong_correct = sum(x == 0 and y == 1 for x, y in zip(a, b))
    correct_wrong = sum(x == 1 and y == 0 for x, y in zip(a, b))
    return {
        "first": first,
        "second": second,
        "delta_correct_count": sum(b) - sum(a),
        "delta_accuracy": (sum(b) - sum(a)) / len(a),
        "wrong_to_correct": wrong_correct,
        "correct_to_wrong": correct_wrong,
        "mcnemar_exact_p": exact_mcnemar(wrong_correct, correct_wrong),
        "paired_bootstrap_95_ci": paired_bootstrap(a, b),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--merged-output", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in open(args.manifest) if line.strip()]
    selector_rows = [json.loads(line) for line in open(args.selector) if line.strip()]
    order = [row["sample_id"] for row in manifest]
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    selector = {row["sample_id"]: row for row in selector_rows}
    expected = {(sample_id, condition) for sample_id in order for condition in CONDITIONS}
    hashes = {
        "manifest": sha256(args.manifest),
        "selector": sha256(args.selector),
        "protocol": sha256(args.protocol),
        "runner": sha256(args.runner),
        "shards": {path: sha256(path) for path in args.shards},
    }

    raw = []
    for shard in args.shards:
        raw.extend(json.loads(line) for line in open(shard) if line.strip())
    by_key = {}
    duplicates = []
    for row in raw:
        key = (row.get("sample_id"), row.get("condition"))
        if key in by_key:
            duplicates.append(key)
        by_key[key] = row

    index_contract = True
    prompt_contract = True
    provenance_contract = True
    inference_contract = True
    decoding_contracts = set()
    parsed = True
    for key, row in by_key.items():
        sample_id, condition = key
        if sample_id not in selector or condition not in CONDITIONS:
            index_contract = False
            continue
        indices = row.get("selected_source_indices", [])
        source_frames = row.get("source_total_frames", 0)
        index_contract &= (
            indices == selector[sample_id]["indices_by_condition"][condition]
            and len(indices) == 8
            and len(set(indices)) == 8
            and indices == sorted(indices)
            and all(0 <= value < source_frames for value in indices)
        )
        question = manifest_by_id[sample_id]["selector_prompt"]
        prompt = "".join(f"Frame{i + 1}: <image>\n" for i in range(8)) + question + "\nRespond with only the letter A, B, or C."
        prompt_contract &= (
            row.get("question_sha256") == text_sha256(question)
            and row.get("prompt_sha256") == text_sha256(prompt)
        )
        provenance_contract &= (
            row.get("manifest_sha256") == hashes["manifest"]
            and row.get("selector_sha256") == hashes["selector"]
            and row.get("protocol_sha256") == hashes["protocol"]
            and row.get("runner_sha256") == hashes["runner"]
        )
        decoding = row.get("decoding", {})
        decoding_contracts.add(json.dumps(decoding, sort_keys=True))
        inference_contract &= (
            row.get("model_id") == MODEL
            and row.get("model_revision") == REVISION
            and row.get("audio") == "off"
            and row.get("answer_used") is False
            and row.get("frame_budget") == 8
            and decoding.get("num_beams") == 1
            and decoding.get("max_new_tokens") == 8
            and decoding.get("do_sample") is False
        )
        parsed &= row.get("parse_status") == "parsed" and row.get("prediction") in "ABC"

    prediction_counts = {
        condition: collections.Counter(
            row["prediction"] for (sample_id, name), row in by_key.items() if name == condition
        )
        for condition in CONDITIONS
    }
    no_collapse = all(
        counts and max(counts.values()) / 48 < 0.75 for counts in prediction_counts.values()
    )
    hard_checks = {
        "manifest_48_unique": len(manifest) == len(set(order)) == 48,
        "selector_48_unique_and_matching": len(selector_rows) == len(selector) == 48 and set(selector) == set(order),
        "exact_288_prediction_rows": len(raw) == 288,
        "prediction_keys_exact": set(by_key) == expected,
        "no_duplicate_prediction_keys": not duplicates,
        "all_predictions_parsed": parsed,
        "answer_blind_audio_off_frozen_inference": inference_contract,
        "identical_runtime_decoding_contract": len(decoding_contracts) == 1,
        "indices_exactly_match_selector_and_are_valid_k8": index_contract,
        "question_and_prompt_hashes_exact": prompt_contract,
        "provenance_hashes_exact": provenance_contract,
        "no_option_position_collapse_below_0_75": no_collapse,
    }
    technical_pass = all(hard_checks.values())

    merged = Path(args.merged_output)
    if technical_pass:
        with merged.open("w") as sink:
            for sample_id in order:
                for condition in CONDITIONS:
                    sink.write(json.dumps(by_key[(sample_id, condition)]) + "\n")

    metrics = {}
    comparisons = {}
    promotion_checks = {}
    promote = False
    if technical_pass:
        enriched = []
        for sample_id in sorted(order):
            answer = manifest_by_id[sample_id]["answer_evaluation_only"]
            item = {}
            for condition in CONDITIONS:
                row = dict(by_key[(sample_id, condition)])
                row["correct"] = row["prediction"] == answer
                item[condition] = row
            enriched.append(item)
        for condition in CONDITIONS:
            correct = sum(row[condition]["correct"] for row in enriched)
            counts = prediction_counts[condition]
            metrics[condition] = {
                "correct": correct,
                "n": 48,
                "accuracy": correct / 48,
                "prediction_counts": dict(sorted(counts.items())),
                "max_prediction_fraction": max(counts.values()) / 48,
            }
        comparisons = {
            "signed_vs_srvr_uniform": comparison(enriched, "srvr_uniform_8", "signed_repulsion_8"),
            "signed_vs_semantic_attract": comparison(enriched, "semantic_attract_8", "signed_repulsion_8"),
            "signed_vs_visual_coverage": comparison(enriched, "visual_coverage_8", "signed_repulsion_8"),
            "signed_vs_shuffled_repulsion": comparison(enriched, "shuffled_repulsion_8", "signed_repulsion_8"),
        }
        first = comparisons["signed_vs_srvr_uniform"]
        second = comparisons["signed_vs_semantic_attract"]
        promotion_checks = {
            "signed_minus_srvr_at_least_3_of_48": first["delta_correct_count"] >= 3,
            "signed_minus_semantic_attract_at_least_3_of_48": second["delta_correct_count"] >= 3,
            "signed_at_least_38_of_48": metrics["signed_repulsion_8"]["correct"] >= 38,
            "signed_vs_srvr_ci_lower_not_below_minus_1_of_48": first["paired_bootstrap_95_ci"][0] >= -1 / 48,
            "signed_vs_attract_ci_lower_not_below_minus_1_of_48": second["paired_bootstrap_95_ci"][0] >= -1 / 48,
        }
        promote = all(promotion_checks.values())

    report = {
        "status": "PROMOTE" if promote else ("STOP_S2FS" if technical_pass else "FAIL_TECHNICAL_AUDIT"),
        "scope": "sequential method-development diagnostic on frozen CMTB Dev-48",
        "technical_pass": technical_pass,
        "promote_to_full291": promote,
        "hard_checks": hard_checks,
        "promotion_checks": promotion_checks,
        "metrics": metrics,
        "comparisons": comparisons,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS},
        "audit_note": (
            "InternVL model.chat deterministically appends eos_token_id to the caller's generation "
            "dictionary; the audit requires the three frozen fields and exact row-wise equality of "
            "the resulting runtime dictionary."
        ),
        "input_hashes": hashes,
        "merged_output": str(merged) if technical_pass else None,
        "merged_output_sha256": sha256(merged) if technical_pass else None,
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# S²FS InternVL CMTB Dev-48 frozen decision report",
        "",
        f"- Status: **{report['status']}**",
        f"- Technical audit: **{'PASS' if technical_pass else 'FAIL'}**",
        f"- Promote to Full-291: **{promote}**",
        "- Scope: sequential method-development diagnostic; not independent confirmation.",
        "",
        "## Accuracy",
        "",
        "| Condition | Correct | Accuracy | Predictions |",
        "|---|---:|---:|---|",
    ]
    for condition in CONDITIONS:
        value = metrics.get(condition, {})
        lines.append(
            f"| {condition} | {value.get('correct', 'NA')}/{value.get('n', 'NA')} | "
            f"{100 * value.get('accuracy', float('nan')):.1f}% | {value.get('prediction_counts', {})} |"
        )
    lines += ["", "## Paired comparisons", ""]
    for name, value in comparisons.items():
        lines.append(
            f"- {name}: {value['delta_correct_count']:+d}/48 ({100*value['delta_accuracy']:+.1f} pp), "
            f"W→C={value['wrong_to_correct']}, C→W={value['correct_to_wrong']}, "
            f"McNemar p={value['mcnemar_exact_p']:.6g}, 95% CI "
            f"[{100*value['paired_bootstrap_95_ci'][0]:.1f}, {100*value['paired_bootstrap_95_ci'][1]:.1f}] pp."
        )
    lines += ["", "## Frozen promotion checks", ""]
    lines.extend(f"- {key}: {value}" for key, value in promotion_checks.items())
    lines += ["", "## Technical checks", ""]
    lines.extend(f"- {key}: {value}" for key, value in hard_checks.items())
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
