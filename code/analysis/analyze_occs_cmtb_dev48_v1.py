#!/usr/bin/env python3
"""Merge, audit, and gate the frozen OCCS InternVL CMTB Dev-48 run."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path


NEW_CONDITIONS = (
    "option_maxsim_8",
    "option_contrast_8",
    "shuffled_option_contrast_8",
)
LOCKED_CONDITIONS = ("srvr_uniform_8", "visual_coverage_8")
ALL_CONDITIONS = LOCKED_CONDITIONS + NEW_CONDITIONS
MODEL = "OpenGVLab/InternVL2_5-8B"
REVISION = "e9e4c0dc1db56bfab10458671519b7fa3dd29463"
LOCKED_SELECTOR_SHA256 = "3e38931f2697296166fb08561b4860ec9c15e58a3b02b711e07fd68280d69772"
LOCKED_PREDICTIONS_SHA256 = "f3da9e9e59426acb4af176d51f43de182ca8b3eb9df5b7af94091b30f3bc9895"
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPETITIONS = 20000
OPTION_PATTERN = re.compile(
    r"^(?P<stem>.*?)\s*\(A\)\s*(?P<a>.*?)\s*\(B\)\s*(?P<b>.*?)\s*\(C\)\s*(?P<c>.*?)\s*$",
    re.DOTALL,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_options(prompt: str) -> tuple[str, str, str]:
    match = OPTION_PATTERN.fullmatch(prompt)
    if match is None:
        raise ValueError(f"strict A/B/C parse failed: {prompt!r}")
    options = tuple(match.group(name).strip() for name in ("a", "b", "c"))
    if len(options) != 3 or any(not option for option in options):
        raise ValueError("invalid option parse")
    return options


def exact_mcnemar(wrong_correct: int, correct_wrong: int) -> float:
    discordant = wrong_correct + correct_wrong
    if discordant == 0:
        return 1.0
    tail_limit = min(wrong_correct, correct_wrong)
    tail = sum(math.comb(discordant, value) for value in range(tail_limit + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_bootstrap(first: list[int], second: list[int]) -> list[float]:
    rng = random.Random(BOOTSTRAP_SEED)
    effects = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = [rng.randrange(len(first)) for _ in first]
        effects.append(sum(second[index] - first[index] for index in sampled) / len(first))
    effects.sort()
    return [
        effects[int(0.025 * BOOTSTRAP_REPETITIONS)],
        effects[int(0.975 * BOOTSTRAP_REPETITIONS)],
    ]


def comparison(enriched: list[dict], first: str, second: str) -> dict:
    first_correct = [int(row[first]["correct"]) for row in enriched]
    second_correct = [int(row[second]["correct"]) for row in enriched]
    wrong_correct = sum(a == 0 and b == 1 for a, b in zip(first_correct, second_correct))
    correct_wrong = sum(a == 1 and b == 0 for a, b in zip(first_correct, second_correct))
    delta = sum(second_correct) - sum(first_correct)
    return {
        "first": first,
        "second": second,
        "delta_correct_count": delta,
        "delta_accuracy": delta / len(first_correct),
        "wrong_to_correct": wrong_correct,
        "correct_to_wrong": correct_wrong,
        "mcnemar_exact_p": exact_mcnemar(wrong_correct, correct_wrong),
        "paired_bootstrap_95_ci": paired_bootstrap(first_correct, second_correct),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vad", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--locked-selector", required=True)
    parser.add_argument("--locked-predictions", required=True)
    parser.add_argument("--new-shards", nargs="+", required=True)
    parser.add_argument("--merged-output", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    selector_rows = read_jsonl(args.selector)
    locked_selector_rows = read_jsonl(args.locked_selector)
    locked_prediction_rows = read_jsonl(args.locked_predictions)
    new_rows = [row for path in args.new_shards for row in read_jsonl(path)]

    order = [row["sample_id"] for row in manifest]
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    selector = {row["sample_id"]: row for row in selector_rows}
    locked_selector = {row["sample_id"]: row for row in locked_selector_rows}
    expected_new = {(sample_id, condition) for sample_id in order for condition in NEW_CONDITIONS}
    expected_locked = {(sample_id, condition) for sample_id in order for condition in LOCKED_CONDITIONS}

    new_by_key: dict[tuple[str, str], dict] = {}
    duplicate_new = []
    for row in new_rows:
        key = (row.get("sample_id"), row.get("condition"))
        if key in new_by_key:
            duplicate_new.append(key)
        new_by_key[key] = row
    locked_by_key = {
        (row["sample_id"], row["condition"]): row
        for row in locked_prediction_rows
        if row.get("condition") in LOCKED_CONDITIONS
    }

    hashes = {
        "manifest": sha256(args.manifest),
        "vad": sha256(args.vad),
        "selector": sha256(args.selector),
        "protocol": sha256(args.protocol),
        "runner": sha256(args.runner),
        "locked_selector": sha256(args.locked_selector),
        "locked_predictions": sha256(args.locked_predictions),
        "new_shards": {path: sha256(path) for path in args.new_shards},
    }

    selector_contract = True
    option_hash_contract = True
    for sample_id, row in selector.items():
        options = parse_options(manifest_by_id[sample_id]["selector_prompt"])
        expected_option_hashes = [text_sha256(option) for option in options]
        by_condition = row.get("indices_by_condition", {})
        vad_start, vad_end = row.get("vad_source_frame_interval", [-1, -1])
        selector_contract &= (
            row.get("status") == "PASS"
            and row.get("correct_answer_available_to_selector") is False
            and row.get("answerer_prediction_available_to_selector") is False
            and row.get("strict_option_parse") is True
            and row.get("option_count") == 3
            and row.get("frame_budget") == 8
            and row.get("encoder") == "google/siglip-so400m-patch14-384"
            and row.get("encoder_revision") == "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
            and set(by_condition) == set(NEW_CONDITIONS)
            and row.get("input_hashes", {}).get("manifest") == hashes["manifest"]
            and row.get("input_hashes", {}).get("vad") == hashes["vad"]
            and row.get("input_hashes", {}).get("protocol") == hashes["protocol"]
        )
        option_hash_contract &= row.get("option_text_sha256") == expected_option_hashes
        for condition in NEW_CONDITIONS:
            indices = by_condition.get(condition, [])
            selector_contract &= (
                len(indices) == 8
                and len(set(indices)) == 8
                and indices == sorted(indices)
                and all(vad_start <= index <= vad_end for index in indices)
            )

    new_prediction_contract = True
    prompt_contract = True
    provenance_contract = True
    decoding_contracts = set()
    parsed = True
    for (sample_id, condition), row in new_by_key.items():
        if sample_id not in selector or condition not in NEW_CONDITIONS:
            new_prediction_contract = False
            continue
        selected = selector[sample_id]
        indices = row.get("selected_source_indices", [])
        question = manifest_by_id[sample_id]["selector_prompt"]
        prompt = "".join(f"Frame{i + 1}: <image>\n" for i in range(8)) + question + "\nRespond with only the letter A, B, or C."
        decoding = row.get("decoding", {})
        decoding_contracts.add(json.dumps(decoding, sort_keys=True))
        new_prediction_contract &= (
            row.get("model_id") == MODEL
            and row.get("model_revision") == REVISION
            and row.get("audio") == "off"
            and row.get("answer_used") is False
            and row.get("generation_compat_applied") is False
            and row.get("transformers_version") == "4.47.1"
            and row.get("torch_version") == "2.4.1+cu121"
            and row.get("frame_budget") == 8
            and indices == selected["indices_by_condition"][condition]
            and decoding.get("num_beams") == 1
            and decoding.get("max_new_tokens") == 8
            and decoding.get("do_sample") is False
        )
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
        parsed &= row.get("parse_status") == "parsed" and row.get("prediction") in "ABC"

    locked_contract = True
    for (sample_id, condition), row in locked_by_key.items():
        indices = row.get("selected_source_indices", [])
        locked_contract &= (
            sample_id in locked_selector
            and indices == locked_selector[sample_id]["indices_by_condition"][condition]
            and row.get("model_id") == MODEL
            and row.get("model_revision") == REVISION
            and row.get("audio") == "off"
            and row.get("answer_used") is False
            and row.get("frame_budget") == 8
            and row.get("parse_status") == "parsed"
            and row.get("prediction") in "ABC"
        )

    all_rows_by_key = {**locked_by_key, **new_by_key}
    prediction_counts = {
        condition: collections.Counter(
            all_rows_by_key[(sample_id, condition)]["prediction"] for sample_id in order
        )
        if all((sample_id, condition) in all_rows_by_key for sample_id in order)
        else collections.Counter()
        for condition in ALL_CONDITIONS
    }
    no_collapse = all(
        counts and max(counts.values()) / 48 < 0.75 for counts in prediction_counts.values()
    )

    hard_checks = {
        "manifest_48_unique": len(manifest) == len(set(order)) == 48,
        "selector_48_unique_and_matching":
            len(selector_rows) == len(selector) == 48 and set(selector) == set(order),
        "strict_three_option_parse_and_hashes_exact": option_hash_contract,
        "selector_label_excluded_and_indices_valid": selector_contract,
        "exact_144_new_prediction_rows": len(new_rows) == 144,
        "new_prediction_keys_exact": set(new_by_key) == expected_new,
        "no_duplicate_new_prediction_keys": not duplicate_new,
        "all_new_predictions_parsed": parsed,
        "new_answer_blind_audio_off_frozen_inference": new_prediction_contract,
        "identical_new_runtime_decoding_contract": len(decoding_contracts) == 1,
        "new_question_and_prompt_hashes_exact": prompt_contract,
        "new_provenance_hashes_exact": provenance_contract,
        "locked_selector_hash_exact": hashes["locked_selector"] == LOCKED_SELECTOR_SHA256,
        "locked_predictions_hash_exact": hashes["locked_predictions"] == LOCKED_PREDICTIONS_SHA256,
        "locked_prediction_keys_exact": set(locked_by_key) == expected_locked,
        "locked_baseline_rows_match_selector_and_inference_contract": locked_contract,
        "no_option_position_collapse_below_0_75": no_collapse,
    }
    technical_pass = all(hard_checks.values())

    metrics: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}
    promotion_checks: dict[str, bool] = {}
    promote = False
    merged = Path(args.merged_output)
    if technical_pass:
        merged.parent.mkdir(parents=True, exist_ok=True)
        with merged.open("w") as sink:
            for sample_id in order:
                for condition in ALL_CONDITIONS:
                    sink.write(json.dumps(all_rows_by_key[(sample_id, condition)]) + "\n")

        enriched = []
        for sample_id in sorted(order):
            answer = manifest_by_id[sample_id]["answer_evaluation_only"]
            item = {}
            for condition in ALL_CONDITIONS:
                row = dict(all_rows_by_key[(sample_id, condition)])
                row["correct"] = row["prediction"] == answer
                item[condition] = row
            enriched.append(item)
        for condition in ALL_CONDITIONS:
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
            "contrast_vs_srvr_uniform": comparison(enriched, "srvr_uniform_8", "option_contrast_8"),
            "contrast_vs_visual_coverage": comparison(enriched, "visual_coverage_8", "option_contrast_8"),
            "contrast_vs_option_maxsim": comparison(enriched, "option_maxsim_8", "option_contrast_8"),
            "contrast_vs_shuffled_option_contrast": comparison(
                enriched, "shuffled_option_contrast_8", "option_contrast_8"
            ),
        }
        srvr = comparisons["contrast_vs_srvr_uniform"]
        visual = comparisons["contrast_vs_visual_coverage"]
        maxsim = comparisons["contrast_vs_option_maxsim"]
        shuffled = comparisons["contrast_vs_shuffled_option_contrast"]
        promotion_checks = {
            "contrast_minus_srvr_at_least_3_of_48": srvr["delta_correct_count"] >= 3,
            "contrast_minus_visual_coverage_at_least_3_of_48": visual["delta_correct_count"] >= 3,
            "contrast_minus_option_maxsim_at_least_2_of_48": maxsim["delta_correct_count"] >= 2,
            "contrast_minus_shuffled_at_least_2_of_48": shuffled["delta_correct_count"] >= 2,
            "contrast_vs_srvr_ci_lower_not_below_minus_1_of_48":
                srvr["paired_bootstrap_95_ci"][0] >= -1 / 48,
            "contrast_vs_visual_ci_lower_not_below_minus_1_of_48":
                visual["paired_bootstrap_95_ci"][0] >= -1 / 48,
            "contrast_at_least_43_of_48": metrics["option_contrast_8"]["correct"] >= 43,
            "locked_baselines_still_42_of_48":
                metrics["srvr_uniform_8"]["correct"] == 42
                and metrics["visual_coverage_8"]["correct"] == 42,
        }
        promote = all(promotion_checks.values())

    report = {
        "status": "PROMOTE_TO_FULL291_PROTOCOL" if promote else (
            "STOP_OCCS" if technical_pass else "FAIL_TECHNICAL_AUDIT"
        ),
        "scope": "sequential method-development kill-test on frozen CMTB Dev-48",
        "known_baseline_disclosure": (
            "The locked 42/48 SRVR-uniform and visual-coverage outcomes were known before OCCS was frozen."
        ),
        "technical_pass": technical_pass,
        "promote_to_full291_protocol": promote,
        "hard_checks": hard_checks,
        "promotion_checks": promotion_checks,
        "metrics": metrics,
        "comparisons": comparisons,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS},
        "input_hashes": hashes,
        "merged_output": str(merged) if technical_pass else None,
        "merged_output_sha256": sha256(merged) if technical_pass else None,
    }
    Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# OCCS InternVL CMTB Dev-48 frozen decision report",
        "",
        f"- Status: **{report['status']}**",
        f"- Technical audit: **{'PASS' if technical_pass else 'FAIL'}**",
        f"- Promote to Full-291 protocol: **{promote}**",
        "- Scope: sequential method-development kill-test; not independent confirmation.",
        "- Disclosure: the locked 42/48 baseline outcomes were known before OCCS was frozen.",
        "",
        "## Accuracy",
        "",
        "| Condition | Correct | Accuracy | Predictions |",
        "|---|---:|---:|---|",
    ]
    for condition in ALL_CONDITIONS:
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
            f"[{100*value['paired_bootstrap_95_ci'][0]:.1f}, "
            f"{100*value['paired_bootstrap_95_ci'][1]:.1f}] pp."
        )
    lines += ["", "## Frozen promotion checks", ""]
    lines.extend(f"- {key}: {value}" for key, value in promotion_checks.items())
    lines += ["", "## Technical checks", ""]
    lines.extend(f"- {key}: {value}" for key, value in hard_checks.items())
    Path(args.output_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if technical_pass else 1)


if __name__ == "__main__":
    main()
