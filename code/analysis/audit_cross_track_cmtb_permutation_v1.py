#!/usr/bin/env python3
"""Independently audit the Claude-track full CMTB option permutation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path("/workspace/svhalluc")
O = ROOT / "outputs"
C = ROOT / "claude_track/outputs"
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
BASELINE = O / "video_salmonn2_cmtb291_normalized_v1.jsonl"
ROTATE = {"A": "C", "B": "A", "C": "B"}  # old letter -> new letter
INVERSE = {value: key for key, value in ROTATE.items()}
OPTIONS = {"A", "B", "C"}
SEED = 20260804
REPETITIONS = 20_000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_json(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_letter(value: str) -> str:
    match = re.search(r"(?:^|\()\s*([ABC])(?:\)|\.|\s|$)", value.strip())
    return match.group(1) if match else "?"


def exact_mcnemar(left: list[bool], right: list[bool]) -> tuple[int, int, float]:
    left_only = sum(a and not b for a, b in zip(left, right))
    right_only = sum(b and not a for a, b in zip(left, right))
    n = left_only + right_only
    p = 1.0 if n == 0 else float(binomtest(min(left_only, right_only), n, 0.5).pvalue)
    return left_only, right_only, p


def main() -> None:
    manifest_rows = [row for row in read_jsonl(MANIFEST) if row["task"] == "CMTB"]
    manifest = {row["sample_id"]: row for row in manifest_rows}
    maps = []
    results = []
    inputs = []
    source_paths = []
    for shard in (0, 1):
        map_path = C / f"cmtb_option_permute_v1_shard{shard}_map.jsonl"
        input_path = C / f"cmtb_option_permute_v1_shard{shard}.json"
        result_path = C / f"cmtb_option_permute_v1_output_shard{shard}/test_results.json"
        log_path = C / f"cmtb_option_permute_v1_shard{shard}.log"
        maps.extend(read_jsonl(map_path))
        inputs.extend(read_json(input_path))
        results.extend(read_json(result_path))
        source_paths.extend([map_path, input_path, result_path, log_path])

    map_by_id = {row["sample_id"]: row for row in maps}
    input_by_id = {row["sample_id"]: row for row in inputs}
    video_to_id = {row["video_path"]: row["sample_id"] for row in manifest_rows}
    if len(video_to_id) != 291:
        raise RuntimeError("CMTB videos are not unique")
    result_by_id = {}
    for row in results:
        path = row["id"][0]
        sid = video_to_id.get(path)
        if sid is None or sid in result_by_id:
            raise RuntimeError(f"Unknown or duplicate result path: {path}")
        result_by_id[sid] = row

    baseline_rows = [
        row for row in read_jsonl(BASELINE)
        if row["condition"] == "VAD_localized_clip"
    ]
    baseline = {row["sample_id"]: row for row in baseline_rows}
    ids = sorted(manifest)
    coverage_checks = {
        "manifest_291": len(manifest_rows) == len(manifest) == 291,
        "map_291": len(maps) == len(map_by_id) == 291 and set(map_by_id) == set(ids),
        "input_291": len(inputs) == len(input_by_id) == 291 and set(input_by_id) == set(ids),
        "result_291": len(results) == len(result_by_id) == 291 and set(result_by_id) == set(ids),
        "baseline_291": len(baseline_rows) == len(baseline) == 291 and set(baseline) == set(ids),
        "logs_exit_zero": all(path.read_text().rstrip().endswith("exit=0") for path in source_paths if path.suffix == ".log"),
    }

    permutation_checks = []
    perm_predictions = {}
    for sid in ids:
        original = manifest[sid]
        mapped = map_by_id[sid]
        input_row = input_by_id[sid]
        result = result_by_id[sid]
        expected_new_answer = ROTATE[original["answer"]]
        permutation_checks.append(
            mapped["original_answer"] == original["answer"]
            and mapped["answer"] == expected_new_answer
            and input_row["conversations"][1]["value"] == expected_new_answer
            and result["ref"] == expected_new_answer
            and mapped["question"] == input_row["conversations"][0]["value"].replace("<image>\n", "", 1)
        )
        perm_predictions[sid] = parse_letter(result["pred"])

    base_correct = [bool(baseline[sid]["correct"]) for sid in ids]
    perm_correct = [perm_predictions[sid] == map_by_id[sid]["answer"] for sid in ids]
    parse_failures = sum(perm_predictions[sid] not in OPTIONS for sid in ids)
    base_predictions = {sid: baseline[sid]["prediction"] for sid in ids}
    content_equivalent_predictions = {
        sid: INVERSE.get(perm_predictions[sid], "?") for sid in ids
    }
    content_agreement = sum(
        base_predictions[sid] == content_equivalent_predictions[sid] for sid in ids
    ) / len(ids)
    left_only, right_only, p = exact_mcnemar(perm_correct, base_correct)
    delta = np.asarray(perm_correct, dtype=float) - np.asarray(base_correct, dtype=float)
    rng = np.random.default_rng(SEED)
    draws = delta[rng.integers(0, len(delta), size=(REPETITIONS, len(delta)))].mean(axis=1)
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    checks = {
        **coverage_checks,
        "all_permutations_exact": all(permutation_checks) and len(permutation_checks) == 291,
        "zero_parse_failures": parse_failures == 0,
        "baseline_accuracy_reproduces": sum(base_correct) == 237,
        "permuted_accuracy_reproduces": sum(perm_correct) == 239,
        "accuracy_difference_small": abs(float(np.mean(perm_correct)) - float(np.mean(base_correct))) < 0.02,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "status": status,
        "design": "Cyclic option-text rotation with the same VAD-localized video support; old A/B/C moves to new C/A/B.",
        "n": 291,
        "baseline": {
            "correct": sum(base_correct),
            "accuracy": float(np.mean(base_correct)),
            "prediction_distribution": dict(Counter(base_predictions.values())),
        },
        "permuted": {
            "correct": sum(perm_correct),
            "accuracy": float(np.mean(perm_correct)),
            "prediction_distribution_new_letters": dict(Counter(perm_predictions.values())),
            "prediction_distribution_mapped_to_original_content": dict(Counter(content_equivalent_predictions.values())),
            "parse_failures": parse_failures,
        },
        "paired": {
            "permuted_minus_baseline": float(np.mean(delta)),
            "bootstrap_95ci": ci,
            "permuted_only_correct": left_only,
            "baseline_only_correct": right_only,
            "exact_mcnemar_p": p,
            "prediction_content_agreement": content_agreement,
        },
        "checks": checks,
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in source_paths + [MANIFEST, BASELINE]},
        "claim_boundary": "The full CMTB localization result is robust to one deterministic cyclic option permutation. This rejects a fixed-letter shortcut for this intervention; it does not prove invariance to every wording or ordering.",
    }
    (O / "cross_track_cmtb_option_permutation_audit_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Cross-track CMTB option-permutation audit v1",
        "",
        f"Status: **{status}**",
        "",
        f"- Original localized support: {sum(base_correct)}/291 ({100*np.mean(base_correct):.1f}%).",
        f"- Cyclically permuted options: {sum(perm_correct)}/291 ({100*np.mean(perm_correct):.1f}%).",
        f"- Paired difference: {100*np.mean(delta):+.1f} points; 95% CI [{100*ci[0]:.1f}, {100*ci[1]:.1f}]; McNemar p={p:.4g}.",
        f"- Prediction agreement after mapping permuted letters back to original option content: {100*content_agreement:.1f}%.",
        f"- Parse failures: {parse_failures}.",
        "",
        "## Interpretation",
        "",
        result["claim_boundary"],
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    lines.extend(f"| {name.replace('_', ' ')} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks.items())
    (O / "cross_track_cmtb_option_permutation_audit_v1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "checks": checks, "paired": result["paired"]}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
