#!/usr/bin/env python3
"""Audit and analyze the bounded TA-AE README-default sensitivity."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
N = 291
CONDITIONS = ("TA_AE_adapted_Raw_AV", "TA_AE_adapted_Video_only")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired(left: dict[str, dict], right: dict[str, dict], seed: int) -> dict:
    ids = sorted(left)
    if len(ids) != N or set(ids) != set(right):
        raise RuntimeError("paired coverage mismatch")
    left_only = sum(left[sid]["correct"] and not right[sid]["correct"] for sid in ids)
    right_only = sum(right[sid]["correct"] and not left[sid]["correct"] for sid in ids)
    differences = {sid: int(left[sid]["correct"]) - int(right[sid]["correct"]) for sid in ids}
    generator = random.Random(seed)
    bootstrap = [
        sum(differences[sid] for sid in generator.choices(ids, k=N)) / N
        for _ in range(20_000)
    ]
    return {
        "n": N,
        "left_accuracy": sum(row["correct"] for row in left.values()) / N,
        "right_accuracy": sum(row["correct"] for row in right.values()) / N,
        "left_minus_right": sum(differences.values()) / N,
        "right_wrong_to_left_correct": left_only,
        "right_correct_to_left_wrong": right_only,
        "exact_mcnemar_p": exact_p(left_only, right_only),
        "paired_video_bootstrap_95ci": [
            percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)
        ],
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": seed,
    }


def main() -> None:
    paths = [
        ROOT / f"outputs/taae_readme_default_cmtb291_v1_shard{i}.jsonl"
        for i in range(4)
    ]
    freeze_path = ROOT / "outputs/taae_readme_default_execution_freeze_v1.json"
    manifest_path = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
    wrapper = ROOT / "src/ta_ae/run_taae_readme_default_cmtb291_v1.py"
    base_runner = ROOT / "src/ta_ae/run_taae_adapted_cmtb291_v1.py"
    protocol = ROOT / "paper/ta_ae_readme_default_sensitivity_protocol_v1.md"
    intervention = ROOT / "outputs/taae_videollama21_temporal_intervention_readme_default_v1.npz"
    missing = [str(path) for path in paths + [freeze_path] if not path.is_file()]
    if missing:
        raise RuntimeError({"missing": missing})
    data = [row for path in paths for row in rows(path)]
    manifest = {row["sample_id"]: row for row in rows(manifest_path)}
    freeze = json.loads(freeze_path.read_text())
    groups = {
        condition: {row["sample_id"]: row for row in data if row["condition"] == condition}
        for condition in CONDITIONS
    }
    duplicates = Counter((row.get("sample_id"), row.get("condition")) for row in data)
    checks = {
        "post_outcome_boundary_explicit": freeze.get("post_outcome_sensitivity") is True,
        "single_setting_search_closed": freeze.get("additional_hyperparameter_search_permitted") is False,
        "exact_rows": len(data) == 2 * N,
        "exact_population": all(len(group) == N and set(group) == set(manifest) for group in groups.values()),
        "unique_rows": len(duplicates) == 2 * N and all(value == 1 for value in duplicates.values()),
        "condition_set": set(row.get("condition") for row in data) == set(CONDITIONS),
        "frozen_hashes": (
            freeze.get("wrapper_sha256") == sha256(wrapper)
            and freeze.get("base_runner_sha256") == sha256(base_runner)
            and freeze.get("protocol_sha256") == sha256(protocol)
            and freeze.get("manifest_sha256") == sha256(manifest_path)
            and freeze.get("intervention_sha256") == sha256(intervention)
        ),
        "row_hashes": all(
            row.get("runner_sha256") == freeze.get("wrapper_sha256")
            and row.get("manifest_sha256") == freeze.get("manifest_sha256")
            and row.get("intervention_sha256") == freeze.get("intervention_sha256")
            for row in data
        ),
        "setting_exact": all(
            row.get("top_k_heads") == 32
            and row.get("alpha") == 8.0
            and row.get("downsample_ratio") == 4
            and row.get("routing") == "task_classified_temporal_variant"
            for row in data
        ),
        "inputs_exact": all(
            row.get("question") == manifest[row["sample_id"]]["question"]
            and row.get("answer") == manifest[row["sample_id"]]["answer"]
            and row.get("video_id") == manifest[row["sample_id"]]["video_id"]
            and row.get("video_path") == manifest[row["sample_id"]]["video_path"]
            for row in data
        ),
        "decoder_exact": all(
            row.get("seed") == 20260824
            and row.get("max_new_tokens") == 32
            and row.get("do_sample") is False
            and row.get("temperature") == 0.0
            for row in data
        ),
        "hook_exercised": all(
            isinstance(row.get("hook_call_count"), int) and row["hook_call_count"] > 0
            and isinstance(row.get("hook_edited_elements"), int) and row["hook_edited_elements"] > 0
            and isinstance(row.get("hook_max_delta_abs"), (int, float))
            and math.isfinite(row["hook_max_delta_abs"]) and row["hook_max_delta_abs"] > 0
            for row in data
        ),
        "parse_consistent": all(
            row.get("parse_status") in {"parsed", "parse_failure"}
            and row.get("correct") is (row.get("prediction") == row.get("answer"))
            and (row.get("prediction") in {"A", "B", "C"} if row["parse_status"] == "parsed" else row.get("prediction") is None)
            for row in data
        ),
    }
    audit = {
        "status": "PASS_TAAE_README_DEFAULT_FULL291_PROVENANCE" if all(checks.values()) else "FAIL",
        "checks": checks,
        "row_counts": {condition: len(group) for condition, group in groups.items()},
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "analyzer_sha256": sha256(Path(__file__).resolve()),
    }
    audit_path = ROOT / "outputs/taae_readme_default_cmtb291_audit_v1.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] == "FAIL":
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    standard = []
    for i in range(4):
        standard.extend(rows(ROOT / f"outputs/videollama21_standard_cmtb291_v1_shard{i}.jsonl"))
    baselines = {
        "Raw_AV": {r["sample_id"]: r for r in standard if r["condition"] == "Raw_AV"},
        "Video_only": {r["sample_id"]: r for r in standard if r["condition"] == "Video_only"},
        "VAD_localized_clip": {r["sample_id"]: r for r in standard if r["condition"] == "VAD_localized_clip"},
    }
    result = {
        "status": audit["status"],
        "scope": "post-outcome, single README-default sensitivity; not a preregistered primary baseline",
        "setting": {"top_k": 32, "alpha": 8.0, "ratio": 4},
        "accuracy": {
            **{name: sum(row["correct"] for row in group.values()) / N for name, group in baselines.items()},
            **{f"README_default_{name}": sum(row["correct"] for row in group.values()) / N for name, group in groups.items()},
        },
        "parse_failures": {
            f"README_default_{name}": sum(row["parse_status"] == "parse_failure" for row in group.values())
            for name, group in groups.items()
        },
        "comparisons": {
            "readme_raw_vs_raw": paired(groups[CONDITIONS[0]], baselines["Raw_AV"], 2026083001),
            "readme_video_vs_video": paired(groups[CONDITIONS[1]], baselines["Video_only"], 2026083002),
            "srvr_vs_readme_raw": paired(baselines["VAD_localized_clip"], groups[CONDITIONS[0]], 2026083003),
            "srvr_vs_readme_video": paired(baselines["VAD_localized_clip"], groups[CONDITIONS[1]], 2026083004),
        },
        "reporting_boundary": "Selected after the primary adapted run failed, solely because K32/alpha8 are the public README defaults; both settings must remain visible.",
    }
    output = ROOT / "outputs/taae_readme_default_cmtb291_analysis_v1.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = ROOT / "outputs/taae_readme_default_cmtb291_report_v1.md"
    report.write_text(
        "# TA-AE README-default sensitivity\n\n"
        f"Audit: **{audit['status']}**. This is a post-outcome bounded sensitivity, not a preregistered primary baseline.\n\n"
        + "\n".join(f"- {name}: {100*value:.1f}%" for name, value in result["accuracy"].items())
        + "\n\nBoth the prospectively frozen K128/alpha32 result and this K32/alpha8 README-default result must be reported; no hyperparameter search follows.\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

