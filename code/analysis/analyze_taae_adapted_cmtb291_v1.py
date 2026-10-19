#!/usr/bin/env python3
"""Audit and analyze adapted TA-AE on held-out VideoLLaMA2.1 CMTB-291."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
N = 291
MODEL_ID = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
MODEL_REVISION = "d944d4298cb0188c3d5406eff42929cb67731361"
DATASET_REVISION = "367fd47c60031e68f36acc4dd82903098eb4178a"
TA_AE_COMMIT = "ca6a9a05d7f362c3e7dd6701d9246f9fa1a66cbe"
NEW_CONDITIONS = ("TA_AE_adapted_Raw_AV", "TA_AE_adapted_Video_only")
ALL_CONDITIONS = (
    "Raw_AV",
    "Video_only",
    "VAD_localized_clip",
    "AVCD_Raw_AV",
    *NEW_CONDITIONS,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=ROOT / "outputs/taae_adapted_execution_freeze_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/taae_adapted_cmtb291_analysis_v1.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "outputs/taae_adapted_cmtb291_audit_v1.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/taae_adapted_cmtb291_report_v1.md",
    )
    return parser.parse_args()


def jsonl(path: Path) -> list[dict]:
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


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired(left: dict[str, dict], right: dict[str, dict], seed: int) -> dict:
    ids = sorted(left)
    if set(ids) != set(right) or len(ids) != N:
        raise RuntimeError("paired coverage mismatch")
    left_only = sum(bool(left[sid]["correct"]) and not right[sid]["correct"] for sid in ids)
    right_only = sum(bool(right[sid]["correct"]) and not left[sid]["correct"] for sid in ids)
    differences = {
        sid: int(left[sid]["correct"]) - int(right[sid]["correct"])
        for sid in ids
    }
    randomizer = random.Random(seed)
    bootstrap = []
    for _ in range(20_000):
        sample = randomizer.choices(ids, k=N)
        bootstrap.append(sum(differences[sid] for sid in sample) / N)
    return {
        "n": N,
        "left_accuracy": sum(row["correct"] for row in left.values()) / N,
        "right_accuracy": sum(row["correct"] for row in right.values()) / N,
        "left_minus_right": sum(differences.values()) / N,
        "right_wrong_to_left_correct": left_only,
        "right_correct_to_left_wrong": right_only,
        "exact_mcnemar_p": exact_p(left_only, right_only),
        "paired_video_bootstrap_95ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": seed,
    }


def main() -> None:
    args = arguments()
    standard_paths = [
        ROOT / f"outputs/videollama21_standard_cmtb291_v1_shard{i}.jsonl"
        for i in range(4)
    ]
    avcd_paths = [
        ROOT / f"outputs/videollama21_avcd_cmtb291_v1_shard{i}.jsonl"
        for i in range(4)
    ]
    taae_paths = [
        ROOT / f"outputs/taae_adapted_cmtb291_v1_shard{i}.jsonl"
        for i in range(4)
    ]
    paths = standard_paths + avcd_paths + taae_paths
    missing = [str(path) for path in paths + [args.freeze] if not path.is_file()]
    if missing:
        raise RuntimeError({"missing": missing})

    standard = [row for path in standard_paths for row in jsonl(path)]
    avcd = [row for path in avcd_paths for row in jsonl(path)]
    taae = [row for path in taae_paths for row in jsonl(path)]
    groups = {
        "Raw_AV": {row["sample_id"]: row for row in standard if row["condition"] == "Raw_AV"},
        "Video_only": {row["sample_id"]: row for row in standard if row["condition"] == "Video_only"},
        "VAD_localized_clip": {
            row["sample_id"]: row for row in standard if row["condition"] == "VAD_localized_clip"
        },
        "AVCD_Raw_AV": {row["sample_id"]: row for row in avcd if row["condition"] == "AVCD_Raw_AV"},
        **{
            condition: {row["sample_id"]: row for row in taae if row["condition"] == condition}
            for condition in NEW_CONDITIONS
        },
    }
    freeze = json.loads(args.freeze.read_text())
    manifest_path = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
    manifest = {row["sample_id"]: row for row in jsonl(manifest_path)}
    ids = set(manifest)

    duplicate_counts = Counter((row["sample_id"], row["condition"]) for row in taae)
    checks = {
        "execution_frozen_after_preflight_before_fulln": (
            freeze.get("status") == "FROZEN_AFTER_STRUCTURAL_PREFLIGHT_BEFORE_FULLN"
        ),
        "exact_source_row_counts": (
            len(standard) == 3 * N and len(avcd) == N and len(taae) == 2 * N
        ),
        "exact_population_each_condition": all(
            len(group) == N and set(group) == ids for group in groups.values()
        ),
        "new_rows_unique": (
            len(duplicate_counts) == 2 * N and all(count == 1 for count in duplicate_counts.values())
        ),
        "new_condition_set_exact": set(row.get("condition") for row in taae) == set(NEW_CONDITIONS),
        "frozen_file_hashes_match": (
            freeze.get("runner_sha256")
            == sha256(ROOT / "src/ta_ae/run_taae_adapted_cmtb291_v1.py")
            and freeze.get("protocol_sha256")
            == sha256(ROOT / "paper/ta_ae_adapted_cmtb291_protocol_v1.md")
            and freeze.get("manifest_sha256") == sha256(manifest_path)
            and freeze.get("intervention_sha256")
            == sha256(ROOT / "outputs/taae_videollama21_temporal_intervention_v1.npz")
        ),
        "preflight_passed": freeze.get("preflight_status")
        == "PASS_OUTCOME_BLIND_STRUCTURAL_PREFLIGHT",
        "model_and_method_exact": all(
            row.get("model_id") == MODEL_ID
            and row.get("model_revision") == MODEL_REVISION
            and row.get("ta_ae_code_commit") == TA_AE_COMMIT
            and row.get("ta_ae_dataset_revision") == DATASET_REVISION
            and row.get("routing") == "task_classified_temporal_variant"
            and row.get("top_k_heads") == 128
            and row.get("alpha") == 32.0
            and row.get("downsample_ratio") == 4
            for row in taae
        ),
        "row_embedded_hashes_match_freeze": all(
            row.get("runner_sha256") == freeze.get("runner_sha256")
            and row.get("protocol_sha256") == freeze.get("protocol_sha256")
            and row.get("manifest_sha256") == freeze.get("manifest_sha256")
            and row.get("intervention_sha256") == freeze.get("intervention_sha256")
            for row in taae
        ),
        "prompt_answer_video_exact": all(
            row.get("question") == manifest[row["sample_id"]]["question"]
            and row.get("answer") == manifest[row["sample_id"]]["answer"]
            and row.get("video_id") == manifest[row["sample_id"]]["video_id"]
            and row.get("video_path") == manifest[row["sample_id"]]["video_path"]
            for row in taae
        ),
        "decoder_exact": all(
            row.get("seed") == 20260824
            and row.get("max_new_tokens") == 32
            and row.get("do_sample") is False
            and row.get("temperature") == 0.0
            for row in taae
        ),
        "hook_exercised_and_finite": all(
            isinstance(row.get("hook_call_count"), int)
            and row["hook_call_count"] > 0
            and isinstance(row.get("hook_edited_elements"), int)
            and row["hook_edited_elements"] > 0
            and isinstance(row.get("hook_max_delta_abs"), (int, float))
            and math.isfinite(row["hook_max_delta_abs"])
            and row["hook_max_delta_abs"] > 0
            for row in taae
        ),
        "media_interface_exact": all(
            row.get("loader", {}).get("va")
            is (row["condition"] == "TA_AE_adapted_Raw_AV")
            and row.get("loader", {}).get("has_audio_tensor")
            is (row["condition"] == "TA_AE_adapted_Raw_AV")
            and row.get("loader", {}).get("has_video_tensor") is True
            for row in taae
        ),
        "parse_and_correctness_consistent": all(
            row.get("parse_status") in {"parsed", "parse_failure"}
            and (
                (row["parse_status"] == "parsed" and row.get("prediction") in {"A", "B", "C"})
                or (row["parse_status"] == "parse_failure" and row.get("prediction") is None)
            )
            and row.get("correct") is (row.get("prediction") == row.get("answer"))
            for row in taae
        ),
        "parse_failures_retained_wrong": all(
            not row["correct"] for row in taae if row["parse_status"] == "parse_failure"
        ),
        "baseline_audit_passed": json.loads(
            (ROOT / "outputs/videollama21_full291_stage_audit_v1.json").read_text()
        ).get("status")
        == "PASS_FULL291_PROVENANCE_AND_SPLIT_LINEAGE",
    }
    audit = {
        "status": "PASS_TAAE_ADAPTED_FULL291_PROVENANCE" if all(checks.values()) else "FAIL",
        "checks": checks,
        "row_counts": {condition: len(group) for condition, group in groups.items()},
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "freeze_sha256": sha256(args.freeze),
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "naming_boundary": (
            "TA-AE-style (adapted, task-classified); not an exact official TA-AE reproduction"
        ),
    }
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] == "FAIL":
        print(json.dumps(audit, indent=2, sort_keys=True))
        raise SystemExit(1)

    comparisons = {
        "taae_raw_vs_raw": paired(groups["TA_AE_adapted_Raw_AV"], groups["Raw_AV"], 2026082901),
        "taae_raw_vs_avcd": paired(
            groups["TA_AE_adapted_Raw_AV"], groups["AVCD_Raw_AV"], 2026082902
        ),
        "srvr_vs_taae_raw": paired(
            groups["VAD_localized_clip"], groups["TA_AE_adapted_Raw_AV"], 2026082903
        ),
        "taae_video_vs_video": paired(
            groups["TA_AE_adapted_Video_only"], groups["Video_only"], 2026082904
        ),
        "srvr_vs_taae_video": paired(
            groups["VAD_localized_clip"], groups["TA_AE_adapted_Video_only"], 2026082905
        ),
    }
    analysis = {
        "scope": "held-out CMTB-291 adapted TA-AE same-checkpoint comparison",
        "method_name": "TA-AE-style (adapted, task-classified)",
        "accuracy": {
            condition: sum(row["correct"] for row in group.values()) / N
            for condition, group in groups.items()
        },
        "parse_failures": {
            condition: Counter(row["parse_status"] for row in group.values())["parse_failure"]
            for condition, group in groups.items()
        },
        "comparisons": comparisons,
        "audit_status": audit["status"],
        "claim_boundary": (
            "The public classifier was unavailable and checkpoint-specific vectors were recomputed; "
            "this evaluates a frozen task-classified adaptation, not the complete official adaptive path."
        ),
    }
    args.output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    accuracy = analysis["accuracy"]
    report = [
        "# Adapted TA-AE on held-out CMTB-291",
        "",
        "Naming: **TA-AE-style (adapted, task-classified)**; this is not an exact official reproduction.",
        "",
        f"- Raw AV: {accuracy['Raw_AV']:.1%}",
        f"- Official AVCD: {accuracy['AVCD_Raw_AV']:.1%}",
        f"- Adapted TA-AE on Raw AV: {accuracy['TA_AE_adapted_Raw_AV']:.1%}",
        f"- Video-only: {accuracy['Video_only']:.1%}",
        f"- Adapted TA-AE on Video-only: {accuracy['TA_AE_adapted_Video_only']:.1%}",
        f"- SRVR: {accuracy['VAD_localized_clip']:.1%}",
        "",
    ]
    for name, result in comparisons.items():
        report.append(
            f"- {name}: {result['left_minus_right']:+.1%}; "
            f"95% CI [{result['paired_video_bootstrap_95ci'][0]:+.1%}, "
            f"{result['paired_video_bootstrap_95ci'][1]:+.1%}]; "
            f"exact McNemar p={result['exact_mcnemar_p']:.4g}"
        )
    report.extend(
        [
            "",
            "The released TA-AE classifier/weights were unavailable. Vectors were recomputed on the exact "
            "VideoLLaMA2.1 checkpoint and the temporal-variant route plus published global setting "
            "K=128, alpha=32 were frozen before CMTB inference.",
            f"Audit: {audit['status']}",
        ]
    )
    args.report.write_text("\n".join(report) + "\n")
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
