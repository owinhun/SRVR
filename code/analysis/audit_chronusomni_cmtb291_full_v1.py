#!/usr/bin/env python3
"""Audit full-population coverage and input matching for ChronusOmni CMTB."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
PROTOCOL = ROOT / "paper/chronusomni_cmtb291_strong_temporal_baseline_protocol_v1.md"
AUDIT = ROOT / "outputs/chronusomni_cmtb291_factorial_full_audit_v1.json"
CONDITIONS = (
    "no_pointer_audio_off",
    "no_pointer_audio_on",
    "correct_pointer_audio_off",
    "correct_pointer_audio_on",
)
POINTER_TEMPLATE = (
    "The relevant speech activity occurs from {start:.2f} to {end:.2f} seconds. "
    "Answer using only the visual action during that interval. Do not infer or use transcript content."
)
QUESTION_SUFFIX = "Answer with only the option letter A, B, or C."
MODEL_REVISION = "7ffb0916d49ae31d5d381d58dd7d0a40d6776d01"
CODE_COMMIT = "f9730d2294e90f6090ec960ac4810d896278234d"
PROTOCOL_SHA256 = "78a05865756c8a2a2ec0613b8ce40639d7e411d2ee9fc51243c0fb6209788f0d"
SOURCE_SHA256 = "9221a8af1f67e9da22bedbb2bca9c68333ec6a38c5f614219c05376b2b451f9e"
MANIFEST_SHA256 = "dd4f9cd881310d3ca280219dbc0f789c7997d78728eded30d153054e5ad92416"
AUDIO_MANIFEST_SHA256 = "636abdafffc18f4f84d67a0032a9ecb444930d33c154e3a686c033239af88aa9"
RUNNER_CODE_SHA256 = "3cf62105db1da5d9758b1ddc47d820e2c1be5c83fcf53c5725bfe25e2ed1c2dc"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        default=[
            ROOT / f"outputs/chronusomni_cmtb291_factorial_v1_shard{index}.jsonl"
            for index in range(4)
        ],
    )
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = cli()
    manifest = read_jsonl(MANIFEST)
    manifest_by_id = {row["sample_id"]: row for row in manifest}
    rows = [row for path in args.predictions for row in read_jsonl(path)]
    keys = [(row.get("sample_id"), row.get("condition")) for row in rows]
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row.get("sample_id")].append(row)

    expected_ids = set(manifest_by_id)
    expected_keys = {(sample_id, condition) for sample_id in expected_ids for condition in CONDITIONS}
    observed_keys = set(keys)
    item_contract_checks: dict[str, dict[str, bool]] = {}
    for sample_id in sorted(expected_ids):
        item_rows = by_sample[sample_id]
        by_condition = {row.get("condition"): row for row in item_rows}
        complete = set(by_condition) == set(CONDITIONS) and len(item_rows) == 4
        if not complete:
            item_contract_checks[sample_id] = {"complete": False}
            continue
        source = manifest_by_id[sample_id]
        expected_base_prompt = source["question"] + "\n" + QUESTION_SUFFIX
        start, end = source["speech_start_sec"], source["speech_end_sec"]
        expected_pointer_prompt = POINTER_TEMPLATE.format(start=start, end=end) + "\n\n" + expected_base_prompt
        contracts = [
            json.dumps(by_condition[condition].get("visual_contract"), sort_keys=True)
            for condition in CONDITIONS
        ]
        answers = [by_condition[condition].get("answer") for condition in CONDITIONS]
        video_ids = [by_condition[condition].get("video_id") for condition in CONDITIONS]
        intervals = [by_condition[condition].get("speech_interval_sec") for condition in CONDITIONS]
        failures = [by_condition[condition].get("failure") for condition in CONDITIONS]
        successful_item = all(failure is None for failure in failures)
        failed_item = all(failure is not None for failure in failures)
        checks = {
            "complete": True,
            "visual_contract_identical": len(set(contracts)) == 1,
            "frame_count_64_or_consistent_denominator_failure": (
                successful_item
                and all(
                    (by_condition[condition].get("visual_contract") or {}).get("frame_count") == 64
                    for condition in CONDITIONS
                )
            ) or (
                failed_item
                and all(by_condition[condition].get("visual_contract") is None for condition in CONDITIONS)
            ),
            "failure_contract_consistent": (
                successful_item
                or (
                    failed_item
                    and len(set(failures)) == 1
                    and all(
                        by_condition[condition].get("parse_status") == "runtime_failure"
                        and by_condition[condition].get("prediction") is None
                        and not by_condition[condition].get("correct")
                        for condition in CONDITIONS
                    )
                )
            ),
            "answer_identical_and_manifest_matched": len(set(answers)) == 1 and answers[0] == source["answer"],
            "video_identical_and_manifest_matched": len(set(video_ids)) == 1 and video_ids[0] == source["video_id"],
            "speech_interval_identical_and_manifest_matched": (
                len({json.dumps(interval) for interval in intervals}) == 1
                and intervals[0] == [start, end]
            ),
            "no_pointer_prompt_exact": all(
                by_condition[condition].get("prompt") == expected_base_prompt
                for condition in CONDITIONS[:2]
            ),
            "correct_pointer_prompt_exact": all(
                by_condition[condition].get("prompt") == expected_pointer_prompt
                for condition in CONDITIONS[2:]
            ),
            "prompt_sha256_self_consistent": all(
                by_condition[condition].get("prompt_sha256")
                == hashlib.sha256(by_condition[condition].get("prompt", "").encode()).hexdigest()
                for condition in CONDITIONS
            ),
            "audio_flags_exact": (
                not by_condition["no_pointer_audio_off"].get("audio_on")
                and by_condition["no_pointer_audio_on"].get("audio_on")
                and not by_condition["correct_pointer_audio_off"].get("audio_on")
                and by_condition["correct_pointer_audio_on"].get("audio_on")
            ),
            "audio_hash_contract_exact": (
                by_condition["no_pointer_audio_off"].get("audio_sha256") is None
                and by_condition["correct_pointer_audio_off"].get("audio_sha256") is None
                and bool(by_condition["no_pointer_audio_on"].get("audio_sha256"))
                and by_condition["no_pointer_audio_on"].get("audio_sha256")
                == by_condition["correct_pointer_audio_on"].get("audio_sha256")
            ),
            "deterministic_decoder_exact": all(
                by_condition[condition].get("decoding")
                == {"do_sample": False, "max_new_tokens": 32, "num_beams": 1}
                for condition in CONDITIONS
            ),
        }
        item_contract_checks[sample_id] = checks

    frozen_hashes_ok = all(
        row.get("protocol_sha256") == PROTOCOL_SHA256
        and row.get("source_manifest_sha256") == SOURCE_SHA256
        and row.get("joined_manifest_sha256") == MANIFEST_SHA256
        and row.get("audio_manifest_sha256") == AUDIO_MANIFEST_SHA256
        and row.get("code_sha256") == RUNNER_CODE_SHA256
        for row in rows
    ) and bool(rows)
    frozen_revisions_ok = all(
        row.get("environment", {}).get("checkpoint_revision") == MODEL_REVISION
        and row.get("environment", {}).get("chronus_code_commit") == CODE_COMMIT
        for row in rows
    ) and bool(rows)
    per_item_ok = all(
        checks.get("complete") and all(checks.values())
        for checks in item_contract_checks.values()
    )
    checks = {
        "manifest_has_291_unique_items": len(manifest) == 291 and len(expected_ids) == 291,
        "exact_1164_rows": len(rows) == 291 * 4,
        "no_duplicate_sample_condition": len(keys) == len(observed_keys),
        "exact_sample_condition_coverage": observed_keys == expected_keys,
        "all_item_input_contracts_match": per_item_ok,
        "frozen_hashes_match": frozen_hashes_ok and sha256(PROTOCOL) == PROTOCOL_SHA256,
        "frozen_model_and_code_revisions_match": frozen_revisions_ok,
        "all_rows_full_mode": bool(rows) and all(row.get("mode") == "full" for row in rows),
        "all_runtime_package_paths_on_data4": bool(rows) and all(
            str(row.get("environment", {}).get("target_site_packages", "")).startswith("/data4/")
            for row in rows
        ),
    }
    passed = all(checks.values())
    failed_item_checks = {
        sample_id: [name for name, value in item_checks.items() if not value]
        for sample_id, item_checks in item_contract_checks.items()
        if not item_checks.get("complete") or not all(item_checks.values())
    }
    payload = {
        "status": "PASS_FULL_INPUT_CONTRACT_AUDIT" if passed else "FAIL_FULL_INPUT_CONTRACT_AUDIT",
        "pass": passed,
        "checks": checks,
        "row_count": len(rows),
        "observed_sample_count": len(by_sample),
        "condition_counts": Counter(row.get("condition") for row in rows),
        "parse_failure_count": sum(row.get("prediction") not in {"A", "B", "C"} for row in rows),
        "runtime_failure_count": sum(row.get("parse_status") == "runtime_failure" for row in rows),
        "failed_item_checks": failed_item_checks,
        "hashes": {
            "manifest": sha256(MANIFEST),
            "protocol": sha256(PROTOCOL),
            "prediction_shards": {
                str(path): sha256(path) for path in args.predictions if path.exists()
            },
            "script": sha256(Path(__file__)),
        },
    }
    args.audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
