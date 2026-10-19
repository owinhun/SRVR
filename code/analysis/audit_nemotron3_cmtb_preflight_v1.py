#!/usr/bin/env python3
"""Audit the frozen Nemotron 3 Nano Omni 12-item mechanical preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/nemotron3_cmtb291_preflight12_v1.jsonl"
PROTOCOL = ROOT / "paper/nemotron3_nano_omni_cmtb291_nonqwen_factorial_protocol_v1.md"
ADDENDUM = ROOT / "paper/nemotron3_direct_hf_engine_addendum_v4.md"
RUNNER = ROOT / "src/nemotron/run_nemotron3_cmtb291_factorial_v1.py"
DEFAULT_PREDICTIONS = ROOT / "outputs/nemotron3_cmtb291_preflight_v8.jsonl"
DEFAULT_CACHE_AUDIT = ROOT / "outputs/nemotron3_cmtb291_preflight_cache_write_audit_v8.json"
DEFAULT_AUDIT = ROOT / "outputs/nemotron3_cmtb291_preflight_audit_v8.json"

CONDITIONS = {
    "no_pointer_audio_off",
    "no_pointer_audio_on",
    "correct_pointer_audio_off",
    "correct_pointer_audio_on",
}
MODEL_REVISION = "24e67ea000b7c2837fc8f9488aa2008524fac8ba"
REPOSITORY_REVISION = "1a9a23952bab2ffcbbf7044d08c9dfcb55fd7331"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--cache-audit", type=Path, default=DEFAULT_CACHE_AUDIT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
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
    expected = read_jsonl(MANIFEST)
    expected_ids = [row["sample_id"] for row in expected]
    rows = read_jsonl(args.predictions) if args.predictions.is_file() else []
    cache = json.loads(args.cache_audit.read_text()) if args.cache_audit.is_file() else {
        "status": "MISSING_CACHE_AUDIT",
        "outside_data4_cache_writes": ["cache audit missing"],
    }

    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row.get("sample_id")].append(row)
    keys = [(row.get("sample_id"), row.get("condition")) for row in rows]
    complete_ids = [
        sample_id
        for sample_id in expected_ids
        if len(by_sample[sample_id]) == 4
        and {row.get("condition") for row in by_sample[sample_id]} == CONDITIONS
    ]
    parsed = [row for row in rows if row.get("prediction") in {"A", "B", "C"}]
    predictions = Counter(row.get("prediction") for row in parsed)
    max_share = max(predictions.values(), default=0) / max(1, len(parsed))

    item_contracts = {}
    for sample_id in complete_ids:
        item_rows = by_sample[sample_id]
        sampling = {json.dumps(row.get("sampling_contract"), sort_keys=True) for row in item_rows}
        visual_hashes = {
            row.get("visual_contract", {}).get("canonical_tensor_sha256") for row in item_rows
        }
        visual_shapes = {
            json.dumps(row.get("visual_contract", {}).get("shape")) for row in item_rows
        }
        video_counts = {
            row.get("visual_contract", {}).get("video_placeholder_count") for row in item_rows
        }
        visual_matches = all(
            row.get("visual_contract", {}).get("matches_first_cell_exactly") is True
            for row in item_rows
        )
        audio_ok = True
        for row in item_rows:
            contract = row.get("audio_contract", {})
            if row.get("audio_on"):
                audio_ok &= bool(contract.get("sound_clips_present"))
                audio_ok &= contract.get("audio_placeholder_count", 0) > 0
                audio_ok &= bool(contract.get("sound_clip_lengths"))
            else:
                audio_ok &= not bool(contract.get("sound_clips_present"))
                audio_ok &= contract.get("audio_placeholder_count") == 0
        item_contracts[sample_id] = {
            "sampling_identical": len(sampling) == 1,
            "visual_hash_identical": len(visual_hashes) == 1 and None not in visual_hashes,
            "visual_shape_identical": len(visual_shapes) == 1,
            "video_placeholder_count_identical": len(video_counts) == 1,
            "all_visual_match_flags_true": visual_matches,
            "audio_contract_distinct_and_valid": bool(audio_ok),
        }

    runner_hash = sha256(RUNNER)
    environment_ok = bool(rows) and all(
        row.get("model_revision") == MODEL_REVISION
        and row.get("repository_revision") == REPOSITORY_REVISION
        and row.get("code_sha256") == runner_hash
        and row.get("environment", {}).get("mamba_fast_path") is True
        and row.get("environment", {}).get("mamba_ssm") == "2.2.6.post3"
        and row.get("environment", {}).get("causal_conv1d") == "1.6.2.post1"
        and row.get("environment", {}).get("transformers") == "5.6.0"
        and row.get("environment", {}).get("huggingface_hub") == "1.5.0"
        and row.get("environment", {}).get("open_clip_torch") == "3.2.0"
        and row.get("environment", {}).get("rectangular_evs_compatibility") is True
        and row.get("environment", {}).get("multigpu_evs_mask_compatibility") is True
        and row.get("environment", {}).get("patched_modeling_sha256")
        == "8b0177c2971c99c407c152e576b6978ce459c4af856d86d6872e771e8c4c6592"
        and row.get("environment", {}).get("cuda_visible_devices") == "2,3"
        for row in rows
    )
    cache_ok = (
        cache.get("status") == "PASS_NO_RUNTIME_CACHE_WRITE_OUTSIDE_DATA4"
        and not cache.get("outside_data4_cache_writes")
    )
    checks = {
        "exact_preflight_selection": set(by_sample) == set(expected_ids),
        "no_duplicate_sample_condition": len(keys) == len(set(keys)),
        "at_least_11_items_all_four_conditions": len(complete_ids) >= 11,
        "at_least_42_of_48_parse": len(parsed) >= 42,
        "at_least_two_predicted_letters": len(predictions) >= 2,
        "no_predicted_letter_above_90_percent": max_share <= 0.90,
        "input_contracts_pass_for_completed_items": len(complete_ids) >= 11
        and all(all(values.values()) for values in item_contracts.values()),
        "frozen_model_runner_and_fast_path": environment_ok,
        "no_runtime_cache_write_outside_data4": cache_ok,
    }
    gate_pass = all(checks.values())
    payload = {
        "status": "PASS_MECHANICAL_PREFLIGHT" if gate_pass else "STOP_MECHANICAL_PREFLIGHT_FAILED",
        "gate_pass": gate_pass,
        "accuracy_was_not_computed_or_used_as_a_gate": True,
        "checks": checks,
        "expected_sample_ids": expected_ids,
        "observed_sample_ids": sorted(by_sample),
        "row_count": len(rows),
        "complete_item_count": len(complete_ids),
        "complete_sample_ids": complete_ids,
        "parse_count": len(parsed),
        "parse_failure_count": len(rows) - len(parsed),
        "runtime_failure_count": sum(row.get("parse_status") == "runtime_failure" for row in rows),
        "condition_counts": Counter(row.get("condition") for row in rows),
        "prediction_counts": predictions,
        "maximum_prediction_share": max_share,
        "item_contracts": item_contracts,
        "cache_write_audit": cache,
        "hashes": {
            "predictions": sha256(args.predictions) if args.predictions.is_file() else None,
            "cache_audit": sha256(args.cache_audit) if args.cache_audit.is_file() else None,
            "manifest": sha256(MANIFEST),
            "protocol": sha256(PROTOCOL),
            "addendum": sha256(ADDENDUM),
            "runner": runner_hash,
            "auditor": sha256(Path(__file__)),
        },
    }
    args.audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
