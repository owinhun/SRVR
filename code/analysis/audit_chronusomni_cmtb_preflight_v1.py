#!/usr/bin/env python3
"""Audit the frozen 12-item ChronusOmni CMTB mechanical preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
PROTOCOL = ROOT / "paper/chronusomni_cmtb291_strong_temporal_baseline_protocol_v1.md"
OUTPUT = ROOT / "outputs/chronusomni_cmtb291_preflight_v1.jsonl"
CACHE_AUDIT = ROOT / "outputs/chronusomni_cmtb291_preflight_cache_write_audit_v1.json"
AUDIT = ROOT / "outputs/chronusomni_cmtb291_preflight_audit_v1.json"
NAMESPACE = "chronus-cmtb-preflight-v1"
CONDITIONS = {
    "no_pointer_audio_off",
    "no_pointer_audio_on",
    "correct_pointer_audio_off",
    "correct_pointer_audio_on",
}
MODEL_REVISION = "7ffb0916d49ae31d5d381d58dd7d0a40d6776d01"
CODE_COMMIT = "f9730d2294e90f6090ec960ac4810d896278234d"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=OUTPUT)
    parser.add_argument("--cache-audit", type=Path, default=CACHE_AUDIT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = cli()
    manifest = read_jsonl(MANIFEST)
    expected_ids = [
        row["sample_id"]
        for row in sorted(manifest, key=lambda row: stable_int(NAMESPACE + "|" + row["sample_id"]))[:12]
    ]
    rows = read_jsonl(args.predictions) if args.predictions.exists() else []
    cache = json.loads(args.cache_audit.read_text()) if args.cache_audit.exists() else {
        "status": "MISSING_CACHE_AUDIT", "outside_data4_model_cache_writes": ["audit_missing"]
    }
    keys = [(row.get("sample_id"), row.get("condition")) for row in rows]
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row.get("sample_id")].append(row)
    complete_samples = [
        sample_id
        for sample_id in expected_ids
        if {row.get("condition") for row in by_sample[sample_id]} == CONDITIONS
        and len(by_sample[sample_id]) == 4
    ]
    parsed = [row for row in rows if row.get("prediction") in {"A", "B", "C"}]
    visual_contract_match = {}
    for sample_id in expected_ids:
        contracts = [
            json.dumps(row.get("visual_contract"), sort_keys=True)
            for row in by_sample[sample_id]
            if row.get("condition") in CONDITIONS
        ]
        visual_contract_match[sample_id] = len(contracts) == 4 and len(set(contracts)) == 1
    revisions_ok = all(
        row.get("environment", {}).get("checkpoint_revision") == MODEL_REVISION
        and row.get("environment", {}).get("chronus_code_commit") == CODE_COMMIT
        for row in rows
    ) and bool(rows)
    selection_ok = set(by_sample) == set(expected_ids)
    unique_keys = len(keys) == len(set(keys))
    cache_ok = (
        cache.get("status") == "PASS_NO_MODEL_CACHE_WRITE_OUTSIDE_DATA4"
        and not cache.get("outside_data4_model_cache_writes")
    )
    checks = {
        "exact_preflight_selection": selection_ok,
        "no_duplicate_sample_condition": unique_keys,
        "at_least_11_items_all_four_conditions": len(complete_samples) >= 11,
        "at_least_42_of_48_parse": len(parsed) >= 42,
        "visual_contract_matches_within_every_completed_item": all(
            visual_contract_match[sample_id] for sample_id in complete_samples
        ) and len(complete_samples) >= 11,
        "frozen_model_and_code_revisions": revisions_ok,
        "no_model_cache_write_outside_data4": cache_ok,
    }
    gate_pass = all(checks.values())
    payload = {
        "status": "PASS_MECHANICAL_PREFLIGHT" if gate_pass else "STOP_MECHANICAL_PREFLIGHT_FAILED",
        "gate_pass": gate_pass,
        "accuracy_was_not_a_gate": True,
        "checks": checks,
        "expected_sample_ids": expected_ids,
        "observed_sample_ids": sorted(by_sample),
        "row_count": len(rows),
        "complete_item_count": len(complete_samples),
        "complete_sample_ids": complete_samples,
        "parse_count": len(parsed),
        "parse_failure_count": len(rows) - len(parsed),
        "runtime_failure_count": sum(row.get("parse_status") == "runtime_failure" for row in rows),
        "condition_counts": Counter(row.get("condition") for row in rows),
        "prediction_counts": Counter(row.get("prediction") for row in rows),
        "visual_contract_match": visual_contract_match,
        "cache_write_audit": cache,
        "hashes": {
            "predictions": sha256(args.predictions) if args.predictions.exists() else None,
            "cache_audit": sha256(args.cache_audit) if args.cache_audit.exists() else None,
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
        },
    }
    args.audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
