#!/usr/bin/env python3
"""Gate standard inference on a complete, unique, provenance-locked AVCD CMTB-291 run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
EXPECTED_MANIFEST = "3e0d8cd8f0cafbdc596652dac0a663eef1527ae857be92623677bbc74adcb5a8"
EXPECTED_RUNNER = "8883bbef6c829b609294b2bc2279a177add994ce4412c0ce7e0b0a80ece0f813"
EXPECTED_AVCD = "92aeaf1656e6ec7760e3cd3be4a287c88a30d37c"
OUT = ROOT / "outputs/videollama21_avcd_cmtb291_completion_audit_v1.json"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    paths = [ROOT / f"outputs/videollama21_avcd_cmtb291_v1_shard{i}.jsonl" for i in range(4)]
    all_rows = [row for path in paths for row in rows(path)] if all(path.is_file() for path in paths) else []
    keys = [(row.get("sample_id"), row.get("condition")) for row in all_rows]
    checks = {
        "four_shards_exist": all(path.is_file() for path in paths),
        "exact_291_rows": len(all_rows) == 291,
        "unique_sample_condition": len(set(keys)) == len(keys) == 291,
        "condition_exact": all(row.get("condition") == "AVCD_Raw_AV" for row in all_rows),
        "manifest_hash_exact": all(row.get("manifest_sha256") == EXPECTED_MANIFEST for row in all_rows),
        "runner_hash_exact": all(row.get("runner_sha256") == EXPECTED_RUNNER for row in all_rows),
        "official_commit_exact": all(row.get("avcd_official_revision") == EXPECTED_AVCD for row in all_rows),
        "parse_failure_policy": all(not row["correct"] for row in all_rows if row.get("parse_status") == "parse_failure"),
    }
    result = {
        "status": "PASS_AVCD_COMPLETE_BEFORE_STANDARD" if all(checks.values()) else "FAIL",
        "checks": checks,
        "shard_counts": {path.name: len(rows(path)) if path.is_file() else 0 for path in paths},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
