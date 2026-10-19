#!/usr/bin/env python3
"""Freeze the exact CMTB Dev-48 population for the VideoITG kill test.

This script uses no AVLLM prediction or correctness.  It joins the original
frozen Dev-48 membership with official SVHalluc metadata and source media.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
DEV_SOURCE = ROOT / "outputs/factorized_router_cmtb_dev48_asr_v1.jsonl"
OFFICIAL = ROOT / "data/SVHalluc/samples.json"
PROTOCOL = ROOT / "paper/videoitg_cmtb_strong_baseline_preflight_protocol_v1.md"
DOWNSTREAM = ROOT / "paper/videoitg_cmtb_downstream_freeze_v2.md"
OUT = ROOT / "outputs/videoitg_cmtb_dev48_manifest_v1.jsonl"
AUDIT = ROOT / "outputs/videoitg_cmtb_dev48_manifest_audit_v1.json"
EXPECTED_DEV_HASH = "2eace856f7dd1e4d5245f7c2fcf8cef1c02a157ed499cf343496d821e14dc2fb"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    dev_rows = read_jsonl(DEV_SOURCE)
    official = json.loads(OFFICIAL.read_text())
    ids = [row["sample_id"] for row in dev_rows]
    if len(ids) != len(set(ids)) or len(ids) != 48:
        raise RuntimeError("frozen CMTB development population is not unique N=48")

    rows = []
    for sample_id in ids:
        official_index = int(sample_id.rsplit("_", 1)[1])
        item = official[official_index]
        if item.get("task_type_abbr") != "CMTB":
            raise RuntimeError(f"not CMTB: {sample_id}")
        video = ROOT / "data/SVHalluc" / item["video_path"]
        if not video.exists():
            raise RuntimeError(f"missing video: {video}")
        # The selector sees exactly the official question/options but no answer.
        rows.append({
            "sample_id": sample_id,
            "official_index": official_index,
            "video_id": item["video_path"],
            "video_path": str(video),
            "video_sha256": sha(video),
            "selector_prompt": item["question"],
            "answer_evaluation_only": item["answer"],
            "selector_answer_blind": True,
            "selector_conditions": ["uniform_32", "videoitg_32"],
            "downstream_model": "OpenGVLab/InternVL2_5-8B",
            "downstream_revision": "e9e4c0dc1db56bfab10458671519b7fa3dd29463",
            "downstream_audio": "off",
        })

    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    video_ids = [row["video_id"] for row in rows]
    checks = {
        "exact_48_unique_sample_ids": len(rows) == len(set(ids)) == 48,
        "exact_48_unique_videos": len(set(video_ids)) == 48,
        "all_official_cmtb": all(official[row["official_index"]]["task_type_abbr"] == "CMTB" for row in rows),
        "all_media_exist": all(Path(row["video_path"]).exists() for row in rows),
        "selector_prompts_are_official_questions": all(
            row["selector_prompt"] == official[row["official_index"]]["question"] for row in rows
        ),
        "answers_are_evaluation_only": all(row["selector_answer_blind"] for row in rows),
        "original_frozen_dev_membership_hash_known": all(
            value.get("manifest_sha256") == EXPECTED_DEV_HASH
            for path in (ROOT / "outputs/qwen3_phase0.jsonl", ROOT / "outputs/qwen25_phase0.jsonl")
            for value in read_jsonl(path)
        ),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "population": "frozen video-disjoint CMTB Dev-48, development-only",
        "n": len(rows),
        "checks": checks,
        "inputs": {
            str(DEV_SOURCE.relative_to(ROOT)): sha(DEV_SOURCE),
            str(OFFICIAL.relative_to(ROOT)): sha(OFFICIAL),
            str(PROTOCOL.relative_to(ROOT)): sha(PROTOCOL),
            str(DOWNSTREAM.relative_to(ROOT)): sha(DOWNSTREAM),
        },
        "output": str(OUT.relative_to(ROOT)),
        "output_sha256": sha(OUT),
        "answer_exposure_contract": "answer_evaluation_only is forbidden from VideoITG selector and Phi prompt construction",
        "storage_contract": "checkpoint, cache, env, prepared inputs, and inference outputs remain outside /workspace",
    }
    AUDIT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
