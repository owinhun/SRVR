#!/usr/bin/env python3
"""Freeze the full CMTB-291 manifest for standard VideoLLaMA2.1 conditions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
BASE = ROOT / "outputs/videollama21_avcd_cmtb291_manifest_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
AUDIT = ROOT / "outputs/videollama21_standard_cmtb291_manifest_audit_v1.json"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = rows(BASE)
    cues = {row["video_id"]: row for row in rows(CUES)}
    if not (len(base) == len({row["sample_id"] for row in base}) == 291):
        raise RuntimeError("base manifest mismatch")

    manifest = []
    for row in base:
        cue = cues.get(row["video_id"])
        if cue is None or cue.get("vad_status") != "ok":
            raise RuntimeError(f"missing frozen automatic VAD cue: {row['sample_id']}")
        item = dict(row)
        item.update({
            "speech_start_sec": cue["speech_start_sec"],
            "speech_end_sec": cue["speech_end_sec"],
            "media_duration_sec": cue["media_duration_sec"],
            "vad_definition": (
                "frozen automatic Silero VAD interval: first detected speech start "
                "through last detected speech end; no outcome-dependent tuning"
            ),
            "vad_config": cue["vad_config"],
            "vad_model_revision": cue["vad_model_revision"],
            "cue_generation_code_sha256": cue["cue_generation_code_sha256"],
        })
        manifest.append(item)

    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    forbidden = {"prediction", "correct", "raw_output", "model_answer", "accuracy"}
    checks = {
        "base_manifest_already_frozen": BASE.is_file(),
        "manifest_291_unique": len(manifest) == len({row["sample_id"] for row in manifest}) == 291,
        "video_unique_291": len({row["video_id"] for row in manifest}) == 291,
        "same_order_as_avcd": [row["sample_id"] for row in manifest] == [row["sample_id"] for row in base],
        "all_source_files_exist": all(Path(row["video_path"]).is_file() for row in manifest),
        "all_intervals_valid": all(
            0 <= row["speech_start_sec"] < row["speech_end_sec"] <= row["media_duration_sec"] + 0.05
            for row in manifest
        ),
        "no_outcome_fields": all(not (forbidden & set(row)) for row in manifest),
        "answer_distribution_complete": sum(Counter(row["answer"] for row in manifest).values()) == 291,
    }
    audit = {
        "status": "PASS_FROZEN_BEFORE_STANDARD_INFERENCE" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n": len(manifest),
        "answer_distribution": dict(Counter(row["answer"] for row in manifest)),
        "conditions_frozen": ["Raw_AV", "Video_only", "VAD_localized_clip"],
        "comparison_contract": (
            "same population, source video, question/options/order, checkpoint, parser, "
            "and deterministic decoding; only media interface differs by condition"
        ),
        "input_sha256": {BASE.name: sha(BASE), CUES.name: sha(CUES)},
        "manifest_sha256": sha(OUT),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if audit["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
