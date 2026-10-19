#!/usr/bin/env python3
"""Prepare a prediction-blind full CMTB-291 manifest for official AVCD."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
SOURCE = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT = ROOT / "outputs/videollama21_avcd_cmtb291_manifest_v1.jsonl"
AUDIT = ROOT / "outputs/videollama21_avcd_cmtb291_manifest_audit_v1.json"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_rows = [row for row in rows(SOURCE) if row.get("task") == "CMTB"]
    cues = {row["video_id"]: row for row in rows(CUES)}
    if len(source_rows) != 291 or len({row["sample_id"] for row in source_rows}) != 291:
        raise RuntimeError("locked CMTB-291 source mismatch")
    if len({row["video_id"] for row in source_rows}) != 291:
        raise RuntimeError("CMTB-291 must be video unique")

    manifest = []
    errors = []
    for row in sorted(source_rows, key=lambda value: value["sample_id"]):
        cue = cues.get(row["video_id"])
        path = Path(row["video_path"])
        if not path.is_file():
            errors.append(f"{row['sample_id']}: missing video")
            continue
        if cue is None or cue.get("vad_status") != "ok" or cue.get("asr_status") != "ok":
            errors.append(f"{row['sample_id']}: frozen cue QC failure")
            continue
        manifest.append({
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "video_path": row["video_path"],
            "task": "CMTB",
            "question": row["question"],
            "answer": row["answer"],
            "qc_pass": True,
            "qc_definition": "locked_CMTB291+source_exists+frozen_cue_vad_ok+asr_ok",
            "selection_rule": (
                "all 291 locked CMTB evaluation rows; no prediction, correctness, "
                "audio-perturbation QC, or model output used"
            ),
            "official_index": row["official_index"],
            "source_file_sha256": cue["source_file_sha256"],
        })
    if errors or len(manifest) != 291:
        raise RuntimeError({"errors": errors, "manifest_n": len(manifest)})

    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest))
    forbidden = {"prediction", "correct", "raw_output", "model_answer", "accuracy"}
    checks = {
        "source_291_unique": len(source_rows) == len({row["sample_id"] for row in source_rows}) == 291,
        "manifest_291_unique": len(manifest) == len({row["sample_id"] for row in manifest}) == 291,
        "video_unique_291": len({row["video_id"] for row in manifest}) == 291,
        "all_source_files_exist": all(Path(row["video_path"]).is_file() for row in manifest),
        "all_qc_pass": all(row["qc_pass"] is True for row in manifest),
        "no_outcome_fields": all(not (forbidden & set(row)) for row in manifest),
        "answer_distribution_complete": sum(Counter(row["answer"] for row in manifest).values()) == 291,
    }
    audit = {
        "status": "PASS_FROZEN_BEFORE_INFERENCE" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n": len(manifest),
        "answer_distribution": dict(Counter(row["answer"] for row in manifest)),
        "selection_rule": manifest[0]["selection_rule"],
        "important_change_from_dev44": (
            "Official AVCD consumes the original source video directly. The historical "
            "audio-perturbation QC that reduced Dev-48 to Dev-44 is not an AVCD input requirement."
        ),
        "input_sha256": {SOURCE.name: sha(SOURCE), CUES.name: sha(CUES)},
        "manifest_sha256": sha(OUT),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if audit["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
