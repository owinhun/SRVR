#!/usr/bin/env python3
"""Build a seed-fixed, prediction-free CMTB-120 temporal-span manifest."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
CODEBOOK = ROOT / "paper/cmtb120_temporal_span_annotation_codebook_v1.md"
OUT = ROOT / "data/cmtb120_temporal_span_annotation_blind_v1.jsonl"
AUDIT = ROOT / "outputs/cmtb120_temporal_span_annotation_freeze_v1.json"
SALT = "cmtb120-independent-temporal-spans-v1-20260814"
N = 120


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def options(question: str) -> dict[str, str]:
    matches = list(re.finditer(r"\(([ABC])\)\s*", question))
    if len(matches) != 3:
        raise RuntimeError(f"Cannot parse options: {question}")
    out = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        out[match.group(1)] = question[match.end():end].strip()
    return out


def main() -> None:
    rows = [r for r in read(MANIFEST) if r["task"] == "CMTB"]
    cues = {r["video_id"]: r for r in read(CUES)}
    if len(rows) != len({r["sample_id"] for r in rows}) or len(rows) != 291:
        raise RuntimeError("Expected unique CMTB-291")
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{SALT}|{r['sample_id']}".encode()).hexdigest())
    selected = ordered[:N]
    output = []
    for index, row in enumerate(selected, start=1):
        cue = cues[row["video_id"]]
        output.append({
            "annotation_index": index,
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "media_path": row["video_path"],
            "media_duration_sec": cue["media_duration_sec"],
            "question": row["question"],
            "options": options(row["question"]),
            "official_current_answer": row["answer"],
            "asr_transcript": cue["asr_transcript"],
            "model_predictions_or_correctness_exposed": False,
            "human_annotation": None,
        })
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in output))
    audit = {
        "status": "FROZEN_READY_FOR_TWO_INDEPENDENT_HUMANS",
        "population_parent_n": 291, "selected_n": len(output),
        "selection": "first 120 by salted SHA-256 sample-id order",
        "salt": SALT, "prediction_fields_read": False,
        "hashes": {"manifest": sha(MANIFEST), "cues": sha(CUES), "codebook": sha(CODEBOOK), "blind_manifest": sha(OUT), "builder": sha(Path(__file__))},
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
