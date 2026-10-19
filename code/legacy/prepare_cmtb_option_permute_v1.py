#!/usr/bin/env python3
"""CMTB option-permutation robustness check.

Already confirmed: VAD_localized_clip accuracy improves uniformly across all
three ANSWER letters (A:52->84%, B:56->85%, C:30->76%), which argues against
a position-tracking artifact. This is a stronger, independent check: cyclically
rotate the option TEXT (whatever was at A moves to C, B moves to A, C moves to
B) while keeping the same VAD-localized video clip evidence. If accuracy on
this permuted CMTB collapses toward chance or drops sharply, the original
81.4% would be suspect. If it stays high, that's strong confirmation the
model is reading option content, not exploiting a position/label shortcut.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT_DIR = ROOT / "claude_track/outputs"
AUDIT = OUT_DIR / "cmtb_option_permute_prepare_audit_v1.json"
BASE_CONFIG = ROOT / "configs/models/video_salmonn2_matched_random3_v1.json"
GPUS = [0, 1]
SHARDS = len(GPUS)
EXPECTED_CMTB = 291
CONDITION = "VAD_localized_clip_option_permute"

# text at old letter moves to this new letter
ROTATE = {"A": "C", "B": "A", "C": "B"}
OPTION_RE = re.compile(r"\(A\)\s*(.*?)\s*\(B\)\s*(.*?)\s*\(C\)\s*(.*)$")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shard(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % SHARDS


def permute_question(question: str) -> tuple[str, dict]:
    m = OPTION_RE.search(question)
    if not m:
        raise ValueError(f"Could not parse options: {question}")
    a, b, c = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    prefix = question[: m.start()]
    # new_A = old_B text, new_B = old_C text, new_C = old_A text
    new_q = f"{prefix}(A) {b} (B) {c} (C) {a}"
    return new_q, {"A": b, "B": c, "C": a}


def trainer_row(row: dict, new_question: str, new_answer: str, start: float, end: float) -> dict:
    return {
        "id": f"{row['sample_id']}__{CONDITION}",
        "sample_id": row["sample_id"],
        "task": row["task"],
        "condition": CONDITION,
        "video": row["video_path"],
        "timestamps": [start, end],
        "conversations": [
            {"from": "human", "value": "<image>\n" + new_question},
            {"from": "gpt", "value": new_answer},
        ],
    }


def mapping_row(row: dict, new_question: str, new_answer: str, start: float, end: float) -> dict:
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "condition": CONDITION,
        "original_answer": row["answer"],
        "answer": new_answer,
        "video_id": row["video_id"],
        "video_path": row["video_path"],
        "question": new_question,
        "original_question": row["question"],
        "clip_start_sec": start,
        "clip_end_sec": end,
    }


def main() -> None:
    manifest_rows = [row for row in read_jsonl(MANIFEST) if row["task"] == "CMTB"]
    if len(manifest_rows) != EXPECTED_CMTB:
        raise RuntimeError(f"Frozen CMTB coverage mismatch: {len(manifest_rows)} != {EXPECTED_CMTB}")
    cues = {row["video_id"]: row for row in read_jsonl(CUES)}

    shard_buckets: list[list[dict]] = [[] for _ in range(SHARDS)]
    map_buckets: list[list[dict]] = [[] for _ in range(SHARDS)]

    for row in sorted(manifest_rows, key=lambda item: item["official_index"]):
        cue = cues.get(row["video_id"])
        if cue is None:
            raise RuntimeError(f"Missing frozen cue for {row['video_id']}")
        start, end = float(cue["speech_start_sec"]), float(cue["speech_end_sec"])
        new_question, _ = permute_question(row["question"])
        new_answer = ROTATE[row["answer"]]

        b = shard(row["sample_id"])
        shard_buckets[b].append(trainer_row(row, new_question, new_answer, start, end))
        map_buckets[b].append(mapping_row(row, new_question, new_answer, start, end))

    base_config = json.loads(BASE_CONFIG.read_text())
    shard_counts = []
    launch_commands = []
    tag = "cmtb_option_permute"
    for shard_id in range(SHARDS):
        input_path = OUT_DIR / f"{tag}_v1_shard{shard_id}.json"
        map_path = OUT_DIR / f"{tag}_v1_shard{shard_id}_map.jsonl"
        input_path.write_text(json.dumps(shard_buckets[shard_id], ensure_ascii=False, indent=2) + "\n")
        map_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in map_buckets[shard_id])
        )
        config = dict(base_config)
        config.update({
            "output_dir": str(OUT_DIR / f"{tag}_v1_trainer_shard{shard_id}"),
            "test_data_path": str(input_path),
            "test_output_dir": str(OUT_DIR / f"{tag}_v1_output_shard{shard_id}"),
            "audio_visual": True,
            "dataloader_num_workers": 4,
        })
        config_path = OUT_DIR / f"{tag}_v1_shard{shard_id}_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        shard_counts.append(len(shard_buckets[shard_id]))
        gpu = GPUS[shard_id]
        port = 29600 + shard_id
        log_path = OUT_DIR / f"{tag}_v1_shard{shard_id}.log"
        launch_commands.append(
            f"CUDA_VISIBLE_DEVICES={gpu} torchrun --nproc_per_node=1 --master_port={port} "
            f"src/qwen/run_video_salmonn2_config.py {config_path} > {log_path} 2>&1; "
            f"echo exit=$? >> {log_path}"
        )

    audit = {
        "design": "CMTB VAD_localized_clip option-permute: A<-B, B<-C, C<-A text rotation",
        "condition": CONDITION,
        "total_samples": len(manifest_rows),
        "shard_counts": shard_counts,
        "gpu_mapping": dict(zip(range(SHARDS), GPUS)),
        "launch_commands": launch_commands,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(AUDIT.read_text())


if __name__ == "__main__":
    main()
