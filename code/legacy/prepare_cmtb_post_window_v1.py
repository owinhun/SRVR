#!/usr/bin/env python3
"""Test a post-narration window: [vad_end, vad_end + span] instead of [vad_start, vad_end].

Motivation: residual54 visual labeling repeatedly found narration describing
an action that had *already finished* by the time it's spoken, with the
correct current-visual action appearing right after the VAD interval ends.
Forward-padding (end+2s/+4s) gave a wash (fixed ~7-8, broke ~6-11). This
variant tests a stronger hypothesis: shift the whole window to start where
the original ends, same duration, to see if "right after speech" is a better
anchor than "during speech" for this backbone/dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT_DIR = ROOT / "claude_track/outputs"
AUDIT = OUT_DIR / "cmtb_post_window_prepare_audit_v1.json"
BASE_CONFIG = ROOT / "configs/models/video_salmonn2_matched_random3_v1.json"
GPUS = [0, 1]
EXPECTED_CMTB = 291
CONDITION = "VAD_post_window"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shard(sample_id: str, n_shards: int) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % n_shards


def trainer_row(row: dict, start: float, end: float) -> dict:
    return {
        "id": f"{row['sample_id']}__{CONDITION}",
        "sample_id": row["sample_id"], "task": row["task"], "condition": CONDITION,
        "video": row["video_path"], "timestamps": [start, end],
        "conversations": [
            {"from": "human", "value": "<image>\n" + row["question"]},
            {"from": "gpt", "value": row["answer"]},
        ],
    }


def mapping_row(row: dict, start: float, end: float) -> dict:
    return {
        "sample_id": row["sample_id"], "task": row["task"], "condition": CONDITION,
        "answer": row["answer"], "video_id": row["video_id"], "video_path": row["video_path"],
        "question": row["question"], "clip_start_sec": start, "clip_end_sec": end,
    }


def main() -> None:
    manifest_rows = [row for row in read_jsonl(MANIFEST) if row["task"] == "CMTB"]
    if len(manifest_rows) != EXPECTED_CMTB:
        raise RuntimeError(f"Frozen CMTB coverage mismatch: {len(manifest_rows)} != {EXPECTED_CMTB}")
    cues = {row["video_id"]: row for row in read_jsonl(CUES)}

    n_shards = len(GPUS)
    shard_buckets: list[list[tuple]] = [[] for _ in range(n_shards)]
    clamped = 0
    for row in sorted(manifest_rows, key=lambda item: item["official_index"]):
        cue = cues[row["video_id"]]
        vs, ve = float(cue["speech_start_sec"]), float(cue["speech_end_sec"])
        duration = float(cue["media_duration_sec"])
        span = ve - vs
        start = ve
        end = ve + span
        if end > duration:
            end = duration
            clamped += 1
        b = shard(row["sample_id"], n_shards)
        shard_buckets[b].append((row, start, end))

    base_config = json.loads(BASE_CONFIG.read_text())
    launch_commands = []
    for shard_id in range(n_shards):
        inputs = [trainer_row(r, s, e) for r, s, e in shard_buckets[shard_id]]
        maps = [mapping_row(r, s, e) for r, s, e in shard_buckets[shard_id]]
        input_path = OUT_DIR / f"cmtb_post_window_v1_shard{shard_id}.json"
        map_path = OUT_DIR / f"cmtb_post_window_v1_shard{shard_id}_map.jsonl"
        input_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")
        map_path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in maps))
        config = dict(base_config)
        config.update({
            "output_dir": str(OUT_DIR / f"cmtb_post_window_v1_trainer_shard{shard_id}"),
            "test_data_path": str(input_path),
            "test_output_dir": str(OUT_DIR / f"cmtb_post_window_v1_output_shard{shard_id}"),
            "audio_visual": True, "dataloader_num_workers": 4,
        })
        config_path = OUT_DIR / f"cmtb_post_window_v1_shard{shard_id}_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        gpu = GPUS[shard_id]
        port = 29630 + shard_id
        log_path = OUT_DIR / f"cmtb_post_window_v1_shard{shard_id}.log"
        launch_commands.append(
            f"CUDA_VISIBLE_DEVICES={gpu} torchrun --nproc_per_node=1 --master_port={port} "
            f"src/qwen/run_video_salmonn2_config.py {config_path} > {log_path} 2>&1; "
            f"echo exit=$? >> {log_path}"
        )

    audit = {"design": "CMTB post-narration window [vad_end, vad_end+span]", "clamped_at_duration": clamped,
              "launch_commands": launch_commands}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(AUDIT.read_text())


if __name__ == "__main__":
    main()
