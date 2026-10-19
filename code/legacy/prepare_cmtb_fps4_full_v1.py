#!/usr/bin/env python3
"""Full CMTB-291 VAD_localized_clip at video_fps=4 (vs baseline fps=1).

The residual-54-only fps4 test recovered 11/54 (20.4%) vs fps2's 4/54,
suggesting denser sampling helps MULTI_ACTION_INTERVAL-type failures. This
runs the full 291-sample set at fps=4 to check whether it's a net gain over
the 81.4% fps=1 baseline, or a wash like forward-padding was (which also
looked promising on the residual subset before breaking other samples).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT_DIR = ROOT / "claude_track/outputs"
BASE_CONFIG = ROOT / "configs/models/video_salmonn2_matched_random3_v1.json"
GPUS = [0, 1, 2, 3]
EXPECTED_CMTB = 291
CONDITION = "VAD_localized_clip_fps4"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def shard(sample_id: str, n_shards: int) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % n_shards


def trainer_row(row: dict, start: float, end: float) -> dict:
    return {
        "id": f"{row['sample_id']}__{CONDITION}", "sample_id": row["sample_id"],
        "task": row["task"], "condition": CONDITION, "video": row["video_path"],
        "timestamps": [start, end],
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
    for row in sorted(manifest_rows, key=lambda item: item["official_index"]):
        cue = cues[row["video_id"]]
        start, end = float(cue["speech_start_sec"]), float(cue["speech_end_sec"])
        b = shard(row["sample_id"], n_shards)
        shard_buckets[b].append((row, start, end))

    base_config = json.loads(BASE_CONFIG.read_text())
    launch_commands = []
    for shard_id in range(n_shards):
        inputs = [trainer_row(r, s, e) for r, s, e in shard_buckets[shard_id]]
        maps = [mapping_row(r, s, e) for r, s, e in shard_buckets[shard_id]]
        input_path = OUT_DIR / f"cmtb_fps4_v1_shard{shard_id}.json"
        map_path = OUT_DIR / f"cmtb_fps4_v1_shard{shard_id}_map.jsonl"
        input_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")
        map_path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in maps))
        config = dict(base_config)
        config.update({
            "output_dir": str(OUT_DIR / f"cmtb_fps4_v1_trainer_shard{shard_id}"),
            "test_data_path": str(input_path),
            "test_output_dir": str(OUT_DIR / f"cmtb_fps4_v1_output_shard{shard_id}"),
            "audio_visual": True, "dataloader_num_workers": 4, "video_fps": 4,
        })
        config_path = OUT_DIR / f"cmtb_fps4_v1_shard{shard_id}_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        gpu = GPUS[shard_id]
        port = 29670 + shard_id
        log_path = OUT_DIR / f"cmtb_fps4_v1_shard{shard_id}.log"
        launch_commands.append(
            f"CUDA_VISIBLE_DEVICES={gpu} torchrun --nproc_per_node=1 --master_port={port} "
            f"src/qwen/run_video_salmonn2_config.py {config_path} > {log_path} 2>&1; "
            f"echo exit=$? >> {log_path}"
        )
    print(json.dumps({"launch_commands": launch_commands}, indent=2))


if __name__ == "__main__":
    main()
