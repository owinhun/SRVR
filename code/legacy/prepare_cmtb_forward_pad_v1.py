#!/usr/bin/env python3
"""Test whether extending the VAD interval forward in time recovers residual errors.

Motivation: LLM-vision-based labeling of the 54 SRVR residual errors found that
30% (16/54) show a "narration lag" pattern -- the speech heard during the VAD
window describes an action that was already finishing, while the visually
correct answer (the NEXT step) appears just after vad_end, in the padding
zone. This tests two forward-padding variants of VAD_localized_clip:
end + 2s and end + 4s (start unchanged), on the full CMTB-291 set, to see if
this simple boundary adjustment recovers accuracy beyond 81.4% baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT_DIR = ROOT / "claude_track/outputs"
AUDIT = OUT_DIR / "cmtb_forward_pad_prepare_audit_v1.json"
BASE_CONFIG = ROOT / "configs/models/video_salmonn2_matched_random3_v1.json"
GPUS = [0, 1, 2, 3]
EXPECTED_CMTB = 291
PADS = {"pad2s": 2.0, "pad4s": 4.0}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shard(sample_id: str, n_shards: int) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % n_shards


def trainer_row(row: dict, condition: str, start: float, end: float) -> dict:
    return {
        "id": f"{row['sample_id']}__{condition}",
        "sample_id": row["sample_id"],
        "task": row["task"],
        "condition": condition,
        "video": row["video_path"],
        "timestamps": [start, end],
        "conversations": [
            {"from": "human", "value": "<image>\n" + row["question"]},
            {"from": "gpt", "value": row["answer"]},
        ],
    }


def mapping_row(row: dict, condition: str, start: float, end: float) -> dict:
    return {
        "sample_id": row["sample_id"], "task": row["task"], "condition": condition,
        "answer": row["answer"], "video_id": row["video_id"], "video_path": row["video_path"],
        "question": row["question"], "clip_start_sec": start, "clip_end_sec": end,
    }


def main() -> None:
    manifest_rows = [row for row in read_jsonl(MANIFEST) if row["task"] == "CMTB"]
    if len(manifest_rows) != EXPECTED_CMTB:
        raise RuntimeError(f"Frozen CMTB coverage mismatch: {len(manifest_rows)} != {EXPECTED_CMTB}")
    cues = {row["video_id"]: row for row in read_jsonl(CUES)}

    base_config = json.loads(BASE_CONFIG.read_text())
    launch_commands = []
    n_shards = 2  # per condition, 2 shards -> 4 total jobs across 4 GPUs

    for tag, pad in PADS.items():
        condition = f"VAD_localized_clip_{tag}"
        shard_buckets: list[list[dict]] = [[] for _ in range(n_shards)]
        for row in sorted(manifest_rows, key=lambda item: item["official_index"]):
            cue = cues[row["video_id"]]
            start = float(cue["speech_start_sec"])
            end = min(float(cue["media_duration_sec"]), float(cue["speech_end_sec"]) + pad)
            b = shard(row["sample_id"], n_shards)
            shard_buckets[b].append((row, start, end))

        for shard_id in range(n_shards):
            inputs = [trainer_row(r, condition, s, e) for r, s, e in shard_buckets[shard_id]]
            maps = [mapping_row(r, condition, s, e) for r, s, e in shard_buckets[shard_id]]
            input_path = OUT_DIR / f"cmtb_{tag}_v1_shard{shard_id}.json"
            map_path = OUT_DIR / f"cmtb_{tag}_v1_shard{shard_id}_map.jsonl"
            input_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2) + "\n")
            map_path.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in maps))
            config = dict(base_config)
            config.update({
                "output_dir": str(OUT_DIR / f"cmtb_{tag}_v1_trainer_shard{shard_id}"),
                "test_data_path": str(input_path),
                "test_output_dir": str(OUT_DIR / f"cmtb_{tag}_v1_output_shard{shard_id}"),
                "audio_visual": True, "dataloader_num_workers": 4,
            })
            config_path = OUT_DIR / f"cmtb_{tag}_v1_shard{shard_id}_config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            gpu = GPUS[(PADS.__len__() and list(PADS).index(tag)) * n_shards + shard_id]
            port = 29610 + list(PADS).index(tag) * 10 + shard_id
            log_path = OUT_DIR / f"cmtb_{tag}_v1_shard{shard_id}.log"
            launch_commands.append(
                f"CUDA_VISIBLE_DEVICES={gpu} torchrun --nproc_per_node=1 --master_port={port} "
                f"src/qwen/run_video_salmonn2_config.py {config_path} > {log_path} 2>&1; "
                f"echo exit=$? >> {log_path}"
            )

    audit = {"design": "CMTB VAD forward-padding: end+2s, end+4s", "launch_commands": launch_commands}
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(AUDIT.read_text())


if __name__ == "__main__":
    main()
