#!/usr/bin/env python3
"""Prepare TA+TF SRVR replication, v2: one job per CONDITION (not combined).

v1 combined all conditions for a sample into a single test_data_path file.
The vendored video-SALMONN-2 harness (llava/train/train.py) deduplicates
predict() results by key = str(id) + prompt_text, where id is derived only
from (video_path, audio_path) -- NOT from our "timestamps" field. Since
Video_only / VAD_localized_clip / Matched_random_* share the same video,
same (absent) audio, and same question text, they collided and only one
survived per sample. Confirmed by inspecting raw shard test_results.json
(v1 shard0: 166/429 rows survived) and the merge loop in train.py (lines
~561-573: `if key_id not in temp_dict: new_res.append(it)`).

Fix: run each condition as its own separate job/test_data_path. Within one
condition, every TA/TF sample maps 1:1 to a unique video (confirmed earlier:
TA unique_videos==323==total, TF unique_videos==247==total), so no two rows
in a single-condition job ever share (video, audio, question) -- no collision.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT_DIR = ROOT / "claude_track/outputs"
AUDIT = OUT_DIR / "ta_tf_srvr_prepare_audit_v2.json"
BASE_CONFIG = ROOT / "configs/models/video_salmonn2_matched_random3_v1.json"
SEEDS = (20260805, 20260806, 20260807)
GPUS = [0, 1, 2, 3, 5, 6, 7]
SHARDS = len(GPUS)
TASKS = {"TA", "TF"}
EXPECTED_COUNTS = {"TA": 323, "TF": 247}
CONDITIONS = ["Raw_AV", "Video_only", "VAD_localized_clip", "Matched_random_0"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shard(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % SHARDS


def strict_span(sample_id, duration, start, end, length, seed):
    ranges: list[tuple[float, float]] = []
    if start >= length:
        ranges.append((0.0, start - length))
    if duration - end >= length:
        ranges.append((end, duration - length))
    if not ranges:
        return None
    rng = random.Random(f"{seed}:{sample_id}")
    lo, hi = ranges[rng.randrange(len(ranges))]
    return lo if hi <= lo else rng.uniform(lo, hi)


def trainer_row(row: dict, condition: str, timestamps=None) -> dict:
    item = {
        "id": f"{row['sample_id']}__{condition}",
        "sample_id": row["sample_id"],
        "task": row["task"],
        "condition": condition,
        "video": row["video_path"],
        "conversations": [
            {"from": "human", "value": "<image>\n" + row["question"]},
            {"from": "gpt", "value": row["answer"]},
        ],
    }
    if condition == "Raw_AV":
        item["audio"] = row["video_path"]
    if timestamps is not None:
        item["timestamps"] = timestamps
    return item


def mapping_row(row, condition, cue, *, timestamps=None, random_seed=None) -> dict:
    return {
        "sample_id": row["sample_id"],
        "task": row["task"],
        "condition": condition,
        "answer": row["answer"],
        "video_id": row["video_id"],
        "video_path": row["video_path"],
        "question": row["question"],
        "clip_start_sec": timestamps[0] if timestamps else None,
        "clip_end_sec": timestamps[1] if timestamps else None,
        "true_vad_start_sec": cue["speech_start_sec"],
        "true_vad_end_sec": cue["speech_end_sec"],
        "duration_sec": cue["media_duration_sec"],
        "random_seed": random_seed,
        "source_file_sha256": cue["source_file_sha256"],
    }


def main() -> None:
    manifest_rows = [row for row in read_jsonl(MANIFEST) if row["task"] in TASKS]
    counts = Counter(row["task"] for row in manifest_rows)
    for task, expected in EXPECTED_COUNTS.items():
        if counts.get(task) != expected:
            raise RuntimeError(f"Frozen {task} coverage mismatch: {counts.get(task)} != {expected}")
    cues = {row["video_id"]: row for row in read_jsonl(CUES)}

    # Build one row per (sample, condition), keyed by condition first.
    per_condition_rows: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    per_condition_maps: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    strict_feasible, strict_infeasible = [], []

    for row in sorted(manifest_rows, key=lambda item: item["official_index"]):
        cue = cues.get(row["video_id"])
        if cue is None:
            raise RuntimeError(f"Missing frozen cue for {row['video_id']}")
        start, end = float(cue["speech_start_sec"]), float(cue["speech_end_sec"])
        duration = float(cue["media_duration_sec"])
        qc_pass = (
            cue.get("vad_status") == "ok"
            and cue.get("asr_status") == "ok"
            and 0.0 <= start < end <= duration + 1e-6
            and bool(cue.get("source_file_sha256"))
        )
        if not qc_pass:
            raise RuntimeError(f"Non-qc-pass cue: {row['sample_id']}")

        for condition, timestamps in (
            ("Raw_AV", None),
            ("Video_only", None),
            ("VAD_localized_clip", [start, end]),
        ):
            per_condition_rows[condition].append(trainer_row(row, condition, timestamps))
            per_condition_maps[condition].append(mapping_row(row, condition, cue, timestamps=timestamps))

        length = end - start
        random_start = strict_span(row["sample_id"], duration, start, end, length, SEEDS[0])
        if random_start is not None:
            strict_feasible.append(row["sample_id"])
            timestamps = [random_start, random_start + length]
            per_condition_rows["Matched_random_0"].append(trainer_row(row, "Matched_random_0", timestamps))
            per_condition_maps["Matched_random_0"].append(
                mapping_row(row, "Matched_random_0", cue, timestamps=timestamps, random_seed=SEEDS[0])
            )
        else:
            strict_infeasible.append(row["sample_id"])

    base_config = json.loads(BASE_CONFIG.read_text())
    condition_shard_counts = {}
    launch_commands = []

    for condition in CONDITIONS:
        rows = per_condition_rows[condition]
        maps = per_condition_maps[condition]
        # Shard this condition's rows across the GPU pool.
        shard_buckets: list[list[dict]] = [[] for _ in range(SHARDS)]
        map_buckets: list[list[dict]] = [[] for _ in range(SHARDS)]
        for row, m in zip(rows, maps):
            b = shard(row["sample_id"])
            shard_buckets[b].append(row)
            map_buckets[b].append(m)

        shard_counts = []
        for shard_id in range(SHARDS):
            tag = f"{condition.lower()}"
            input_path = OUT_DIR / f"ta_tf_srvr_v2_{tag}_shard{shard_id}.json"
            map_path = OUT_DIR / f"ta_tf_srvr_v2_{tag}_shard{shard_id}_map.jsonl"
            input_path.write_text(json.dumps(shard_buckets[shard_id], ensure_ascii=False, indent=2) + "\n")
            map_path.write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in map_buckets[shard_id])
            )
            config = dict(base_config)
            config.update({
                "output_dir": str(OUT_DIR / f"ta_tf_srvr_v2_{tag}_trainer_shard{shard_id}"),
                "test_data_path": str(input_path),
                "test_output_dir": str(OUT_DIR / f"ta_tf_srvr_v2_{tag}_output_shard{shard_id}"),
                "audio_visual": True,
                "dataloader_num_workers": 4,
            })
            config_path = OUT_DIR / f"ta_tf_srvr_v2_{tag}_shard{shard_id}_config.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            shard_counts.append(len(shard_buckets[shard_id]))
            gpu = GPUS[shard_id]
            port = 29520 + CONDITIONS.index(condition) * 10 + shard_id
            log_path = OUT_DIR / f"ta_tf_srvr_v2_{tag}_shard{shard_id}.log"
            launch_commands.append(
                f"CUDA_VISIBLE_DEVICES={gpu} torchrun --nproc_per_node=1 --master_port={port} "
                f"src/qwen/run_video_salmonn2_config.py {config_path} > {log_path} 2>&1; "
                f"echo exit=$? >> {log_path}"
            )
        condition_shard_counts[condition] = shard_counts

    audit = {
        "design": "TA+TF SRVR v2: one separate job per condition to avoid merge-dedup collision",
        "manifest_sha256": sha256(MANIFEST),
        "cues_sha256": sha256(CUES),
        "task_samples": dict(counts),
        "total_samples": len(manifest_rows),
        "strict_feasible_samples": len(strict_feasible),
        "strict_infeasible_count": len(strict_infeasible),
        "conditions": CONDITIONS,
        "condition_row_totals": {c: len(per_condition_rows[c]) for c in CONDITIONS},
        "condition_shard_counts": condition_shard_counts,
        "gpu_mapping": dict(zip(range(SHARDS), GPUS)),
        "launch_commands": launch_commands,
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(AUDIT.read_text())


if __name__ == "__main__":
    main()
