#!/usr/bin/env python3
"""Prepare the frozen four-GPU video-SALMONN-2 CMTB-291 closing experiment."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUT_PREFIX = ROOT / "outputs/video_salmonn2_cmtb291_v1"
AUDIT = ROOT / "outputs/video_salmonn2_cmtb291_prepare_audit_v1.json"
BASE_CONFIG = ROOT / "configs/models/video_salmonn2_matched_random3_v1.json"
SEEDS = (20260805, 20260806, 20260807)
SHARDS = 4


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


def strict_span(
    sample_id: str,
    duration: float,
    start: float,
    end: float,
    length: float,
    seed: int,
) -> float | None:
    """Match the already-frozen Dev-44 strict control implementation exactly."""
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


def trainer_row(row: dict, condition: str, cue: dict, timestamps=None) -> dict:
    item = {
        "id": f"{row['sample_id']}__{condition}",
        "sample_id": row["sample_id"],
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


def mapping_row(
    row: dict,
    condition: str,
    cue: dict,
    *,
    timestamps=None,
    primary: bool,
    random_index: int | None = None,
    random_seed: int | None = None,
    random_rule: str | None = None,
) -> dict:
    return {
        "sample_id": row["sample_id"],
        "input_id": f"{row['sample_id']}__{condition}",
        "condition": condition,
        "answer": row["answer"],
        "video_id": row["video_id"],
        "video_path": row["video_path"],
        "question": row["question"],
        "qc_pass": True,
        "qc_definition": "frozen_cue_vad_ok+asr_ok+valid_bounded_span+source_sha256_present",
        "analysis_subset": "primary" if primary else "sensitivity",
        "clip_start_sec": timestamps[0] if timestamps else None,
        "clip_end_sec": timestamps[1] if timestamps else None,
        "true_vad_start_sec": cue["speech_start_sec"],
        "true_vad_end_sec": cue["speech_end_sec"],
        "duration_sec": cue["media_duration_sec"],
        "random_index": random_index,
        "random_seed": random_seed,
        "random_rule": random_rule,
        "source_file_sha256": cue["source_file_sha256"],
    }


def main() -> None:
    manifest_rows = [row for row in read_jsonl(MANIFEST) if row["task"] == "CMTB"]
    cues = {row["video_id"]: row for row in read_jsonl(CUES)}
    if len(manifest_rows) != 291 or len({row["sample_id"] for row in manifest_rows}) != 291:
        raise RuntimeError("Frozen CMTB-291 coverage mismatch")

    inputs: list[list[dict]] = [[] for _ in range(SHARDS)]
    maps: list[list[dict]] = [[] for _ in range(SHARDS)]
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
            raise RuntimeError(f"Non-qc-pass CMTB cue: {row['sample_id']}")

        sid_shard = shard(row["sample_id"])
        fixed = (
            ("Raw_AV", None),
            ("Video_only", None),
            ("VAD_localized_clip", [start, end]),
        )
        for condition, timestamps in fixed:
            inputs[sid_shard].append(trainer_row(row, condition, cue, timestamps))
            maps[sid_shard].append(
                mapping_row(
                    row, condition, cue, timestamps=timestamps, primary=True
                )
            )

        length = end - start
        random_starts = [
            strict_span(row["sample_id"], duration, start, end, length, seed)
            for seed in SEEDS
        ]
        if all(value is not None for value in random_starts):
            strict_feasible.append(row["sample_id"])
            for index, (seed, random_start) in enumerate(zip(SEEDS, random_starts)):
                assert random_start is not None
                timestamps = [random_start, random_start + length]
                condition = f"Matched_random_{index}"
                inputs[sid_shard].append(trainer_row(row, condition, cue, timestamps))
                maps[sid_shard].append(
                    mapping_row(
                        row,
                        condition,
                        cue,
                        timestamps=timestamps,
                        primary=True,
                        random_index=index,
                        random_seed=seed,
                        random_rule="same-duration_strict-nonoverlap_sample-seeded",
                    )
                )
        else:
            strict_infeasible.append(row["sample_id"])
            # Frozen during the earlier TER study. It is deliberately excluded
            # from the primary matched-random comparison because it may overlap.
            timestamps = [float(cue["random_start_sec"]), float(cue["random_end_sec"])]
            condition = "Matched_random_sensitivity_frozen"
            inputs[sid_shard].append(trainer_row(row, condition, cue, timestamps))
            maps[sid_shard].append(
                mapping_row(
                    row,
                    condition,
                    cue,
                    timestamps=timestamps,
                    primary=False,
                    random_rule="frozen-ter-random-fallback-sensitivity-only",
                )
            )

    base_config = json.loads(BASE_CONFIG.read_text())
    shard_counts = []
    for shard_id in range(SHARDS):
        input_path = Path(f"{OUT_PREFIX}_shard{shard_id}.json")
        map_path = Path(f"{OUT_PREFIX}_shard{shard_id}_map.jsonl")
        input_path.write_text(json.dumps(inputs[shard_id], ensure_ascii=False, indent=2) + "\n")
        map_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in maps[shard_id])
        )
        config = dict(base_config)
        config.update(
            {
                "output_dir": str(ROOT / f"outputs/video_salmonn2_cmtb291_trainer_v1_shard{shard_id}"),
                "test_data_path": str(input_path),
                "test_output_dir": str(ROOT / f"outputs/video_salmonn2_cmtb291_output_v1_shard{shard_id}"),
                "audio_visual": True,
                "dataloader_num_workers": 4,
            }
        )
        config_path = ROOT / f"configs/models/video_salmonn2_cmtb291_v1_shard{shard_id}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        shard_counts.append(len(inputs[shard_id]))

    audit = {
        "design": "CMTB-291 Raw AV, video-only, exact VAD clip, and 3 matched-random replicates",
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST),
        "cues": str(CUES),
        "cues_sha256": sha256(CUES),
        "cmtb_samples": len(manifest_rows),
        "qc_pass_samples": len(manifest_rows),
        "qc_definition": "frozen_cue_vad_ok+asr_ok+valid_bounded_span+source_sha256_present",
        "strict_feasible_samples": len(strict_feasible),
        "strict_infeasible_samples": strict_infeasible,
        "random_seeds": list(SEEDS),
        "primary_random_policy": "same duration and complete non-overlap; all three sample-seeded replicates",
        "sensitivity_policy": "pre-existing frozen TER random interval for strict-infeasible samples; excluded from primary",
        "shard_rule": "sha256(sample_id) mod 4; all conditions for a video stay together",
        "shard_input_rows": shard_counts,
        "condition_rows": dict(
            sorted(Counter(item["condition"] for group in maps for item in group).items())
        ),
        "prompt_policy": "verbatim frozen question and option order; no CMTB-291 tuning",
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(AUDIT.read_text())


if __name__ == "__main__":
    main()
