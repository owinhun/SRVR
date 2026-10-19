#!/usr/bin/env python3
"""Answer-blind InternVL evaluation for frozen S2FS CMTB Dev-48 indices."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from run_internvl_videoitg_cmtb_dev48_v1 import (
    build_transform,
    load_frames,
    parse_choice,
    safe_manifest_rows,
    sha256,
    text_sha256,
)


CONDITIONS = (
    "full_uniform_8",
    "srvr_uniform_8",
    "visual_coverage_8",
    "semantic_attract_8",
    "signed_repulsion_8",
    "shuffled_repulsion_8",
)
FRAME_BUDGET = 8


def load_selector(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in rows:
                raise ValueError(f"duplicate selector row: {sample_id}")
            if row.get("status") != "PASS" or row.get("answer_used") is not False:
                raise ValueError(f"unsafe selector row: {sample_id}")
            if set(row["indices_by_condition"]) != set(CONDITIONS):
                raise ValueError(f"condition mismatch: {sample_id}")
            rows[sample_id] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    rows = safe_manifest_rows(args.manifest)
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    selector = load_selector(args.selector)
    missing = [row["sample_id"] for row in rows if row["sample_id"] not in selector]
    if missing:
        raise ValueError(f"missing selector rows: {missing[:3]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open() as handle:
            completed = {
                (row["sample_id"], row["condition"])
                for row in (json.loads(line) for line in handle if line.strip())
            }

    load_start = time.time()
    model = AutoModel.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda:0",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, trust_remote_code=True, use_fast=False
    )
    load_seconds = time.time() - load_start
    transform = build_transform(448)
    generation = {"num_beams": 1, "max_new_tokens": 8, "do_sample": False}
    static_hashes = {
        "manifest_sha256": sha256(args.manifest),
        "selector_sha256": sha256(args.selector),
        "protocol_sha256": sha256(args.protocol),
        "runner_sha256": sha256(__file__),
    }

    total_jobs = len(rows) * len(args.conditions)
    job_number = 0
    written = 0
    with output.open("a") as sink:
        for row in rows:
            selected = selector[row["sample_id"]]
            if selected["video_sha256"] != row["video_sha256"]:
                raise ValueError(f"video hash mismatch: {row['sample_id']}")
            for condition in args.conditions:
                job_number += 1
                if (row["sample_id"], condition) in completed:
                    continue
                indices = [int(value) for value in selected["indices_by_condition"][condition]]
                if (
                    len(indices) != FRAME_BUDGET
                    or len(set(indices)) != FRAME_BUDGET
                    or indices != sorted(indices)
                    or any(index < 0 or index >= selected["source_total_frames"] for index in indices)
                ):
                    raise ValueError(f"bad {condition} indices: {row['sample_id']}")

                started = time.time()
                torch.cuda.reset_peak_memory_stats()
                pixels, decoded_fps, decoded_frames = load_frames(row["video_path"], indices, transform)
                if decoded_frames != selected["source_total_frames"]:
                    raise ValueError(f"decoded frame mismatch: {row['sample_id']}")
                pixels = pixels.to(device="cuda:0", dtype=torch.bfloat16)
                frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(FRAME_BUDGET))
                question = frame_prefix + row["question"] + "\nRespond with only the letter A, B, or C."
                with torch.inference_mode():
                    raw_output = model.chat(
                        tokenizer,
                        pixels,
                        question,
                        generation,
                        num_patches_list=[1] * FRAME_BUDGET,
                        history=None,
                        return_history=False,
                    )
                prediction, parse_status = parse_choice(raw_output)
                record = {
                    "sample_id": row["sample_id"],
                    "official_index": row["official_index"],
                    "video_id": row["video_id"],
                    "condition": condition,
                    "prediction": prediction,
                    "parse_status": parse_status,
                    "raw_output": raw_output,
                    "model_id": "OpenGVLab/InternVL2_5-8B",
                    "model_revision": args.revision,
                    "audio": "off",
                    "frame_budget": FRAME_BUDGET,
                    "selected_source_indices": indices,
                    "source_total_frames": decoded_frames,
                    "source_fps": decoded_fps,
                    "question_sha256": text_sha256(row["question"]),
                    "prompt_sha256": text_sha256(question),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "decoding": generation,
                    "answer_used": False,
                    "latency_sec": time.time() - started,
                    "peak_cuda_mib": torch.cuda.max_memory_allocated() / (1024**2),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                    **static_hashes,
                }
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                written += 1
                print(json.dumps({
                    "progress": f"{job_number}/{total_jobs}",
                    "sample_id": row["sample_id"],
                    "condition": condition,
                    "prediction": prediction,
                    "parse_status": parse_status,
                    "seconds": round(record["latency_sec"], 2),
                    "peak_cuda_mib": round(record["peak_cuda_mib"], 1),
                }), flush=True)
                del pixels

    print(json.dumps({
        "status": "PASS",
        "samples": len(rows),
        "conditions": args.conditions,
        "newly_written": written,
        "model_load_seconds": load_seconds,
        "answer_used": False,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "output": str(output),
        **static_hashes,
    }), flush=True)


if __name__ == "__main__":
    main()
