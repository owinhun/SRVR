#!/usr/bin/env python3
"""Answer-blind same-budget InternVL evaluation for frozen CMTB Dev-48."""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONDITIONS = ("uniform_32", "videoitg_32", "srvr_32")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def safe_manifest_rows(path):
    """Drop evaluation-only answers before returning any runner input."""
    rows = []
    with open(path) as handle:
        for line in handle:
            raw = json.loads(line)
            rows.append(
                {
                    "sample_id": raw["sample_id"],
                    "official_index": raw["official_index"],
                    "video_id": raw["video_id"],
                    "video_path": raw["video_path"],
                    "video_sha256": raw["video_sha256"],
                    "question": raw["selector_prompt"],
                }
            )
    return rows


def load_map(path, key):
    values = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in values:
                raise ValueError(f"duplicate {sample_id} in {path}")
            values[sample_id] = row[key]
    return values


def load_selector_map(path):
    values = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in values:
                raise ValueError(f"duplicate {sample_id} in {path}")
            chronological = [int(value) for value in row["videoitg_source_indices"]]
            ranked = [
                int(value)
                for value in row.get("videoitg_source_indices_by_score", chronological)
            ]
            values[sample_id] = {
                "chronological": chronological,
                "ranked": ranked,
            }
    return values


def get_index(total_frames, fps, num_segments=32, bound=None):
    max_frame = total_frames - 1
    if bound is None:
        start, end = -100000.0, 100000.0
    else:
        start, end = float(bound[0]), float(bound[1])
    start_idx = max(0, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    if end_idx < start_idx:
        raise ValueError(f"invalid clamped frame bound {(start_idx, end_idx)}")
    segment = float(end_idx - start_idx) / num_segments
    indices = [
        int(start_idx + (segment / 2) + np.round(segment * idx))
        for idx in range(num_segments)
    ]
    indices = [min(max_frame, max(0, idx)) for idx in indices]
    if len(set(indices)) == num_segments:
        return indices
    # Deterministic unique fallback for extremely short spans. common_k is
    # capped by the number of available source frames before this call.
    if num_segments == 1:
        return [round((start_idx + end_idx) / 2)]
    unique = [
        round(start_idx + idx * (end_idx - start_idx) / (num_segments - 1))
        for idx in range(num_segments)
    ]
    if len(set(unique)) != num_segments:
        raise ValueError(
            f"cannot draw {num_segments} unique frames from [{start_idx}, {end_idx}]"
        )
    return unique


def build_transform(input_size=448):
    return transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_frames(video_path, indices, transform):
    # One thread is the frozen compatibility path for the CMTB H.264 files.
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    tensors = []
    for index in indices:
        image = Image.fromarray(vr[int(index)].asnumpy()).convert("RGB")
        tensors.append(transform(image))
    return torch.stack(tensors), float(vr.get_avg_fps()), len(vr)


def parse_choice(text):
    stripped = text.strip()
    patterns = [
        r"^\(?\s*([ABC])\s*\)?(?:\b|[.):,-])",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*\(?([ABC])\)?\b",
        r"\(([ABC])\)",
    ]
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, stripped, flags=re.IGNORECASE))
        if found:
            break
    unique = sorted({choice.upper() for choice in found})
    if len(unique) == 1:
        return unique[0], "parsed"
    return None, "unparsed" if not unique else "ambiguous"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--vad", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    rows = safe_manifest_rows(args.manifest)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    selector_indices = load_selector_map(args.selector)
    vad_raw = {}
    with open(args.vad) as handle:
        for line in handle:
            row = json.loads(line)
            vad_raw[row["sample_id"]] = (
                row["automatic_vad_start_sec"],
                row["automatic_vad_end_sec"],
            )
    for row in rows:
        if row["sample_id"] not in selector_indices or row["sample_id"] not in vad_raw:
            raise ValueError(f"missing selector or VAD for {row['sample_id']}")

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

    total_jobs = len(rows) * len(args.conditions)
    written = 0
    job_number = 0
    with output.open("a") as sink:
        for row in rows:
            vr = VideoReader(row["video_path"], ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps = float(vr.get_avg_fps())
            del vr
            selector_row = selector_indices[row["sample_id"]]
            selector_cap = len(selector_row["chronological"])
            vad_start_idx = max(0, round(float(vad_raw[row["sample_id"]][0]) * fps))
            vad_end_idx = min(
                total_frames - 1,
                round(float(vad_raw[row["sample_id"]][1]) * fps),
            )
            vad_unique_cap = max(1, vad_end_idx - vad_start_idx + 1)
            # Official VideoITG can expose fewer than 32 candidates on short
            # clips. Freeze a common per-item K=min(32,candidates); never
            # duplicate source frames merely to reach the nominal cap.
            common_k = min(selector_cap, vad_unique_cap)
            if not 1 <= common_k <= 32:
                raise ValueError(f"invalid common K={common_k} for {row['sample_id']}")
            # New full-split selector artifacts preserve score rank. Legacy
            # Dev artifacts already satisfy K and use their chronological set.
            if selector_row["ranked"] != selector_row["chronological"]:
                videoitg = sorted(selector_row["ranked"][:common_k])
            else:
                videoitg = selector_row["chronological"][:common_k]
            indices_by_condition = {
                "uniform_32": get_index(total_frames, fps, common_k),
                "videoitg_32": videoitg,
                "srvr_32": get_index(
                    total_frames, fps, common_k, vad_raw[row["sample_id"]]
                ),
            }
            for condition in args.conditions:
                job_number += 1
                if (row["sample_id"], condition) in completed:
                    continue
                indices = indices_by_condition[condition]
                if len(indices) != common_k or any(index < 0 or index >= total_frames for index in indices):
                    raise ValueError(f"bad {condition} frame indices for {row['sample_id']}")
                started = time.time()
                torch.cuda.reset_peak_memory_stats()
                pixels, decoded_fps, decoded_frames = load_frames(
                    row["video_path"], indices, transform
                )
                pixels = pixels.to(device="cuda:0", dtype=torch.bfloat16)
                frame_prefix = "".join(
                    f"Frame{i + 1}: <image>\n" for i in range(common_k)
                )
                question = (
                    frame_prefix
                    + row["question"]
                    + "\nRespond with only the letter A, B, or C."
                )
                with torch.inference_mode():
                    raw_output = model.chat(
                        tokenizer,
                        pixels,
                        question,
                        generation,
                        num_patches_list=[1] * common_k,
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
                    "frame_budget_cap": 32,
                    "actual_common_frame_budget": common_k,
                    "selected_source_indices": indices,
                    "source_total_frames": decoded_frames,
                    "source_fps": decoded_fps,
                    "automatic_vad_interval_sec": list(vad_raw[row["sample_id"]])
                    if condition == "srvr_32"
                    else None,
                    "question_sha256": text_sha256(row["question"]),
                    "prompt_sha256": text_sha256(question),
                    "manifest_sha256": sha256(args.manifest),
                    "selector_sha256": sha256(args.selector),
                    "vad_sha256": sha256(args.vad),
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "decoding": generation,
                    "answer_used": False,
                    "latency_sec": time.time() - started,
                    "peak_cuda_mib": torch.cuda.max_memory_allocated() / (1024**2),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "shard_id": args.shard_id,
                    "num_shards": args.num_shards,
                }
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                written += 1
                print(
                    json.dumps(
                        {
                            "progress": f"{job_number}/{total_jobs}",
                            "sample_id": row["sample_id"],
                            "condition": condition,
                            "prediction": prediction,
                            "parse_status": parse_status,
                            "seconds": round(record["latency_sec"], 2),
                            "peak_cuda_mib": round(record["peak_cuda_mib"], 1),
                        }
                    ),
                    flush=True,
                )
                del pixels

    print(
        json.dumps(
            {
                "status": "PASS",
                "samples": len(rows),
                "conditions": args.conditions,
                "newly_written": written,
                "model_load_seconds": load_seconds,
                "answer_used": False,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "output": str(output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
