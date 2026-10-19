#!/usr/bin/env python3
"""Run the frozen official VideoITG selector on an answer-blind CMTB manifest."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from decord import VideoReader, cpu

from eagle.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from eagle.mm_utils import get_model_name_from_path, tokenizer_image_token
from eagle.model.builder import load_pretrained_model


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pad_sequence(tokenizer, input_ids, padding_value):
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(ids, [0]) for ids in input_ids]
    output = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=padding_value
    )
    if tokenizer.padding_side == "left":
        output = torch.flip(output, [1])
    return output


def candidate_indices(total_frames, original_fps, target_fps=2, cap=512):
    stride = max(1, round(original_fps / target_fps))
    indices = list(range(0, total_frames, stride))
    if len(indices) <= cap:
        return indices
    scale = len(indices) / cap
    return [indices[round((i + 1) * scale - 1)] for i in range(cap)]


def uniform_indices(total_frames, k):
    if total_frames <= k:
        return list(range(total_frames))
    scale = total_frames / k
    return [min(total_frames - 1, round((i + 1) * scale - 1)) for i in range(k)]


def safe_manifest_rows(path):
    """Return only selector inputs; answer fields never leave this function."""
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
                    "selector_prompt": raw["selector_prompt"],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    rows = safe_manifest_rows(args.manifest)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_id]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("empty selector population")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open() as handle:
            completed = {json.loads(line)["sample_id"] for line in handle if line.strip()}

    device = torch.device("cuda:0")
    load_start = time.time()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.checkpoint,
        None,
        get_model_name_from_path(args.checkpoint),
        device_map="cuda:0",
    )
    model.half().eval().to(device)
    load_seconds = time.time() - load_start

    written = 0
    with output.open("a") as sink:
        for ordinal, row in enumerate(rows, start=1):
            if row["sample_id"] in completed:
                continue
            item_start = time.time()
            # A single decoder thread is the frozen compatibility path. The
            # original four-thread preflight failed on an H.264 packet before
            # any downstream prediction; v2 restarts the whole population.
            vr = VideoReader(row["video_path"], ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            fps = float(vr.get_avg_fps())
            candidates = candidate_indices(total_frames, fps)
            frames = vr.get_batch(candidates).asnumpy()
            pixels = image_processor.preprocess(frames, return_tensors="pt")[
                "pixel_values"
            ].half().to(device)

            prompt = DEFAULT_IMAGE_TOKEN + row["selector_prompt"] + "\n"
            ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            input_ids = pad_sequence(tokenizer, [ids], pad_id).to(device)
            attention_mask = input_ids.ne(pad_id).to(device)
            with torch.inference_mode():
                response = model(
                    input_ids,
                    attention_mask=attention_mask,
                    images=[pixels],
                )
            logits = response.logits[0].sigmoid().view(-1)
            if len(logits) != len(candidates):
                raise RuntimeError(
                    f"{row['sample_id']}: {len(logits)} logits for {len(candidates)} frames"
                )
            values, local = torch.sort(logits, descending=True)
            k = min(args.top_k, len(local))
            videoitg_ranked = [candidates[int(i)] for i in local[:k].tolist()]
            videoitg_indices = sorted(videoitg_ranked)
            record = {
                **{key: row[key] for key in ("sample_id", "official_index", "video_id")},
                "status": "PASS",
                "answer_used": False,
                "manifest_sha256": sha256(args.manifest),
                "video_sha256": row["video_sha256"],
                "source_frames": total_frames,
                "source_fps": fps,
                "candidate_frames": len(candidates),
                "selected_k": k,
                "uniform_source_indices": uniform_indices(total_frames, k),
                "videoitg_source_indices": videoitg_indices,
                "videoitg_source_indices_by_score": videoitg_ranked,
                "videoitg_scores_descending": [float(x) for x in values[:k].cpu()],
                "item_seconds": time.time() - item_start,
                "model_load_seconds": load_seconds,
                "peak_cuda_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            written += 1
            print(
                json.dumps(
                    {
                        "progress": f"{ordinal}/{len(rows)}",
                        "sample_id": row["sample_id"],
                        "candidate_frames": len(candidates),
                        "seconds": round(record["item_seconds"], 2),
                        "peak_cuda_mib": round(record["peak_cuda_mib"], 1),
                    }
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "status": "PASS",
                "population": len(rows),
                "newly_written": written,
                "output": str(output),
                "answer_used": False,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
