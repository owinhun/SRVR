#!/usr/bin/env python3
"""Build and freeze the video-disjoint TER large-evaluation cue index."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


VERSION = "ter_large_eval_v1"
TARGET_TASKS = ("TA", "TF", "CMTB", "CSMB")
VAD_CONFIG = {
    "package": "silero-vad",
    "package_version": "6.2.1",
    "threshold": 0.5,
    "sampling_rate": 16000,
    "minimum_speech_duration_ms": 250,
    "minimum_silence_duration_ms": 100,
    "speech_pad_ms": 30,
    "segment_selection_rule": "first detected speech start through last detected speech end",
    "timestamp_rounding_rule": "store 3 decimals; prompt renders 2 decimals",
    "timestamp_padding_sec": 0.0,
    "vad_clip_padding_sec": 1.0,
}
ASR_CONFIG = {
    "model_id": "Systran/faster-whisper-small.en",
    "model_alias": "small.en",
    "revision": "local CTranslate2 model snapshot; content hash recorded at runtime",
    "beam_size": 5,
    "vad_filter": False,
    "language": "en",
    "condition_on_previous_text": True,
    "temperature": 0.0,
}
RANDOM_CONFIG = {
    "seed_namespace": "svhalluc-ter-large-v1",
    "same_length_as_auto_vad": True,
    "exclusion_margin_sec": 1.0,
    "fallback": "farthest feasible same-length interval",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def atomic_text(path: Path, text: str, force: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def media_duration(path: Path) -> float:
    payload = json.loads(run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]))
    return float(payload["format"]["duration"])


def sample_id(task: str, official_index: int) -> str:
    return f"{task.lower()}_{official_index:06d}"


def build_manifest(samples_path: Path, dev_path: Path, output: Path) -> dict[str, Any]:
    official = json.loads(samples_path.read_text())
    dev = read_jsonl(dev_path)
    dev_videos = {row["video_id"] for row in dev}
    root = samples_path.parent
    eligible, excluded = [], []
    for index, row in enumerate(official):
        task = row["task_type_abbr"]
        if task not in TARGET_TASKS:
            continue
        target = excluded if row["video_path"] in dev_videos else eligible
        target.append((index, row))
    if len(eligible) < 1000:
        raise RuntimeError(f"only {len(eligible)} eligible rows remain")
    if {row["video_path"] for _, row in eligible} & dev_videos:
        raise RuntimeError("Dev video overlap is nonzero")
    rows = [{
        "sample_id": sample_id(row["task_type_abbr"], index),
        "official_index": index,
        "official_metadata_path": str(samples_path.resolve()),
        "official_metadata_sha256": sha256(samples_path),
        "video_id": row["video_path"],
        "video_path": str((root / row["video_path"]).resolve()),
        "data_type": row["data_type"],
        "task": row["task_type_abbr"],
        "task_type": row["task_type"],
        "question": row["question"],
        "answer": row["answer"],
    } for index, row in eligible]
    if output.exists():
        prior = read_jsonl(output)
        if prior != rows:
            raise RuntimeError(f"locked manifest exists with different content: {output}")
    else:
        atomic_jsonl(output, rows)
    per_task = {}
    for task in TARGET_TASKS:
        subset = [row for row in rows if row["task"] == task]
        per_task[task] = {
            "qa_count": len(subset),
            "unique_video_count": len({row["video_id"] for row in subset}),
            "answer_distribution": dict(sorted(collections.Counter(
                row["answer"] for row in subset
            ).items())),
        }
    return {
        "manifest_path": str(output.resolve()),
        "manifest_sha256": sha256(output),
        "total_qa_count": len(rows),
        "unique_video_count": len({row["video_id"] for row in rows}),
        "per_task": per_task,
        "excluded_row_count": len(excluded),
        "excluded_video_count": len({row["video_path"] for _, row in excluded}),
        "dev_video_overlap_count": 0,
    }


def deterministic_random_interval(
    video_id: str, start: float, end: float, duration: float
) -> tuple[float, float, bool]:
    length = min(max(0.0, end - start), duration)
    max_start = max(0.0, duration - length)
    margin = RANDOM_CONFIG["exclusion_margin_sec"]
    ranges = []
    left_end = min(max_start, start - margin - length)
    if left_end >= 0:
        ranges.append((0.0, left_end))
    right_start = max(0.0, end + margin)
    if right_start <= max_start:
        ranges.append((right_start, max_start))
    seed = int.from_bytes(hashlib.sha256(
        f"{RANDOM_CONFIG['seed_namespace']}:{video_id}".encode()
    ).digest()[:8], "big")
    rng = random.Random(seed)
    if ranges:
        weights = [max(0.0, hi - lo) for lo, hi in ranges]
        if sum(weights) == 0:
            chosen = ranges[seed % len(ranges)]
            random_start = chosen[0]
        else:
            pick = rng.random() * sum(weights)
            chosen = ranges[-1]
            for candidate, weight in zip(ranges, weights):
                if pick <= weight:
                    chosen = candidate
                    break
                pick -= weight
            random_start = rng.uniform(*chosen)
        return random_start, random_start + length, False
    candidates = [0.0, max_start]
    random_start = max(
        candidates,
        key=lambda value: min(abs(value + length - start), abs(value - end)),
    )
    return random_start, random_start + length, True


class CueModels:
    def __init__(self, device: str, asr_model: str) -> None:
        import torch
        import silero_vad
        from faster_whisper import WhisperModel

        self.torch = torch
        self.silero_vad = silero_vad
        self.vad = silero_vad.load_silero_vad()
        self.asr = WhisperModel(
            asr_model,
            device="cuda" if device == "cuda" else "cpu",
            compute_type="float16" if device == "cuda" else "int8",
        )
        self.asr_model = asr_model

    def vad(self, wav: np.ndarray) -> tuple[list[dict[str, int]], float | None]:
        raise AssertionError("shadowed by model")

    def detect(self, wav: np.ndarray) -> tuple[list[dict[str, int]], float | None]:
        timestamps = self.silero_vad.get_speech_timestamps(
            self.torch.from_numpy(wav.astype(np.float32)),
            self.vad,
            sampling_rate=VAD_CONFIG["sampling_rate"],
            threshold=VAD_CONFIG["threshold"],
            min_speech_duration_ms=VAD_CONFIG["minimum_speech_duration_ms"],
            min_silence_duration_ms=VAD_CONFIG["minimum_silence_duration_ms"],
            speech_pad_ms=VAD_CONFIG["speech_pad_ms"],
            return_seconds=False,
        )
        if not timestamps:
            return [], None
        # Silero does not expose calibrated segment confidence in this API.
        return timestamps, None

    def transcribe(self, path: Path) -> tuple[str, float | None]:
        segments, _ = self.asr.transcribe(
            str(path),
            beam_size=ASR_CONFIG["beam_size"],
            vad_filter=ASR_CONFIG["vad_filter"],
            language=ASR_CONFIG["language"],
            condition_on_previous_text=ASR_CONFIG["condition_on_previous_text"],
            temperature=ASR_CONFIG["temperature"],
        )
        materialized = list(segments)
        text = " ".join(segment.text.strip() for segment in materialized).strip()
        probabilities = [
            math.exp(min(0.0, float(segment.avg_logprob)))
            for segment in materialized if segment.text.strip()
        ]
        return text, (sum(probabilities) / len(probabilities) if probabilities else None)


def cue_one(
    video_id: str, video_path: Path, models: CueModels, code_hash: str,
    vad_model_hash: str | None, asr_model_hash: str | None,
) -> dict[str, Any]:
    try:
        duration = media_duration(video_path)
        source_hash = sha256(video_path)
    except Exception as exc:
        return {
            "video_id": video_id, "video_path": str(video_path),
            "vad_status": f"media_failure:{type(exc).__name__}:{exc}",
            "asr_status": "skipped_media_failure",
            "cue_generation_code_sha256": code_hash,
        }
    with tempfile.TemporaryDirectory(prefix="tercue_", dir="/tmp") as name:
        wav_path = Path(name) / "full.wav"
        try:
            decode_command = [
                "ffmpeg", "-v", "error", "-y", "-i", str(video_path), "-vn",
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
            ]
            try:
                run(decode_command)
                audio_decode_method = "ffmpeg_auto"
            except subprocess.CalledProcessError:
                # Official CSMB ipcm is signed 16-bit little-endian PCM.
                run([
                    "ffmpeg", "-v", "error", "-y", "-c:a", "pcm_s16le",
                    "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(wav_path),
                ])
                audio_decode_method = "ffmpeg_forced_pcm_s16le_ipcm"
            wav, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000:
                raise RuntimeError(f"unexpected sample rate {sr}")
        except Exception as exc:
            return {
                "video_id": video_id, "video_path": str(video_path),
                "media_duration_sec": duration, "source_file_sha256": source_hash,
                "vad_status": f"audio_decode_failure:{type(exc).__name__}:{exc}",
                "asr_status": "skipped_audio_decode_failure",
                "cue_generation_code_sha256": code_hash,
            }
        try:
            segments, confidence = models.detect(wav)
        except Exception as exc:
            segments, confidence = [], None
            vad_status = f"vad_failure:{type(exc).__name__}:{exc}"
        else:
            vad_status = "ok" if segments else "no_speech"
        start = end = speech_duration = None
        transcript = ""
        asr_confidence = None
        if segments:
            start = max(0.0, segments[0]["start"] / 16000)
            end = min(duration, segments[-1]["end"] / 16000)
            speech_duration = max(0.0, end - start)
            segment_path = Path(name) / "speech.wav"
            sf.write(
                segment_path,
                wav[segments[0]["start"]:segments[-1]["end"]],
                16000, subtype="PCM_16",
            )
            try:
                transcript, asr_confidence = models.transcribe(segment_path)
                asr_status = "ok" if transcript else "empty_transcript"
            except Exception as exc:
                asr_status = f"asr_failure:{type(exc).__name__}:{exc}"
        else:
            asr_status = "skipped_no_vad"
        random_start = random_end = None
        fallback = False
        if start is not None and end is not None:
            random_start, random_end, fallback = deterministic_random_interval(
                video_id, start, end, duration
            )
        return {
            "video_id": video_id,
            "video_path": str(video_path.resolve()),
            "speech_start_sec": round(start, 3) if start is not None else None,
            "speech_end_sec": round(end, 3) if end is not None else None,
            "speech_duration_sec": round(speech_duration, 3) if speech_duration is not None else None,
            "vad_status": vad_status,
            "vad_confidence": confidence,
            "vad_confidence_status": "not_exposed_by_frozen_silero_api",
            "vad_segment_count": len(segments),
            "asr_transcript": transcript,
            "asr_confidence": asr_confidence,
            "asr_status": asr_status,
            "media_duration_sec": duration,
            "source_file_sha256": source_hash,
            "audio_decode_method": audio_decode_method,
            "random_start_sec": round(random_start, 3) if random_start is not None else None,
            "random_end_sec": round(random_end, 3) if random_end is not None else None,
            "random_timestamp_fallback": fallback,
            "vad_model_revision": vad_model_hash,
            "asr_model_revision": asr_model_hash,
            "cue_generation_code_sha256": code_hash,
            "vad_config": VAD_CONFIG,
            "asr_config": ASR_CONFIG,
            "random_timestamp_config": RANDOM_CONFIG,
            "version": VERSION,
        }


def write_reports(
    manifest_info: dict[str, Any], cues: list[dict[str, Any]],
    cue_path: Path, summary_path: Path, handoff_path: Path,
    code_path: Path, vad_hash: str | None, asr_hash: str | None,
) -> None:
    vad_status = collections.Counter(row["vad_status"].split(":")[0] for row in cues)
    asr_status = collections.Counter(row["asr_status"].split(":")[0] for row in cues)
    summary = {
        "version": VERSION,
        **manifest_info,
        "cue_index_path": str(cue_path.resolve()),
        "cue_index_sha256": sha256(cue_path),
        "cue_row_count": len(cues),
        "vad_status_counts": dict(sorted(vad_status.items())),
        "asr_status_counts": dict(sorted(asr_status.items())),
        "random_timestamp_fallback_count": sum(
            bool(row.get("random_timestamp_fallback")) for row in cues
        ),
        "vad_config": VAD_CONFIG,
        "asr_config": ASR_CONFIG,
        "random_timestamp_config": RANDOM_CONFIG,
        "vad_model_tree_sha256": vad_hash,
        "asr_model_tree_sha256": asr_hash,
        "cue_generation_code_path": str(code_path.resolve()),
        "cue_generation_code_sha256": sha256(code_path),
    }
    atomic_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    handoff = f"""# TER large evaluation v1 — audio cue handoff

This is a locked, video-disjoint evaluation input. Dev-48 is excluded by
`video_path` across every task. Do not tune VAD, ASR, timestamp formatting,
padding, prompts, or parsing after inspecting model results.

- Manifest: `{manifest_info['manifest_path']}`
- Manifest SHA256: `{manifest_info['manifest_sha256']}`
- QA rows: {manifest_info['total_qa_count']}
- Unique videos / cue rows: {len(cues)}
- Dev overlap: {manifest_info['dev_video_overlap_count']}
- Cue index: `{cue_path.resolve()}`
- Cue index SHA256: `{sha256(cue_path)}`
- VAD statuses: `{dict(vad_status)}`
- ASR statuses: `{dict(asr_status)}`
- Random timestamp fallbacks: {summary['random_timestamp_fallback_count']}

The official labels, questions, and option order are unchanged. No artificial
audio was generated. Rows with VAD/ASR failures remain in the locked files.
"""
    atomic_text(handoff_path, handoff)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path(
        "/workspace/svhalluc/data/SVHalluc/samples.json"))
    parser.add_argument("--dev-manifest", type=Path, default=Path(
        "/workspace/svhalluc/data/phase0_cmtb.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path(
        "/workspace/svhalluc/data/ter_large_eval_v1.jsonl"))
    parser.add_argument("--cues", type=Path, default=Path(
        "/workspace/svhalluc/outputs/ter_cues_large_v1.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path(
        "/workspace/svhalluc/outputs/ter_cues_large_v1_summary.json"))
    parser.add_argument("--handoff", type=Path, default=Path(
        "/workspace/svhalluc/outputs/ter_large_eval_v1_handoff.md"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--limit-videos", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    manifest_info = build_manifest(args.samples, args.dev_manifest, args.manifest)
    manifest = read_jsonl(args.manifest)
    videos = {}
    for row in manifest:
        videos.setdefault(row["video_id"], Path(row["video_path"]))
    ordered = sorted(videos.items())
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    ordered = ordered[args.shard_index::args.shard_count]
    if args.limit_videos is not None:
        ordered = ordered[:args.limit_videos]

    code_path = Path(__file__)
    code_hash = sha256(code_path)
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", "/workspace/.cache"))
    import silero_vad
    vad_hash = tree_hash(Path(silero_vad.__file__).parent / "data")
    asr_hash = tree_hash(
        cache_root / "huggingface" / "hub"
        / "models--Systran--faster-whisper-small.en"
    )
    existing = {row["video_id"]: row for row in read_jsonl(args.cues)}
    models = CueModels(args.device, args.asr_model)
    for index, (video_id, path) in enumerate(ordered, 1):
        if (
            video_id in existing
            and existing[video_id].get("cue_generation_code_sha256") == code_hash
        ):
            print(f"[{index}/{len(ordered)}] reuse {video_id}", flush=True)
            continue
        print(f"[{index}/{len(ordered)}] process {video_id}", flush=True)
        existing[video_id] = cue_one(
            video_id, path, models, code_hash, vad_hash, asr_hash
        )
        atomic_jsonl(args.cues, [existing[key] for key in sorted(existing)])

    if args.limit_videos is None and args.shard_count == 1:
        cues = read_jsonl(args.cues)
        if {row["video_id"] for row in cues} != set(videos):
            raise RuntimeError("cue index does not exactly match manifest videos")
        write_reports(
            manifest_info, cues, args.cues, args.summary, args.handoff,
            code_path, vad_hash, asr_hash,
        )
        print(f"Manifest: {args.manifest}")
        print(f"Cues: {args.cues}")
        print(f"Summary: {args.summary}")
        print(f"Handoff: {args.handoff}")


if __name__ == "__main__":
    main()
