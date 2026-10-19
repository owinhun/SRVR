#!/usr/bin/env python3
"""Build the fixed CMTB Phase-0 manifest and O_proc processing controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SEED = 20260728
COUNT = 48
SAMPLE_RATE = 44100
DEMUCS_MODEL = "htdemucs"
CHOICE_RE = re.compile(r"\(([ABC])\)\s*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Frozen manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def parse_choices(question: str) -> dict[str, str]:
    matches = list(CHOICE_RE.finditer(question))
    choices = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        choices[match.group(1)] = question[match.end():end].strip()
    if set(choices) != {"A", "B", "C"}:
        raise ValueError(f"Could not parse A/B/C choices: {question}")
    return choices


def build_manifest(dataset_root: Path, output: Path) -> None:
    if output.exists():
        print(f"Frozen manifest exists; unchanged: {output}")
        validate_manifest(output)
        return
    metadata = dataset_root / "samples.json"
    official = json.loads(metadata.read_text(encoding="utf-8"))
    candidates = [
        (index, row) for index, row in enumerate(official)
        if row.get("task_type_abbr") == "CMTB"
        and (dataset_root / row["video_path"]).is_file()
    ]
    candidates.sort(key=lambda item: item[1]["video_path"])
    random.Random(SEED).shuffle(candidates)
    selected = candidates[:COUNT]
    if len(selected) != COUNT:
        raise RuntimeError(f"Need {COUNT} accessible CMTB rows, found {len(candidates)}")
    metadata_hash = sha256(metadata)
    rows = []
    for rank, (official_index, row) in enumerate(selected):
        video = (dataset_root / row["video_path"]).resolve()
        rows.append({
            "sample_id": f"cmtb_{official_index:06d}",
            "video_id": row["video_path"],
            "task": "CMTB",
            "video_path": str(video),
            "question": row["question"],
            "choices": parse_choices(row["question"]),
            "answer": row["answer"],
            "official_index": official_index,
            "official_metadata_path": str(metadata.resolve()),
            "official_metadata_sha256": metadata_hash,
            "selection_seed": SEED,
            "selection_rank": rank,
        })
    atomic_jsonl(output, rows)
    validate_manifest(output)
    print(f"Created frozen manifest: {output}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_manifest(path: Path) -> None:
    rows = read_jsonl(path)
    if len(rows) != COUNT:
        raise ValueError(f"Expected {COUNT} rows, found {len(rows)}")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != COUNT:
        raise ValueError("sample_id values are not unique")
    for row in rows:
        if row["task"] != "CMTB":
            raise ValueError(f"{row['sample_id']}: task is not CMTB")
        if row["answer"] not in row["choices"]:
            raise ValueError(f"{row['sample_id']}: answer not in choices")
        video = Path(row["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
        run(["ffprobe", "-v", "error", str(video)], capture=True)


def probe(path: Path) -> dict[str, Any]:
    result = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,sample_rate,channels,duration",
        "-of", "json", str(path),
    ], capture=True)
    payload = json.loads(result.stdout)
    audio = next(s for s in payload["streams"] if s["codec_type"] == "audio")
    video = next(s for s in payload["streams"] if s["codec_type"] == "video")
    return {
        "duration_sec": float(payload["format"]["duration"]),
        "audio_duration_sec": float(audio.get("duration", payload["format"]["duration"])),
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
        "audio_codec": audio["codec_name"],
        "video_codec": video["codec_name"],
    }


def audio_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64))) if signal.size else 0.0


def match_shape(signal: np.ndarray, frames: int, channels: int) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.shape[1] > channels:
        signal = signal[:, :channels]
    elif signal.shape[1] < channels:
        signal = np.repeat(signal, channels, axis=1)
    signal = signal[:frames]
    if len(signal) < frames:
        signal = np.pad(signal, ((0, frames - len(signal)), (0, 0)))
    return signal.astype(np.float32, copy=False)


def vad_bounds(path: Path) -> tuple[float | None, float | None]:
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

        wav = read_audio(str(path), sampling_rate=16000)
        segments = get_speech_timestamps(
            wav, load_silero_vad(), sampling_rate=16000,
            threshold=0.5, return_seconds=True,
        )
        if not segments:
            return None, None
        return float(segments[0]["start"]), float(segments[-1]["end"])
    except Exception:
        return None, None


def process_one(row: dict[str, Any], output_dir: Path, work_root: Path,
                device: str, force: bool) -> dict[str, Any]:
    sample_id = row["sample_id"]
    source = Path(row["video_path"])
    destination = output_dir / f"{sample_id}_o_proc.mp4"
    if destination.exists() and not force:
        try:
            output_probe = probe(destination)
            if destination.stat().st_size > 0:
                return qc_result(row, source, destination, output_probe, reused=True)
        except Exception as exc:
            return failure_row(row, destination, f"invalid_existing_output:{exc}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        original_probe = probe(source)
        with tempfile.TemporaryDirectory(prefix=f"{sample_id}_", dir=work_root) as tmp_name:
            tmp = Path(tmp_name)
            original_wav = tmp / "original.wav"
            run([
                "ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn",
                "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le",
                str(original_wav),
            ])
            original, sr = sf.read(original_wav, dtype="float32", always_2d=True)
            run([
                "python", "-m", "demucs", "--two-stems", "vocals",
                "-n", DEMUCS_MODEL, "-d", device, "-o", str(tmp / "demucs"),
                str(original_wav),
            ])
            stem_root = tmp / "demucs" / DEMUCS_MODEL / original_wav.stem
            vocals, vocals_sr = sf.read(stem_root / "vocals.wav", dtype="float32", always_2d=True)
            residual, residual_sr = sf.read(
                stem_root / "no_vocals.wav", dtype="float32", always_2d=True
            )
            if sr != SAMPLE_RATE or vocals_sr != SAMPLE_RATE or residual_sr != SAMPLE_RATE:
                raise RuntimeError(
                    f"unexpected sample rates original={sr}, vocals={vocals_sr}, residual={residual_sr}"
                )
            vocals = match_shape(vocals, len(original), original.shape[1])
            residual = match_shape(residual, len(original), original.shape[1])
            recombined = vocals + residual
            source_rms = audio_rms(original)
            recombined_rms = audio_rms(recombined)
            if recombined_rms > 1e-12:
                recombined *= source_rms / recombined_rms
            peak = float(np.max(np.abs(recombined)))
            if peak > 0.999:
                recombined *= 0.999 / peak
            recombined_wav = tmp / "recombined.wav"
            sf.write(recombined_wav, recombined, SAMPLE_RATE, subtype="PCM_24")
            temporary_output = destination.with_name(f".{destination.stem}.tmp.mp4")
            run([
                "ffmpeg", "-v", "error", "-y",
                "-i", str(source), "-i", str(recombined_wav),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map_metadata", "0", "-movflags", "+faststart",
                "-shortest", str(temporary_output),
            ])
            probe(temporary_output)
            os.replace(temporary_output, destination)
        return qc_result(row, source, destination, probe(destination), reused=False)
    except Exception as exc:
        return failure_row(row, destination, f"{type(exc).__name__}:{exc}")


def decode_metrics(path: Path, tmp: Path) -> tuple[float, float, bool]:
    wav = tmp / (path.stem + ".wav")
    run([
        "ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn",
        "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(wav),
    ])
    signal, _ = sf.read(wav, dtype="float32", always_2d=True)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    return audio_rms(signal), peak, peak >= 0.9999


def qc_result(row: dict[str, Any], source: Path, destination: Path,
              output_probe: dict[str, Any], reused: bool) -> dict[str, Any]:
    original_probe = probe(source)
    with tempfile.TemporaryDirectory(prefix="phase0_qc_") as tmp_name:
        tmp = Path(tmp_name)
        original_rms, original_peak, original_clipping = decode_metrics(source, tmp)
        output_rms, output_peak, clipping = decode_metrics(destination, tmp)
        original_wav = tmp / (source.stem + ".wav")
        output_wav = tmp / (destination.stem + ".wav")
        original_start, original_end = vad_bounds(original_wav)
        output_start, output_end = vad_bounds(output_wav)
    duration_diff = output_probe["duration_sec"] - original_probe["duration_sec"]
    rms_ratio = output_rms / max(original_rms, 1e-12)
    vad_shift = None
    speech_preserved = output_start is not None and output_end is not None
    if None not in (original_start, original_end, output_start, output_end):
        vad_shift = max(abs(output_start - original_start), abs(output_end - original_end))
        speech_preserved = vad_shift <= 0.25
    reasons = []
    if abs(duration_diff) > 0.05:
        reasons.append("duration_diff_gt_0.05s")
    if clipping:
        reasons.append("clipping")
    if not (10 ** (-0.5 / 20) <= rms_ratio <= 10 ** (0.5 / 20)):
        reasons.append("rms_diff_gt_0.5dB")
    if not speech_preserved:
        reasons.append("speech_not_preserved")
    if output_probe["sample_rate"] != SAMPLE_RATE:
        reasons.append("unexpected_sample_rate")
    return {
        "sample_id": row["sample_id"],
        "o_proc_path": str(destination.resolve()),
        "o_proc_sha256": sha256(destination),
        "qc_pass": not reasons,
        "failure_reason": ";".join(reasons),
        "original_duration_sec": original_probe["duration_sec"],
        "o_proc_duration_sec": output_probe["duration_sec"],
        "duration_diff_sec": duration_diff,
        "sample_rate": output_probe["sample_rate"],
        "channels": output_probe["channels"],
        "clipping_detected": clipping,
        "original_peak": original_peak,
        "o_proc_peak": output_peak,
        "rms_ratio": rms_ratio,
        "speech_preserved": speech_preserved,
        "original_speech_start_sec": original_start,
        "original_speech_end_sec": original_end,
        "o_proc_speech_start_sec": output_start,
        "o_proc_speech_end_sec": output_end,
        "vad_boundary_max_shift_sec": vad_shift,
        "video_codec": output_probe["video_codec"],
        "audio_codec": output_probe["audio_codec"],
        "demucs_model": DEMUCS_MODEL,
        "reused_existing_output": reused,
    }


def failure_row(row: dict[str, Any], destination: Path, reason: str) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"], "o_proc_path": str(destination.resolve()),
        "qc_pass": False, "failure_reason": reason,
    }


def process_manifest(manifest: Path, output_dir: Path, result_path: Path,
                     device: str, limit: int | None, force: bool) -> None:
    validate_manifest(manifest)
    rows = read_jsonl(manifest)
    if limit is not None:
        rows = rows[:limit]
    existing = {}
    if result_path.exists():
        existing = {row["sample_id"]: row for row in read_jsonl(result_path)}
    results = []
    work_root = Path("/tmp")
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        if sample_id in existing and not force:
            result = existing[sample_id]
            path = Path(result["o_proc_path"])
            if path.is_file():
                print(f"[{index}/{len(rows)}] reuse {sample_id}")
                results.append(result)
                continue
        print(f"[{index}/{len(rows)}] process {sample_id}", flush=True)
        results.append(process_one(row, output_dir, work_root, device, force))
        # Atomic checkpoint after every sample, without duplicate IDs.
        merged = {**existing, **{item["sample_id"]: item for item in results}}
        payload = "".join(
            json.dumps(merged[key], ensure_ascii=False, sort_keys=True) + "\n"
            for key in sorted(merged)
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{result_path.name}.", dir=result_path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, result_path)
    print(f"Results: {result_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--dataset-root", type=Path, default=Path("/workspace/svhalluc/data/SVHalluc"))
    manifest.add_argument("--output", type=Path, default=Path("/workspace/svhalluc/data/phase0_cmtb.jsonl"))
    process = subparsers.add_parser("process")
    process.add_argument("--manifest", type=Path, default=Path("/workspace/svhalluc/data/phase0_cmtb.jsonl"))
    process.add_argument("--output-dir", type=Path, default=Path("/workspace/svhalluc/outputs/o_proc"))
    process.add_argument("--result", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase0.jsonl"))
    process.add_argument("--device", default="cuda")
    process.add_argument("--limit", type=int)
    process.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "manifest":
        build_manifest(args.dataset_root, args.output)
    else:
        process_manifest(args.manifest, args.output_dir, args.result, args.device, args.limit, args.force)


if __name__ == "__main__":
    main()
