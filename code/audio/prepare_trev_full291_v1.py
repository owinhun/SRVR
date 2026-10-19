#!/usr/bin/env python3
"""Prepare the frozen full-CMTB time-reversed speech-like control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import phase1_pilot as v1
import phase1_v2 as v2
from phase0 import DEMUCS_MODEL, SAMPLE_RATE, audio_rms, probe, read_jsonl, run, sha256


ROOT = Path("/workspace/svhalluc")
MANIFEST_SHA256 = "9221a8af1f67e9da22bedbb2bca9c68333ec6a38c5f614219c05376b2b451f9e"
CUES_SHA256 = "90b5712968d1e17f5526efdb532e5c1e75cb78ec71320c4db6bbb496d46412bf"
PROTOCOL = ROOT / "paper/v46_qwen25_trev_pointer_full291_protocol_v1.md"
PROTOCOL_SHA256 = "0f48b99e266ae32e0fb0aa60e5725ea2242d04acf7b6b42b156b5652edb9b3ae"
VERSION = "qwen25_trev_pointer_full291_v1_contract_amended"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/ter_large_eval_v1.jsonl")
    parser.add_argument("--cues", type=Path, default=ROOT / "outputs/ter_cues_large_v1.jsonl")
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "outputs/v46_trev_media_contract",
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_shard(sample_id: str, count: int) -> int:
    return int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % count


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(rows, key=lambda item: item["sample_id"])
    )
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def frozen_vad_segments(models: v2.Models, signal: np.ndarray, sr: int) -> list[dict[str, int]]:
    from silero_vad import get_speech_timestamps

    mono = signal.mean(axis=1) if signal.ndim == 2 else signal
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // divisor, sr // divisor)
    raw = get_speech_timestamps(
        models.torch.from_numpy(mono.astype(np.float32)), models.vad,
        sampling_rate=16000, threshold=0.5, min_speech_duration_ms=250,
        min_silence_duration_ms=100, speech_pad_ms=30, return_seconds=False,
    )
    scale = sr / 16000
    return [
        {"start": round(item["start"] * scale), "end": round(item["end"] * scale)}
        for item in raw
    ]


def encoded_video_hash(path: Path) -> str:
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-c", "copy", "-f", "hash", "-hash", "sha256", "-",
        ],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    value = completed.stdout.strip()
    if not value.startswith("SHA256="):
        raise RuntimeError(f"unexpected video hash output: {value}")
    return value.split("=", 1)[1].lower()


def db_ratio(numerator: float, denominator: float) -> float:
    return 20.0 * math.log10(max(numerator, 1e-12) / max(denominator, 1e-12))


def mux_source_audio_contract(
    source: Path, wav: Path, destination: Path, source_info: dict[str, Any]
) -> None:
    """Copy video and match the source audio stream's rate and duration.

    The initial processor preflight showed that a fixed 44.1-kHz output could
    shift Qwen's audio placeholder count by one token for 48-kHz sources.  This
    deterministic mux correction was frozen before benchmark inference.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.contract.tmp.mp4")
    duration = float(source_info["audio_duration_sec"])
    sample_rate = int(source_info["sample_rate"])
    try:
        run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source), "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-af", f"apad,atrim=end={duration:.9f}",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(sample_rate),
            "-ac", "2", "-map_metadata", "0", "-movflags", "+faststart",
            str(temporary),
        ])
        v1.full_media_check(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def process_one(
    row: dict[str, Any], cue: dict[str, Any], output_root: Path,
    models: v2.Models, device: str, force: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    sample_id = row["sample_id"]
    source = Path(row["video_path"])
    destination = output_root / sample_id / "T_rev.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if source.resolve() != Path(cue["video_path"]).resolve():
            raise RuntimeError("manifest_cue_path_mismatch")
        if sha256(source) != cue["source_file_sha256"]:
            raise RuntimeError("source_sha256_mismatch")
        with tempfile.TemporaryDirectory(prefix=f"v46_trev_{sample_id}_", dir="/tmp") as name:
            tmp = Path(name)
            original_wav = tmp / "original.wav"
            run([
                "ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn",
                "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le",
                str(original_wav),
            ])
            original, sr = sf.read(original_wav, dtype="float32", always_2d=True)
            segments = frozen_vad_segments(models, original, sr)
            if not segments:
                raise RuntimeError("no_original_speech_detected")
            mask = v1.interval_mask(segments, len(original))
            recomputed_start = segments[0]["start"] / sr
            recomputed_end = segments[-1]["end"] / sr
            envelope_max_abs_diff = max(
                abs(recomputed_start - float(cue["speech_start_sec"])),
                abs(recomputed_end - float(cue["speech_end_sec"])),
            )

            run([
                "python", "-m", "demucs", "--two-stems", "vocals",
                "-n", DEMUCS_MODEL, "-d", device, "-o", str(tmp / "demucs"),
                str(original_wav),
            ])
            stem_root = tmp / "demucs" / DEMUCS_MODEL / original_wav.stem
            vocals, vocals_sr = sf.read(
                stem_root / "vocals.wav", dtype="float32", always_2d=True
            )
            residual, residual_sr = sf.read(
                stem_root / "no_vocals.wav", dtype="float32", always_2d=True
            )
            if (sr, vocals_sr, residual_sr) != (SAMPLE_RATE, SAMPLE_RATE, SAMPLE_RATE):
                raise RuntimeError("unexpected_stem_sample_rate")
            vocals = v1.exact_audio(vocals, len(original), 2)
            residual = v1.exact_audio(residual, len(original), 2)
            gated = vocals * mask[:, None]
            target = v2.active_rms(gated, mask)
            reversed_raw = v1.reverse_within_segments(gated, segments)
            reversed_raw = v1.match_active_rms(reversed_raw, mask, target)
            carrier, generation = v2.carrier_peak_control(
                reversed_raw, residual, mask, target
            )
            mixed = residual + carrier
            generated_support_iou = v1.mask_iou(
                mask, np.any(np.abs(carrier) > 1e-8, axis=1)
            )

            source_info = probe(source)
            if destination.exists() and not force:
                v1.full_media_check(destination)
            else:
                wav = tmp / "T_rev.wav"
                sf.write(wav, mixed, SAMPLE_RATE, subtype="PCM_24")
                mux_source_audio_contract(source, wav, destination, source_info)

            detail, transformed = v2.media_qc(
                "T_rev", destination, source_info["duration_sec"], models, tmp
            )
            output_info = probe(destination)
            transformed = v1.exact_audio(transformed, len(original), 2)
            original_total_rms = audio_rms(original)
            transformed_total_rms = audio_rms(transformed)
            original_active_rms = v2.active_rms(original, mask)
            transformed_active_rms = v2.active_rms(transformed, mask)
            transformed_segments = frozen_vad_segments(models, transformed, SAMPLE_RATE)
            transformed_mask = v1.interval_mask(transformed_segments, len(transformed))
            vad_iou = v1.mask_iou(mask, transformed_mask)
            lexical = v2.leakage(
                cue.get("asr_transcript", ""), detail["asr_text"],
                detail["asr_confidence"],
            )
            source_video_hash = encoded_video_hash(source)
            transformed_video_hash = encoded_video_hash(destination)
            detail.update(generation)
            detail.update(lexical)
            detail.update({
                "source_audio_codec": source_info["audio_codec"],
                "output_audio_codec": probe(destination)["audio_codec"],
                "original_total_rms": original_total_rms,
                "transformed_total_rms": transformed_total_rms,
                "total_rms_diff_db": db_ratio(transformed_total_rms, original_total_rms),
                "original_active_rms": original_active_rms,
                "transformed_active_rms": transformed_active_rms,
                "active_rms_diff_db": db_ratio(transformed_active_rms, original_active_rms),
                "generated_support_iou": generated_support_iou,
                "vad_iou": vad_iou,
                "original_vad_segments_samples": segments,
                "transformed_vad_segments_samples": transformed_segments,
                "recomputed_speech_start_sec": recomputed_start,
                "recomputed_speech_end_sec": recomputed_end,
                "frozen_speech_start_sec": float(cue["speech_start_sec"]),
                "frozen_speech_end_sec": float(cue["speech_end_sec"]),
                "vad_envelope_max_abs_diff_sec": envelope_max_abs_diff,
                "source_encoded_video_sha256": source_video_hash,
                "transformed_encoded_video_sha256": transformed_video_hash,
                "video_stream_identical": source_video_hash == transformed_video_hash,
            })
            reasons = list(detail["failure_reasons"])
            # The inherited QC expected a fixed 44.1-kHz output.  Under the
            # amended contract, a source-matched 44.1/48-kHz rate is correct.
            inherited_audio_format = "T_rev:audio_format"
            if (
                inherited_audio_format in reasons
                and output_info["sample_rate"] == source_info["sample_rate"]
                and output_info["channels"] == 2
            ):
                reasons.remove(inherited_audio_format)
            stream_duration_diff = (
                output_info["audio_duration_sec"] - source_info["audio_duration_sec"]
            )
            if envelope_max_abs_diff > 0.25:
                reasons.append("T_rev:vad_envelope_mismatch")
            if generated_support_iou < 0.99:
                reasons.append("T_rev:generated_support")
            if vad_iou < 0.30:
                reasons.append("T_rev:vad_iou")
            if abs(detail["active_rms_diff_db"]) > 1.0:
                reasons.append("T_rev:active_rms")
            if lexical["lexical_leak"]:
                reasons.append("T_rev:lexical_leak")
            if source_video_hash != transformed_video_hash:
                reasons.append("T_rev:video_stream_changed")
            if detail["output_audio_codec"] != "aac":
                reasons.append("T_rev:audio_codec")
            if output_info["sample_rate"] != source_info["sample_rate"]:
                reasons.append("T_rev:source_sample_rate")
            if abs(stream_duration_diff) > 0.001:
                reasons.append("T_rev:audio_stream_duration")
            reasons = sorted(set(reasons))
            detail.update({
                "source_sample_rate": source_info["sample_rate"],
                "output_sample_rate": output_info["sample_rate"],
                "source_audio_duration_sec": source_info["audio_duration_sec"],
                "output_audio_duration_sec": output_info["audio_duration_sec"],
                "audio_stream_duration_diff_sec": stream_duration_diff,
                "source_audio_contract_match": (
                    output_info["sample_rate"] == source_info["sample_rate"]
                    and abs(stream_duration_diff) <= 0.001
                ),
            })
            detail["failure_reasons"] = reasons
            detail["qc_pass"] = not reasons
            return {
                "sample_id": sample_id,
                "video_id": row["video_id"],
                "source_path": str(source.resolve()),
                "source_sha256": cue["source_file_sha256"],
                "T_rev_path": str(destination.resolve()),
                "T_rev_sha256": sha256(destination),
                "qc_pass_T_rev": detail["qc_pass"],
                "failure_reasons": reasons,
                "condition_details": detail,
                "transformation": {
                    "vad": "silero-vad 6.2.1 threshold=0.5 min_speech=250ms min_silence=100ms pad=30ms",
                    "separation": "demucs 4.0.1 htdemucs two-stems vocals",
                    "demucs_package_version": importlib.metadata.version("demucs"),
                    "carrier": "vocal stem reversed independently within each VAD segment",
                    "background": "unmodified Demucs residual",
                    "audio_encode": "AAC 192kbps stereo at source sample rate and audio-stream duration",
                    "video_encode": "stream copy",
                    "note": "time-reversed nonlexical speech-like control; not pure lexical isolation",
                },
                "cue_asr_transcript": cue.get("asr_transcript"),
                "cue_asr_confidence": cue.get("asr_confidence"),
                "protocol_path": str(PROTOCOL),
                "protocol_sha256": PROTOCOL_SHA256,
                "manifest_sha256": MANIFEST_SHA256,
                "cue_index_sha256": CUES_SHA256,
                "preparation_code_sha256": sha256(Path(__file__)),
                "elapsed_sec": time.perf_counter() - started,
                "version": VERSION,
            }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "video_id": row["video_id"],
            "source_path": str(source.resolve()),
            "T_rev_path": str(destination.resolve()),
            "qc_pass_T_rev": False,
            "failure_reasons": [f"pipeline:{type(exc).__name__}:{exc}"],
            "protocol_path": str(PROTOCOL),
            "protocol_sha256": PROTOCOL_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "cue_index_sha256": CUES_SHA256,
            "preparation_code_sha256": sha256(Path(__file__)),
            "elapsed_sec": time.perf_counter() - started,
            "version": VERSION,
        }


def main() -> None:
    args = arguments()
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("invalid shard")
    if sha256(args.manifest) != MANIFEST_SHA256:
        raise RuntimeError("frozen manifest hash mismatch")
    if sha256(args.cues) != CUES_SHA256:
        raise RuntimeError("frozen cue-index hash mismatch")
    if sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash mismatch")
    manifest = [row for row in read_jsonl(args.manifest) if row["task"] == "CMTB"]
    if len(manifest) != 291 or len({row["video_id"] for row in manifest}) != 291:
        raise RuntimeError("expected 291 unique CMTB videos")
    cues = {row["video_id"]: row for row in read_jsonl(args.cues)}
    rows = [
        row for row in manifest
        if stable_shard(row["sample_id"], args.num_shards) == args.shard_id
    ]
    if args.limit is not None:
        rows = rows[:args.limit]
    existing = {
        row["sample_id"]: row for row in read_jsonl(args.result)
    } if args.result.exists() else {}
    models = v2.Models(args.asr_model, args.device)
    for index, row in enumerate(rows, start=1):
        prior = existing.get(row["sample_id"])
        if prior and not args.force:
            media = Path(prior.get("T_rev_path", ""))
            if media.is_file() and prior.get("T_rev_sha256") == sha256(media):
                print(f"[{index}/{len(rows)}] reuse {row['sample_id']}", flush=True)
                continue
        print(f"[{index}/{len(rows)}] process {row['sample_id']}", flush=True)
        result = process_one(
            row, cues[row["video_id"]], args.output_root,
            models, args.device, args.force,
        )
        existing[row["sample_id"]] = result
        atomic_jsonl(args.result, list(existing.values()))
        print(
            row["sample_id"], "PASS" if result["qc_pass_T_rev"] else "FAIL",
            result["failure_reasons"], f"{result['elapsed_sec']:.1f}s", flush=True,
        )
    print(f"result={args.result} rows={len(existing)}", flush=True)


if __name__ == "__main__":
    main()
