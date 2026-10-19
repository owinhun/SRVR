#!/usr/bin/env python3
"""Generate B, T_env, and T_rev for the fixed 48-sample CMTB pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample_poly
from sklearn.feature_extraction.text import TfidfVectorizer

from phase0 import DEMUCS_MODEL, SAMPLE_RATE, audio_rms, match_shape, probe, read_jsonl, run, sha256

CONDITIONS = ("B", "T_env", "T_rev")
SEED = 20260728
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "in", "is",
    "it", "its", "of", "on", "or", "she", "some", "that", "the", "their", "them",
    "then", "there", "they", "this", "to", "was", "were", "what", "when", "where",
    "which", "while", "who", "with", "you", "your",
}


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


def content_words(text: str) -> set[str]:
    return {
        word for word in normalize_text(text).split()
        if len(word) >= 3 and word not in STOPWORDS
    }


def lexical_similarity(reference: str, candidate: str) -> float:
    """Character n-gram proxy used because sentence-transformers is incompatible here."""
    if not reference.strip() or not candidate.strip():
        return 0.0
    try:
        matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(
            [reference, candidate]
        )
        return float((matrix[0] @ matrix[1].T).toarray()[0, 0])
    except ValueError:
        return 0.0


def stable_seed(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()[:8], "big")


def exact_audio(signal: np.ndarray, frames: int, channels: int) -> np.ndarray:
    return match_shape(signal, frames, channels)


def interval_mask(segments: list[dict[str, int]], frames: int) -> np.ndarray:
    result = np.zeros(frames, dtype=bool)
    for segment in segments:
        result[max(0, segment["start"]):min(frames, segment["end"])] = True
    return result


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


class Models:
    def __init__(self, asr_model: str, device: str) -> None:
        import torch
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad

        self.torch = torch
        self.vad = load_silero_vad()
        self.asr = WhisperModel(
            asr_model,
            device="cuda" if device == "cuda" else "cpu",
            compute_type="float16" if device == "cuda" else "int8",
        )
        self.asr_model = asr_model

    def vad_segments(self, signal: np.ndarray, sr: int) -> list[dict[str, int]]:
        from silero_vad import get_speech_timestamps

        mono = signal.mean(axis=1) if signal.ndim == 2 else signal
        if sr != 16000:
            divisor = math.gcd(sr, 16000)
            mono = resample_poly(mono, 16000 // divisor, sr // divisor)
        tensor = self.torch.from_numpy(mono.astype(np.float32))
        raw = get_speech_timestamps(
            tensor, self.vad, sampling_rate=16000, threshold=0.5,
            min_speech_duration_ms=150, min_silence_duration_ms=100,
            speech_pad_ms=30, return_seconds=False,
        )
        scale = sr / 16000
        return [
            {"start": round(segment["start"] * scale), "end": round(segment["end"] * scale)}
            for segment in raw
        ]

    def transcribe(self, path: Path) -> str:
        segments, _ = self.asr.transcribe(
            str(path), beam_size=5, vad_filter=False, language="en"
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def speech_shaped_noise(stem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.zeros_like(stem)
    for channel in range(stem.shape[1]):
        spectrum = np.fft.rfft(stem[:, channel])
        phase = rng.uniform(-math.pi, math.pi, len(spectrum))
        phase[0] = 0.0
        if len(stem) % 2 == 0:
            phase[-1] = 0.0
        output[:, channel] = np.fft.irfft(
            np.abs(spectrum) * np.exp(1j * phase), n=len(stem)
        ).astype(np.float32)
    return output


def active_rms(signal: np.ndarray, mask: np.ndarray) -> float:
    return audio_rms(signal[mask]) if mask.any() else 0.0


def match_active_rms(signal: np.ndarray, mask: np.ndarray, target: float) -> np.ndarray:
    current = active_rms(signal, mask)
    if current > 1e-12:
        signal[mask] *= target / current
    return signal


def match_total_rms_and_peak(signal: np.ndarray, target: float) -> np.ndarray:
    current = audio_rms(signal)
    if current > 1e-12:
        signal *= target / current
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 0.999:
        signal *= 0.999 / peak
    return signal.astype(np.float32)


def reverse_within_segments(stem: np.ndarray, segments: list[dict[str, int]]) -> np.ndarray:
    output = np.zeros_like(stem)
    for segment in segments:
        start, end = segment["start"], segment["end"]
        output[start:end] = stem[start:end][::-1]
    return output


def mux(source: Path, wav: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        probe(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    try:
        run([
            "ffmpeg", "-v", "error", "-y", "-i", str(source), "-i", str(wav),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-map_metadata", "0",
            "-movflags", "+faststart", "-shortest", str(temporary),
        ])
        probe(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def decode_audio(media: Path, output: Path) -> np.ndarray:
    run([
        "ffmpeg", "-v", "error", "-y", "-i", str(media), "-vn",
        "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(output),
    ])
    signal, _ = sf.read(output, dtype="float32", always_2d=True)
    return signal


def full_media_check(media: Path) -> None:
    run(["ffprobe", "-v", "error", str(media)], capture=True)
    run([
        "ffmpeg", "-v", "error", "-i", str(media),
        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
    ], capture=True)


def condition_qc(
    condition: str,
    media: Path,
    original_duration: float,
    original_rms: float,
    intended_mask: np.ndarray,
    original_transcript: str,
    models: Models,
    tmp: Path,
) -> dict[str, Any]:
    full_media_check(media)
    media_probe = probe(media)
    wav = tmp / f"qc_{condition}.wav"
    signal = decode_audio(media, wav)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    condition_rms = audio_rms(signal)
    rms_diff_db = 20 * math.log10(max(condition_rms, 1e-12) / max(original_rms, 1e-12))
    detected_segments = models.vad_segments(signal, SAMPLE_RATE)
    detected_mask = interval_mask(detected_segments, len(signal))
    intended = intended_mask[:len(signal)]
    if len(intended) < len(signal):
        intended = np.pad(intended, (0, len(signal) - len(intended)))
    vad_iou = mask_iou(intended, detected_mask)
    transcript = models.transcribe(wav)
    leaked_words = sorted(content_words(original_transcript).intersection(content_words(transcript)))
    similarity = lexical_similarity(original_transcript, transcript)
    lexical_leak = bool(leaked_words) or (
        len(content_words(transcript)) >= 2 and similarity >= 0.35
    )
    duration_diff = media_probe["duration_sec"] - original_duration
    reasons = []
    if abs(duration_diff) > 0.05:
        reasons.append(f"{condition}:duration")
    if media_probe["sample_rate"] != SAMPLE_RATE or media_probe["channels"] != 2:
        reasons.append(f"{condition}:audio_format")
    if peak >= 0.9999:
        reasons.append(f"{condition}:clipping")
    if condition in ("T_env", "T_rev") and abs(rms_diff_db) > 1.0:
        reasons.append(f"{condition}:rms")
    if condition == "T_rev" and vad_iou < 0.30:
        reasons.append(f"{condition}:vad_iou")
    if lexical_leak:
        reasons.append(f"{condition}:lexical_leak")
    return {
        "path": str(media.resolve()),
        "sha256": sha256(media),
        "duration_match": abs(duration_diff) <= 0.05,
        "duration_diff_sec": duration_diff,
        "sample_rate": media_probe["sample_rate"],
        "channels": media_probe["channels"],
        "peak": peak,
        "clipping": peak >= 0.9999,
        "rms": condition_rms,
        "rms_diff_db": rms_diff_db,
        "vad_iou": vad_iou,
        "vad_segments_samples": detected_segments,
        "asr_text": transcript,
        "lexical_overlap_words": leaked_words,
        "lexical_similarity_proxy": similarity,
        "lexical_leak": lexical_leak,
        "qc_pass": not reasons,
        "failure_reasons": reasons,
    }


def failure_row(sample_id: str, paths: dict[str, Path], base_pass: bool, reason: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "B_path": str(paths["B"].resolve()),
        "T_env_path": str(paths["T_env"].resolve()),
        "T_rev_path": str(paths["T_rev"].resolve()),
        "base_o_proc_qc_pass": base_pass,
        "qc_pass_B": False,
        "qc_pass_T_env": False,
        "qc_pass_T_rev": False,
        "failure_reasons": [reason],
    }


def flatten_row(sample_id: str, paths: dict[str, Path], base_pass: bool,
                original_transcript: str, qc: dict[str, dict[str, Any]],
                original_segments: list[dict[str, int]]) -> dict[str, Any]:
    failures = sorted({
        reason for condition in CONDITIONS for reason in qc[condition]["failure_reasons"]
    })
    return {
        "sample_id": sample_id,
        "B_path": str(paths["B"].resolve()),
        "T_env_path": str(paths["T_env"].resolve()),
        "T_rev_path": str(paths["T_rev"].resolve()),
        "base_o_proc_qc_pass": base_pass,
        "duration_match_B": qc["B"]["duration_match"],
        "duration_match_T_env": qc["T_env"]["duration_match"],
        "duration_match_T_rev": qc["T_rev"]["duration_match"],
        "rms_diff_db_B": qc["B"]["rms_diff_db"],
        "rms_diff_db_T_env": qc["T_env"]["rms_diff_db"],
        "rms_diff_db_T_rev": qc["T_rev"]["rms_diff_db"],
        "clipping_B": qc["B"]["clipping"],
        "clipping_T_env": qc["T_env"]["clipping"],
        "clipping_T_rev": qc["T_rev"]["clipping"],
        "vad_iou_B": qc["B"]["vad_iou"],
        "vad_iou_T_env": qc["T_env"]["vad_iou"],
        "vad_iou_T_rev": qc["T_rev"]["vad_iou"],
        "generated_support_iou_T_env": qc["T_env"]["generated_support_iou"],
        "generated_support_iou_T_rev": qc["T_rev"]["generated_support_iou"],
        "asr_text_original": original_transcript,
        "asr_text_B": qc["B"]["asr_text"],
        "asr_text_T_env": qc["T_env"]["asr_text"],
        "asr_text_T_rev": qc["T_rev"]["asr_text"],
        "lexical_leak_B": qc["B"]["lexical_leak"],
        "lexical_leak_T_env": qc["T_env"]["lexical_leak"],
        "lexical_leak_T_rev": qc["T_rev"]["lexical_leak"],
        "lexical_similarity_proxy_B": qc["B"]["lexical_similarity_proxy"],
        "lexical_similarity_proxy_T_env": qc["T_env"]["lexical_similarity_proxy"],
        "lexical_similarity_proxy_T_rev": qc["T_rev"]["lexical_similarity_proxy"],
        "qc_pass_B": qc["B"]["qc_pass"],
        "qc_pass_T_env": qc["T_env"]["qc_pass"],
        "qc_pass_T_rev": qc["T_rev"]["qc_pass"],
        "failure_reasons": failures,
        "original_vad_segments_samples": original_segments,
        "condition_details": qc,
        "asr_model": "Systran/faster-whisper-small.en",
        "semantic_check_method": "ASR content-word overlap + character TF-IDF proxy",
        "T_rev_note": "speech-like time-reversed control; not pure timing-only",
    }


def process_one(row: dict[str, Any], base_pass: bool, output_root: Path,
                models: Models, device: str, force: bool) -> dict[str, Any]:
    sample_id = row["sample_id"]
    source = Path(row["video_path"])
    sample_dir = output_root / sample_id
    paths = {condition: sample_dir / f"{condition}.mp4" for condition in CONDITIONS}
    try:
        original_probe = probe(source)
        sample_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"phase1_{sample_id}_", dir="/tmp") as tmp_name:
            tmp = Path(tmp_name)
            original_wav = tmp / "original.wav"
            run([
                "ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn",
                "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le",
                str(original_wav),
            ])
            original, sr = sf.read(original_wav, dtype="float32", always_2d=True)
            original_segments = models.vad_segments(original, sr)
            intended_mask = interval_mask(original_segments, len(original))
            if not intended_mask.any():
                raise RuntimeError("no_original_speech_detected")
            original_transcript = models.transcribe(original_wav)
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
            if (sr, vocals_sr, residual_sr) != (SAMPLE_RATE, SAMPLE_RATE, SAMPLE_RATE):
                raise RuntimeError(f"unexpected_stem_sample_rates:{sr},{vocals_sr},{residual_sr}")
            vocals = exact_audio(vocals, len(original), 2)
            residual = exact_audio(residual, len(original), 2)
            gated_vocals = vocals * intended_mask[:, None]
            target_active = active_rms(gated_vocals, intended_mask)
            rng = np.random.default_rng(stable_seed(sample_id))
            noise = speech_shaped_noise(gated_vocals, rng)
            envelope = np.sqrt(
                uniform_filter1d(
                    np.square(gated_vocals), size=max(1, round(SAMPLE_RATE * 0.025)),
                    axis=0, mode="nearest",
                ) + 1e-12
            )
            envelope /= max(float(np.sqrt(np.mean(np.square(envelope)))), 1e-12)
            env_component = noise * envelope * intended_mask[:, None]
            env_component = match_active_rms(env_component, intended_mask, target_active)
            reverse_component = reverse_within_segments(gated_vocals, original_segments)
            reverse_component = match_active_rms(reverse_component, intended_mask, target_active)
            target_total = audio_rms(original)
            products = {
                "B": match_total_rms_and_peak(residual.copy(), audio_rms(residual)),
                "T_env": match_total_rms_and_peak(residual + env_component, target_total),
                "T_rev": match_total_rms_and_peak(residual + reverse_component, target_total),
            }
            for condition, signal in products.items():
                wav = tmp / f"{condition}.wav"
                sf.write(wav, exact_audio(signal, len(original), 2), SAMPLE_RATE, subtype="PCM_24")
                mux(source, wav, paths[condition], force)
            qc = {
                condition: condition_qc(
                    condition, paths[condition], original_probe["duration_sec"],
                    target_total, intended_mask, original_transcript, models, tmp,
                )
                for condition in CONDITIONS
            }
            for condition, component in (("T_env", env_component), ("T_rev", reverse_component)):
                support = np.any(np.abs(component) > 1e-8, axis=1)
                support_iou = mask_iou(intended_mask, support)
                qc[condition]["generated_support_iou"] = support_iou
                if support_iou < 0.99:
                    reason = f"{condition}:generated_support_iou"
                    qc[condition]["failure_reasons"].append(reason)
                    qc[condition]["qc_pass"] = False
            return flatten_row(
                sample_id, paths, base_pass, original_transcript, qc, original_segments
            )
    except Exception as exc:
        return failure_row(
            sample_id, paths, base_pass, f"pipeline:{type(exc).__name__}:{exc}"
        )


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_reports(rows: list[dict[str, Any]], summary_path: Path, listen_path: Path) -> None:
    complete = [row for row in rows if "condition_details" in row]
    summary = {
        "sample_count": len(rows),
        "complete_count": len(complete),
        "pipeline_failure_count": len(rows) - len(complete),
        "qc_pass_B": sum(bool(row.get("qc_pass_B")) for row in rows),
        "qc_pass_T_env": sum(bool(row.get("qc_pass_T_env")) for row in rows),
        "qc_pass_T_rev": sum(bool(row.get("qc_pass_T_rev")) for row in rows),
        "failure_reason_counts": {},
        "asr_model": "Systran/faster-whisper-small.en",
        "demucs_model": f"demucs 4.1.0/{DEMUCS_MODEL}",
        "semantic_check_method": "ASR content-word overlap + character TF-IDF proxy",
        "T_rev_note": "speech-like time-reversed control; not pure timing-only",
    }
    reasons: dict[str, int] = {}
    for row in rows:
        for reason in row.get("failure_reasons", []):
            reasons[reason] = reasons.get(reason, 0) + 1
    summary["failure_reason_counts"] = reasons
    for condition in CONDITIONS:
        values = [
            row[f"rms_diff_db_{condition}"] for row in complete
            if row.get(f"rms_diff_db_{condition}") is not None
        ]
        ious = [
            row[f"vad_iou_{condition}"] for row in complete
            if row.get(f"vad_iou_{condition}") is not None
        ]
        summary[f"{condition}_rms_diff_db"] = {
            "min": min(values), "mean": sum(values) / len(values), "max": max(values)
        } if values else None
        summary[f"{condition}_vad_iou"] = {
            "min": min(ious), "mean": sum(ious) / len(ious), "max": max(ious)
        } if ious else None
        if condition in ("T_env", "T_rev"):
            supports = [row[f"generated_support_iou_{condition}"] for row in complete if row.get(f"generated_support_iou_{condition}") is not None]
            summary[f"{condition}_generated_support_iou"] = {"min": min(supports), "mean": sum(supports) / len(supports), "max": max(supports)} if supports else None
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    rng = random.Random(SEED)
    failures = [row for row in rows if row.get("failure_reasons")]
    borderline = [
        row for row in complete
        if any(abs(row[f"rms_diff_db_{c}"]) > 0.5 for c in ("T_env", "T_rev"))
        or row["generated_support_iou_T_env"] < 0.99
        or row["generated_support_iou_T_rev"] < 0.99
        or row["vad_iou_T_rev"] < 0.60
    ]
    random_examples = rng.sample(complete, min(8, len(complete)))
    selected: dict[str, tuple[str, dict[str, Any]]] = {}
    for priority, candidates in (
        ("failure", failures), ("borderline", borderline), ("random", random_examples)
    ):
        for row in candidates:
            selected.setdefault(row["sample_id"], (priority, row))
    listen_path.parent.mkdir(parents=True, exist_ok=True)
    with listen_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "priority", "sample_id", "condition", "path", "qc_pass",
            "rms_diff_db", "vad_iou", "asr_text", "lexical_leak", "failure_reasons",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for priority, row in selected.values():
            for condition in CONDITIONS:
                detail = row.get("condition_details", {}).get(condition, {})
                writer.writerow({
                    "priority": priority,
                    "sample_id": row["sample_id"],
                    "condition": condition,
                    "path": row.get(f"{condition}_path", ""),
                    "qc_pass": row.get(f"qc_pass_{condition}", False),
                    "rms_diff_db": row.get(f"rms_diff_db_{condition}", ""),
                    "vad_iou": row.get(f"vad_iou_{condition}", ""),
                    "asr_text": row.get(f"asr_text_{condition}", ""),
                    "lexical_leak": row.get(f"lexical_leak_{condition}", ""),
                    "failure_reasons": ";".join(detail.get("failure_reasons", row.get("failure_reasons", []))),
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/svhalluc/data/phase0_cmtb.jsonl"))
    parser.add_argument("--base-qc", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase0.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/svhalluc/outputs/phase1_audio_pilot"))
    parser.add_argument("--result", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot_summary.json"))
    parser.add_argument("--listen", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot_listen.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest)
    if len(manifest) != 48:
        raise ValueError(f"Expected frozen 48-sample manifest, found {len(manifest)}")
    if args.limit is not None:
        manifest = manifest[:args.limit]
    base_qc = {row["sample_id"]: row["qc_pass"] for row in read_jsonl(args.base_qc)}
    existing = {
        row["sample_id"]: row for row in read_jsonl(args.result)
    } if args.result.exists() else {}
    models = Models(args.asr_model, args.device)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(manifest, start=1):
        sample_id = row["sample_id"]
        prior = existing.get(sample_id)
        if prior and not args.force and all(Path(prior.get(f"{c}_path", "")).is_file() for c in CONDITIONS):
            print(f"[{index}/{len(manifest)}] reuse {sample_id}", flush=True)
            results.append(prior)
        else:
            print(f"[{index}/{len(manifest)}] process {sample_id}", flush=True)
            results.append(process_one(
                row, base_qc.get(sample_id, False), args.output_root,
                models, args.device, args.force,
            ))
        merged = {**existing, **{item["sample_id"]: item for item in results}}
        atomic_write_jsonl(args.result, [merged[key] for key in sorted(merged)])
    final = read_jsonl(args.result)
    write_reports(final, args.summary, args.listen)
    print(f"Results: {args.result}")
    print(f"Summary: {args.summary}")
    print(f"Listening list: {args.listen}")


if __name__ == "__main__":
    main()
