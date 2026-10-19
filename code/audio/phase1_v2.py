#!/usr/bin/env python3
"""Non-destructive v2 regeneration of the 48-sample Phase-1 audio pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d

import phase1_pilot as v1
from phase0 import DEMUCS_MODEL, SAMPLE_RATE, audio_rms, probe, read_jsonl, run, sha256

CONDITIONS = ("B", "T_env", "T_rev")
SEED = 20260728
GENERIC = v1.STOPWORDS | {
    "all", "also", "back", "because", "but", "can", "come", "could", "did",
    "doesn", "don", "down", "even", "everything", "get", "getting", "go",
    "going", "good", "got", "here", "i'm", "into", "it’s", "it's", "just",
    "know", "like", "little", "make", "more", "much", "now", "one", "only",
    "really", "right", "see", "something", "still", "take", "than", "those",
    "through", "time", "too", "up", "very", "want", "well", "will", "would",
}


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold().replace("’", "'"))


def core_words(text: str) -> list[str]:
    return [word for word in normalize(text) if len(word) >= 3 and word not in GENERIC]


def core_bigrams(text: str) -> set[tuple[str, str]]:
    words = core_words(text)
    return set(zip(words, words[1:]))


class Models(v1.Models):
    def transcribe_scored(self, path: Path) -> tuple[str, float, float]:
        segments, _ = self.asr.transcribe(
            str(path), beam_size=5, vad_filter=False, language="en"
        )
        pieces, logprobs, no_speech = [], [], []
        for segment in segments:
            pieces.append(segment.text.strip())
            weight = max(1.0, float(segment.end - segment.start))
            logprobs.append((float(segment.avg_logprob), weight))
            no_speech.append(float(segment.no_speech_prob))
        if not logprobs:
            return "", 0.0, 1.0
        weighted = sum(value * weight for value, weight in logprobs) / sum(
            weight for _, weight in logprobs
        )
        confidence = float(math.exp(min(0.0, weighted)))
        return " ".join(piece for piece in pieces if piece).strip(), confidence, (
            sum(no_speech) / len(no_speech)
        )


def active_rms(signal: np.ndarray, mask: np.ndarray) -> float:
    return audio_rms(signal[mask]) if mask.any() else 0.0


def carrier_peak_control(
    carrier: np.ndarray, background: np.ndarray, mask: np.ndarray, target_rms: float
) -> tuple[np.ndarray, dict[str, float]]:
    """Constrain only the carrier while preserving background gain."""
    output = carrier.copy()
    scale = max(2.5 * target_rms, 1e-4)
    output[mask] = np.tanh(output[mask] / scale) * scale
    output[~mask] = 0.0
    current = active_rms(output, mask)
    if current > 1e-12:
        output[mask] *= target_rms / current
    lower, upper = -0.985 - background, 0.985 - background
    for _ in range(40):
        output = np.minimum(np.maximum(output, lower), upper)
        output[~mask] = 0.0
        current = active_rms(output, mask)
        if target_rms <= 1e-12 or abs(20 * math.log10(max(current, 1e-12) / target_rms)) <= 0.05:
            break
        gain = min(1.15, target_rms / max(current, 1e-12))
        output[mask] *= gain
    achieved = active_rms(output, mask)
    return output.astype(np.float32), {
        "target_active_rms": target_rms,
        "generated_active_rms": achieved,
        "generated_active_rms_diff_db": 20
        * math.log10(max(achieved, 1e-12) / max(target_rms, 1e-12)),
        "mixed_peak": float(np.max(np.abs(background + output))),
    }


def background_type(signal: np.ndarray) -> dict[str, Any]:
    mono = signal.mean(axis=1)
    rms = audio_rms(mono)
    dbfs = 20 * math.log10(max(rms, 1e-12))
    if dbfs < -42:
        label = "near_silent"
        harmonic_ratio = percussive_ratio = 0.0
    else:
        # Downsample for efficient, deterministic coarse classification.
        y = librosa.resample(mono, orig_sr=SAMPLE_RATE, target_sr=16000)
        harmonic, percussive = librosa.effects.hpss(y)
        total = max(audio_rms(y), 1e-12)
        harmonic_ratio = audio_rms(harmonic) / total
        percussive_ratio = audio_rms(percussive) / total
        if harmonic_ratio >= 0.60 and percussive_ratio < 0.45:
            label = "bgm"
        elif percussive_ratio >= 0.55 and harmonic_ratio < 0.45:
            label = "natural_sound"
        else:
            label = "mixed"
    return {
        "background_type_auto": label,
        "background_rms_dbfs": dbfs,
        "harmonic_rms_ratio": harmonic_ratio,
        "percussive_rms_ratio": percussive_ratio,
        "needs_manual_confirmation": True,
    }


def leakage(
    reference: str, candidate: str, confidence: float
) -> dict[str, Any]:
    reference_words = set(core_words(reference))
    candidate_words = set(core_words(candidate))
    overlap = sorted(reference_words & candidate_words)
    phrases = sorted(" ".join(pair) for pair in core_bigrams(reference) & core_bigrams(candidate))
    similarity = v1.lexical_similarity(reference, candidate)
    strong = bool(phrases) or (
        len(overlap) >= 2 and confidence >= 0.25
    ) or (
        similarity >= 0.35 and confidence >= 0.35 and len(candidate_words) >= 3
    )
    return {
        "lexical_leak": strong,
        "core_overlap_words": overlap,
        "matching_core_bigrams": phrases,
        "lexical_similarity_proxy": similarity,
        "asr_confidence": confidence,
    }


def hardlink_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        import shutil

        shutil.copy2(source, destination)


def decode(media: Path, wav: Path) -> np.ndarray:
    return v1.decode_audio(media, wav)


def media_qc(
    condition: str,
    media: Path,
    original_duration: float,
    models: Models,
    tmp: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    v1.full_media_check(media)
    info = probe(media)
    wav = tmp / f"qc_{condition}.wav"
    signal = decode(media, wav)
    transcript, confidence, no_speech = models.transcribe_scored(wav)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    reasons = []
    if abs(info["duration_sec"] - original_duration) > 0.05:
        reasons.append(f"{condition}:duration")
    if info["sample_rate"] != SAMPLE_RATE or info["channels"] != 2:
        reasons.append(f"{condition}:audio_format")
    if peak >= 0.9999:
        reasons.append(f"{condition}:clipping")
    return {
        "path": str(media.resolve()),
        "sha256": sha256(media),
        "duration_diff_sec": info["duration_sec"] - original_duration,
        "duration_match": abs(info["duration_sec"] - original_duration) <= 0.05,
        "sample_rate": info["sample_rate"],
        "channels": info["channels"],
        "peak": peak,
        "clipping": peak >= 0.9999,
        "rms": audio_rms(signal),
        "asr_text": transcript,
        "asr_confidence": confidence,
        "asr_no_speech_probability": no_speech,
        "failure_reasons": reasons,
    }, signal


def checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    v1.atomic_write_jsonl(path, sorted(rows, key=lambda row: row["sample_id"]))


def process_one(
    row: dict[str, Any],
    v1_row: dict[str, Any],
    output_root: Path,
    models: Models,
    device: str,
    force: bool,
) -> dict[str, Any]:
    sample_id = row["sample_id"]
    source = Path(row["video_path"])
    sample_dir = output_root / sample_id
    paths = {condition: sample_dir / f"{condition}.mp4" for condition in CONDITIONS}
    sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=f"phase1v2_{sample_id}_", dir="/tmp") as name:
            tmp = Path(name)
            original_wav = tmp / "original.wav"
            run([
                "ffmpeg", "-v", "error", "-y", "-i", str(source), "-vn",
                "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le",
                str(original_wav),
            ])
            original, sr = sf.read(original_wav, dtype="float32", always_2d=True)
            segments = models.vad_segments(original, sr)
            mask = v1.interval_mask(segments, len(original))
            if not mask.any():
                raise RuntimeError("no_original_speech_detected")
            original_text, original_confidence, _ = models.transcribe_scored(original_wav)
            run([
                "python", "-m", "demucs", "--two-stems", "vocals",
                "-n", DEMUCS_MODEL, "-d", device, "-o", str(tmp / "demucs"),
                str(original_wav),
            ])
            stem_dir = tmp / "demucs" / DEMUCS_MODEL / original_wav.stem
            vocals, vocals_sr = sf.read(stem_dir / "vocals.wav", dtype="float32", always_2d=True)
            residual, residual_sr = sf.read(
                stem_dir / "no_vocals.wav", dtype="float32", always_2d=True
            )
            if (sr, vocals_sr, residual_sr) != (SAMPLE_RATE, SAMPLE_RATE, SAMPLE_RATE):
                raise RuntimeError("unexpected_stem_sample_rate")
            vocals = v1.exact_audio(vocals, len(original), 2)
            residual = v1.exact_audio(residual, len(original), 2)
            gated = vocals * mask[:, None]
            target = active_rms(gated, mask)
            rng = np.random.default_rng(v1.stable_seed(sample_id))
            noise = v1.speech_shaped_noise(gated, rng)
            envelope = np.sqrt(
                uniform_filter1d(
                    np.square(gated), size=max(1, round(SAMPLE_RATE * 0.025)),
                    axis=0, mode="nearest",
                ) + 1e-12
            )
            envelope /= max(float(np.sqrt(np.mean(np.square(envelope)))), 1e-12)
            env_raw = noise * envelope * mask[:, None]
            env_raw = v1.match_active_rms(env_raw, mask, target)
            rev_raw = v1.reverse_within_segments(gated, segments)
            rev_raw = v1.match_active_rms(rev_raw, mask, target)
            env_carrier, env_generation = carrier_peak_control(env_raw, residual, mask, target)
            rev_carrier, rev_generation = carrier_peak_control(rev_raw, residual, mask, target)
            products = {
                "T_env": residual + env_carrier,
                "T_rev": residual + rev_carrier,
            }
            # B is byte-identical to v1 and stored as a hardlink when possible.
            hardlink_or_copy(Path(v1_row["B_path"]), paths["B"])
            for condition, signal in products.items():
                if paths[condition].exists() and not force:
                    continue
                wav = tmp / f"{condition}.wav"
                sf.write(wav, signal, SAMPLE_RATE, subtype="PCM_24")
                v1.mux(source, wav, paths[condition], force)

            original_duration = probe(source)["duration_sec"]
            details, decoded = {}, {}
            for condition in CONDITIONS:
                details[condition], decoded[condition] = media_qc(
                    condition, paths[condition], original_duration, models, tmp
                )
            bg_type = background_type(decoded["B"])
            for condition in ("T_env", "T_rev"):
                generated = env_generation if condition == "T_env" else rev_generation
                encoded_carrier = decoded[condition][:len(decoded["B"])] - decoded["B"][:len(decoded[condition])]
                encoded_mask = mask[:len(encoded_carrier)]
                encoded_active = active_rms(encoded_carrier, encoded_mask)
                active_diff = 20 * math.log10(
                    max(encoded_active, 1e-12) / max(target, 1e-12)
                )
                outside = audio_rms(encoded_carrier[~encoded_mask])
                support_ratio = outside / max(encoded_active, 1e-12)
                details[condition].update(generated)
                details[condition].update({
                    "encoded_carrier_active_rms": encoded_active,
                    "active_rms_diff_db": active_diff,
                    "outside_inside_rms_ratio": support_ratio,
                    "generated_support_iou": v1.mask_iou(
                        mask, np.any(np.abs(
                            env_carrier if condition == "T_env" else rev_carrier
                        ) > 1e-8, axis=1)
                    ),
                })
                if abs(active_diff) > 0.5:
                    details[condition]["failure_reasons"].append(f"{condition}:active_rms")
                if details[condition]["generated_support_iou"] < 0.99:
                    details[condition]["failure_reasons"].append(f"{condition}:support")
                # Diagnostic only: B and each condition are independently AAC encoded.
                # Pre-encode generated_support_iou is the primary support QC.
                detected = models.vad_segments(decoded[condition], SAMPLE_RATE)
                details[condition]["vad_iou_descriptive"] = v1.mask_iou(
                    mask[:len(decoded[condition])],
                    v1.interval_mask(detected, len(decoded[condition])),
                )

            for condition in CONDITIONS:
                leak = leakage(
                    original_text, details[condition]["asr_text"],
                    details[condition]["asr_confidence"],
                )
                details[condition].update(leak)
                if leak["lexical_leak"]:
                    details[condition]["failure_reasons"].append(f"{condition}:lexical_leak")
                details[condition]["qc_pass"] = not details[condition]["failure_reasons"]
            failures = sorted({
                reason for condition in CONDITIONS
                for reason in details[condition]["failure_reasons"]
            })
            return {
                "sample_id": sample_id,
                "B_path": str(paths["B"].resolve()),
                "T_env_path": str(paths["T_env"].resolve()),
                "T_rev_path": str(paths["T_rev"].resolve()),
                "base_o_proc_qc_pass": v1_row["base_o_proc_qc_pass"],
                "qc_pass_B": details["B"]["qc_pass"],
                "qc_pass_T_env": details["T_env"]["qc_pass"],
                "qc_pass_T_rev": details["T_rev"]["qc_pass"],
                "active_rms_diff_db_T_env": details["T_env"]["active_rms_diff_db"],
                "active_rms_diff_db_T_rev": details["T_rev"]["active_rms_diff_db"],
                "generated_support_iou_T_env": details["T_env"]["generated_support_iou"],
                "generated_support_iou_T_rev": details["T_rev"]["generated_support_iou"],
                "vad_iou_T_env": details["T_env"]["vad_iou_descriptive"],
                "vad_iou_T_rev": details["T_rev"]["vad_iou_descriptive"],
                "lexical_leak_B": details["B"]["lexical_leak"],
                "lexical_leak_T_env": details["T_env"]["lexical_leak"],
                "lexical_leak_T_rev": details["T_rev"]["lexical_leak"],
                "asr_text_original": original_text,
                "asr_confidence_original": original_confidence,
                "asr_text_B": details["B"]["asr_text"],
                "asr_text_T_env": details["T_env"]["asr_text"],
                "asr_text_T_rev": details["T_rev"]["asr_text"],
                "background": bg_type,
                "failure_reasons": failures,
                "condition_details": details,
                "T_rev_note": "speech-like time-reversed control; not pure timing-only",
                "version": "phase1_audio_pilot_v2",
            }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "B_path": str(paths["B"].resolve()),
            "T_env_path": str(paths["T_env"].resolve()),
            "T_rev_path": str(paths["T_rev"].resolve()),
            "qc_pass_B": False, "qc_pass_T_env": False, "qc_pass_T_rev": False,
            "failure_reasons": [f"pipeline:{type(exc).__name__}:{exc}"],
            "version": "phase1_audio_pilot_v2",
        }


def reports(rows: list[dict[str, Any]], summary_path: Path, listen_path: Path,
            backgrounds_path: Path) -> None:
    complete = [row for row in rows if "condition_details" in row]
    reason_counts: dict[str, int] = {}
    for row in rows:
        for reason in row["failure_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary: dict[str, Any] = {
        "version": "phase1_audio_pilot_v2",
        "sample_count": len(rows),
        "complete_count": len(complete),
        "pipeline_failure_count": len(rows) - len(complete),
        "qc_pass_B": sum(bool(row.get("qc_pass_B")) for row in rows),
        "qc_pass_T_env": sum(bool(row.get("qc_pass_T_env")) for row in rows),
        "qc_pass_T_rev": sum(bool(row.get("qc_pass_T_rev")) for row in rows),
        "common_qc_pass": sum(
            bool(row.get("qc_pass_B") and row.get("qc_pass_T_env") and row.get("qc_pass_T_rev"))
            for row in rows
        ),
        "failure_reason_counts": reason_counts,
        "T_rev_note": "speech-like time-reversed control; not pure timing-only",
        "lexical_rule": "strong core content overlap/phrase or high-confidence similarity",
        "support_qc_rule": (
            "pre-encode generated support is primary; VAD and independently "
            "AAC-decoded outside/inside RMS are descriptive"
        ),
    }
    for condition in ("T_env", "T_rev"):
        for metric in ("active_rms_diff_db", "generated_support_iou", "vad_iou"):
            key = f"{metric}_{condition}"
            values = [row[key] for row in complete]
            summary[key] = {
                "min": min(values), "mean": sum(values) / len(values), "max": max(values)
            }
    labels: dict[str, int] = {}
    for row in complete:
        label = row["background"]["background_type_auto"]
        labels[label] = labels.get(label, 0) + 1
    summary["background_type_auto_counts"] = labels
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    with backgrounds_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "sample_id", "B_path", "background_type_auto", "background_rms_dbfs",
            "harmonic_rms_ratio", "percussive_rms_ratio", "manual_background_type",
            "needs_manual_confirmation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in complete:
            writer.writerow({
                "sample_id": row["sample_id"], "B_path": row["B_path"],
                **row["background"], "manual_background_type": "",
            })

    rng = random.Random(SEED)
    failures = [row for row in complete if row["failure_reasons"]]
    random_pass = rng.sample(
        [row for row in complete if not row["failure_reasons"]],
        min(5, sum(not row["failure_reasons"] for row in complete)),
    )
    selected: dict[str, tuple[str, dict[str, Any]]] = {}
    for priority, candidates in (("failure", failures), ("random_pass", random_pass)):
        for row in candidates:
            selected.setdefault(row["sample_id"], (priority, row))
    with listen_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "priority", "sample_id", "condition", "path", "qc_pass",
            "active_rms_diff_db", "generated_support_iou", "vad_iou_descriptive",
            "asr_text", "asr_confidence", "lexical_leak", "failure_reasons",
            "background_type_auto", "manual_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for priority, row in selected.values():
            for condition in CONDITIONS:
                detail = row["condition_details"][condition]
                writer.writerow({
                    "priority": priority, "sample_id": row["sample_id"],
                    "condition": condition, "path": row[f"{condition}_path"],
                    "qc_pass": detail["qc_pass"],
                    "active_rms_diff_db": detail.get("active_rms_diff_db", ""),
                    "generated_support_iou": detail.get("generated_support_iou", ""),
                    "vad_iou_descriptive": detail.get("vad_iou_descriptive", ""),
                    "asr_text": detail["asr_text"],
                    "asr_confidence": detail["asr_confidence"],
                    "lexical_leak": detail["lexical_leak"],
                    "failure_reasons": ";".join(detail["failure_reasons"]),
                    "background_type_auto": row["background"]["background_type_auto"],
                    "manual_notes": "",
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/svhalluc/data/phase0_cmtb.jsonl"))
    parser.add_argument("--v1-qc", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/svhalluc/outputs/phase1_audio_pilot_v2"))
    parser.add_argument("--result", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot_v2.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot_v2_summary.json"))
    parser.add_argument("--listen", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot_v2_listen.csv"))
    parser.add_argument("--backgrounds", type=Path, default=Path("/workspace/svhalluc/outputs/audio_phase1_pilot_v2_backgrounds.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default="small.en")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = read_jsonl(args.manifest)
    if len(manifest) != 48:
        raise ValueError("Phase-1 v2 is restricted to the frozen 48 samples")
    if args.limit is not None:
        manifest = manifest[:args.limit]
    v1_rows = {row["sample_id"]: row for row in read_jsonl(args.v1_qc)}
    existing = {
        row["sample_id"]: row for row in read_jsonl(args.result)
    } if args.result.exists() else {}
    models = Models(args.asr_model, args.device)
    results = []
    for index, row in enumerate(manifest, start=1):
        sample_id = row["sample_id"]
        prior = existing.get(sample_id)
        if prior and not args.force and all(
            Path(prior.get(f"{condition}_path", "")).is_file() for condition in CONDITIONS
        ):
            print(f"[{index}/{len(manifest)}] reuse {sample_id}", flush=True)
            for condition in ("T_env", "T_rev"):
                detail = prior["condition_details"][condition]
                detail["failure_reasons"] = [
                    reason for reason in detail["failure_reasons"]
                    if reason != f"{condition}:encoded_outside_support"
                ]
                detail["qc_pass"] = not detail["failure_reasons"]
                prior[f"qc_pass_{condition}"] = detail["qc_pass"]
            prior["failure_reasons"] = sorted({
                reason
                for condition in CONDITIONS
                for reason in prior["condition_details"][condition]["failure_reasons"]
            })
            results.append(prior)
        else:
            print(f"[{index}/{len(manifest)}] process {sample_id}", flush=True)
            results.append(process_one(
                row, v1_rows[sample_id], args.output_root, models, args.device, args.force
            ))
        checkpoint(args.result, list({**existing, **{
            item["sample_id"]: item for item in results
        }}.values()))
    final = read_jsonl(args.result)
    reports(final, args.summary, args.listen, args.backgrounds)
    print(f"Results: {args.result}")
    print(f"Summary: {args.summary}")
    print(f"Listening: {args.listen}")
    print(f"Background review: {args.backgrounds}")


if __name__ == "__main__":
    main()
