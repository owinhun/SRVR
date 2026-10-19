#!/usr/bin/env python3
"""Build answer-blind CMTB-291 Silero threshold 0.3/0.7 cues and geometry audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import silero_vad
import torch


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
PROTOCOL = ROOT / "paper/cmtb291_vad_threshold_sensitivity_protocol_v1.md"
OUT = ROOT / "outputs/cmtb291_vad_threshold_sensitivity_v1.jsonl"
AUDIT = ROOT / "outputs/cmtb291_vad_threshold_sensitivity_geometry_v1.json"
RATE = 16000
THRESHOLDS = (0.3, 0.7)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decode(path: Path) -> np.ndarray:
    payload = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-",
        ],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return np.frombuffer(payload, dtype=np.float32).copy()


def iou(a: float, b: float, c: float, d: float) -> float:
    inter = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)
    return inter / union if union > 0 else 0.0


def main() -> None:
    if silero_vad.__version__ != "6.2.1":
        raise RuntimeError(f"Silero version drift: {silero_vad.__version__}")
    rows = read(MANIFEST)
    if len(rows) != len({r["sample_id"] for r in rows}) != 291:
        raise RuntimeError("manifest mismatch")
    if len(rows) != 291:
        raise RuntimeError("expected CMTB-291")
    model = silero_vad.load_silero_vad()
    model.eval()
    outputs = []
    for index, row in enumerate(rows, start=1):
        audio = torch.from_numpy(decode(Path(row["video_path"])))
        for threshold in THRESHOLDS:
            segments = silero_vad.get_speech_timestamps(
                audio, model, sampling_rate=RATE, threshold=threshold,
                min_speech_duration_ms=250, min_silence_duration_ms=100,
                speech_pad_ms=30, return_seconds=False,
            )
            if segments:
                start = round(segments[0]["start"] / RATE, 3)
                end = round(segments[-1]["end"] / RATE, 3)
                status = "ok"
            else:
                start = end = None
                status = "no_speech"
            outputs.append({
                "sample_id": row["sample_id"],
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "media_duration_sec": row["media_duration_sec"],
                "threshold": threshold,
                "vad_status": status,
                "speech_start_sec": start,
                "speech_end_sec": end,
                "segment_count": len(segments),
                "reference_0.5_start_sec_analysis_only": row["speech_start_sec"],
                "reference_0.5_end_sec_analysis_only": row["speech_end_sec"],
                "config": {
                    "package": "silero-vad", "package_version": "6.2.1",
                    "sampling_rate": RATE, "minimum_speech_duration_ms": 250,
                    "minimum_silence_duration_ms": 100, "speech_pad_ms": 30,
                    "interval_rule": "first_start_to_last_end",
                },
                "answer_or_prediction_fields_read": False,
                "manifest_sha256": sha(MANIFEST),
                "protocol_sha256": sha(PROTOCOL),
            })
        if index % 25 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)}", flush=True)
    OUT.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in outputs))

    by_threshold = {}
    for threshold in THRESHOLDS:
        subset = [r for r in outputs if r["threshold"] == threshold]
        ok = [r for r in subset if r["vad_status"] == "ok"]
        ious = [
            iou(
                r["speech_start_sec"], r["speech_end_sec"],
                r["reference_0.5_start_sec_analysis_only"],
                r["reference_0.5_end_sec_analysis_only"],
            )
            for r in ok
        ]
        coverage = [
            (r["speech_end_sec"] - r["speech_start_sec"]) / r["media_duration_sec"]
            for r in ok
        ]
        by_threshold[str(threshold)] = {
            "n": len(subset), "ok_n": len(ok), "no_speech_n": len(subset) - len(ok),
            "median_iou_to_0.5": float(np.median(ious)) if ious else None,
            "min_iou_to_0.5": float(np.min(ious)) if ious else None,
            "median_coverage": float(np.median(coverage)) if coverage else None,
        }
    audit = {
        "status": "PASS_READY_FOR_FROZEN_QWEN25_INFERENCE",
        "population_n": 291,
        "thresholds": by_threshold,
        "prediction_fields_read": False,
        "hashes": {
            "manifest": sha(MANIFEST), "protocol": sha(PROTOCOL),
            "cues": sha(OUT), "builder": sha(Path(__file__)),
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

