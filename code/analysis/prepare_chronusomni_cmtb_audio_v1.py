#!/usr/bin/env python3
"""Prediction-blind, full-stream audio preparation for ChronusOmni CMTB."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
SOURCE = ROOT / "data/ter_large_eval_v1.jsonl"
DEST = Path("/data4/svhalluc_data/chronus_cmtb_audio_v1")
MANIFEST = ROOT / "outputs/chronusomni_cmtb_audio_manifest_v1.jsonl"
AUDIT = ROOT / "outputs/chronusomni_cmtb_audio_audit_v1.json"
PROTOCOL = ROOT / "paper/chronusomni_cmtb291_strong_temporal_baseline_protocol_v1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    output = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,sample_rate,channels,duration",
            "-of", "json", str(path),
        ],
        text=True,
    )
    return json.loads(output)


def duration(payload: dict) -> float:
    value = payload.get("format", {}).get("duration")
    if value is not None:
        return float(value)
    values = [float(stream["duration"]) for stream in payload.get("streams", []) if stream.get("duration")]
    if not values:
        raise ValueError("duration unavailable")
    return max(values)


def prepare(row: dict) -> dict:
    source = Path(row["video_path"])
    target = DEST / f"{row['sample_id']}.wav"
    result = {
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "video_path": str(source),
        "audio_path": str(target),
        "status": "pending",
        "model_outputs_read": False,
    }
    try:
        source_probe = probe(source)
        audio_streams = [stream for stream in source_probe.get("streams", []) if stream.get("codec_type") == "audio"]
        if not audio_streams:
            raise RuntimeError("source has no audio stream")
        if not target.is_file():
            temporary = DEST / f".{row['sample_id']}.{os.getpid()}.tmp.wav"
            completed = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(source),
                    "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(temporary),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode != 0 or not temporary.is_file():
                raise RuntimeError(f"ffmpeg failed: {completed.stderr[-500:]}")
            temporary.rename(target)
        target_probe = probe(target)
        source_duration = duration(source_probe)
        audio_duration = duration(target_probe)
        difference = abs(source_duration - audio_duration)
        result.update(
            {
                "status": "valid" if difference <= 0.10 else "duration_mismatch",
                "video_sha256": sha256(source),
                "audio_sha256": sha256(target),
                "video_duration_sec": source_duration,
                "audio_duration_sec": audio_duration,
                "absolute_duration_difference_sec": difference,
                "audio_bytes": target.stat().st_size,
                "source_audio_stream": audio_streams[0],
                "derived_audio_stream": next(
                    stream for stream in target_probe["streams"] if stream.get("codec_type") == "audio"
                ),
            }
        )
    except Exception as error:  # Mechanical failures remain explicit rows.
        result.update({"status": "failure", "failure": f"{type(error).__name__}: {error}"})
    return result


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["task"] == "CMTB"]
    if len(rows) != 291 or len({row["sample_id"] for row in rows}) != 291:
        raise RuntimeError("unexpected CMTB population")
    DEST.mkdir(parents=True, exist_ok=True)
    results_by_id = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(prepare, row): row["sample_id"] for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results_by_id[result["sample_id"]] = result
            print(result["sample_id"], result["status"], flush=True)
    results = [results_by_id[row["sample_id"]] for row in rows]
    MANIFEST.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    audit = {
        "status": "PASS_ALL_AUDIO_VALID" if counts == {"valid": 291} else "COMPLETE_WITH_MEDIA_FAILURES",
        "population_n": len(results),
        "status_counts": counts,
        "max_duration_difference_sec": max(
            (row.get("absolute_duration_difference_sec", 0.0) for row in results), default=0.0
        ),
        "model_outputs_read": False,
        "threshold_retuned": False,
        "hashes": {
            "source_manifest": sha256(SOURCE),
            "protocol": sha256(PROTOCOL),
            "script": sha256(Path(__file__)),
            "audio_manifest": sha256(MANIFEST),
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
