#!/usr/bin/env python3
"""Build deterministic duration-matched black-video media for CMTB-291."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MANIFEST_HASH = "dd4f9cd881310d3ca280219dbc0f789c7997d78728eded30d153054e5ad92416"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "outputs/cmtb291_black_video_media_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/cmtb291_black_video_manifest_v1.jsonl")
    args = parser.parse_args()
    if sha256(args.manifest) != MANIFEST_HASH:
        raise RuntimeError("frozen source manifest hash mismatch")
    rows = read_jsonl(args.manifest)
    if len(rows) != 291 or len({row["sample_id"] for row in rows}) != 291:
        raise RuntimeError("expected unique CMTB-291")
    args.media_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, row in enumerate(rows, 1):
        target = args.media_dir / f"{row['sample_id']}.mp4"
        requested = float(row["media_duration_sec"])
        if not target.is_file():
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", f"color=c=black:s=640x360:r=2:d={requested:.6f}",
                 "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                check=True,
            )
        actual = duration(target)
        if abs(actual - requested) > 0.55:
            raise RuntimeError(f"duration mismatch {row['sample_id']}: {actual} vs {requested}")
        records.append({
            "sample_id": row["sample_id"], "video_id": row["video_id"],
            "black_video_path": str(target), "black_video_sha256": sha256(target),
            "requested_duration_sec": requested, "actual_duration_sec": actual,
            "resolution": "640x360", "fps": 2.0, "audio_stream": False,
            "source_manifest_sha256": MANIFEST_HASH,
        })
        print(index, row["sample_id"], f"{actual:.3f}s", flush=True)
    args.output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    print(f"wrote {len(records)} rows to {args.output}; sha256={sha256(args.output)}", flush=True)


if __name__ == "__main__":
    main()
