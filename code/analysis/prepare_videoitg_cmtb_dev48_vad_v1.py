#!/usr/bin/env python3
"""Freeze answerless automatic-VAD intervals for the VideoITG Dev-48."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = [json.loads(line) for line in open(args.manifest) if line.strip()]
    wanted = {row["sample_id"] for row in manifest}
    intervals = {}
    with open(args.source) as handle:
        for line in handle:
            row = json.loads(line)
            sample_id = row.get("sample_id")
            interval = row.get("true_vad_interval_sec")
            if sample_id not in wanted or interval is None:
                continue
            value = (float(interval[0]), float(interval[1]))
            if sample_id in intervals and intervals[sample_id] != value:
                raise ValueError(f"conflicting VAD intervals for {sample_id}")
            intervals[sample_id] = value
    missing = wanted - intervals.keys()
    if missing:
        raise ValueError(f"missing {len(missing)} intervals: {sorted(missing)[:5]}")

    source_hash = sha256(args.source)
    manifest_hash = sha256(args.manifest)
    output = Path(args.output)
    with output.open("w") as sink:
        for row in manifest:
            start, end = intervals[row["sample_id"]]
            if not 0 <= start < end:
                raise ValueError(f"invalid interval for {row['sample_id']}: {(start, end)}")
            sink.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "video_id": row["video_id"],
                        "automatic_vad_start_sec": start,
                        "automatic_vad_end_sec": end,
                        "vad_source": "frozen_automatic_silero_vad_interval",
                        "source_sha256": source_hash,
                        "dev_manifest_sha256": manifest_hash,
                        "answer_used": False,
                    }
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "status": "PASS",
                "n": len(manifest),
                "output": str(output),
                "output_sha256": sha256(output),
                "answer_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
