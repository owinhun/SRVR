#!/usr/bin/env python3
"""Prepare answer-blind selector and VAD inputs for locked CMTB-291."""

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
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--vad-output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.source) if line.strip()]
    if len(rows) != 291 or len({row["sample_id"] for row in rows}) != 291:
        raise ValueError("source is not exactly 291 unique CMTB samples")
    source_hash = sha256(args.source)
    manifest_path = Path(args.manifest_output)
    vad_path = Path(args.vad_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    vad_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w") as manifest, vad_path.open("w") as vad:
        for row in rows:
            path = Path(row["video_path"])
            if not path.is_file() or not row.get("qc_pass"):
                raise ValueError(f"invalid locked row {row['sample_id']}")
            if sha256(path) != row["source_file_sha256"]:
                raise ValueError(f"media hash mismatch {row['sample_id']}")
            manifest.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "official_index": row["official_index"],
                        "video_id": row["video_id"],
                        "video_path": str(path.resolve()),
                        "video_sha256": row["source_file_sha256"],
                        "selector_prompt": row["question"],
                        "answer_evaluation_only": row["answer"],
                        "selector_answer_blind": True,
                        "downstream_model": "OpenGVLab/InternVL2_5-8B",
                        "downstream_revision": "e9e4c0dc1db56bfab10458671519b7fa3dd29463",
                        "downstream_audio": "off",
                    }
                )
                + "\n"
            )
            vad.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "video_id": row["video_id"],
                        "automatic_vad_start_sec": row["speech_start_sec"],
                        "automatic_vad_end_sec": row["speech_end_sec"],
                        "vad_source": "frozen_automatic_silero_vad_interval",
                        "source_sha256": source_hash,
                        "answer_used": False,
                    }
                )
                + "\n"
            )

    audit = {
        "status": "PASS_PREPARED_NOT_EXECUTED",
        "population": 291,
        "unique_sample_ids": 291,
        "source": args.source,
        "source_sha256": source_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "vad": str(vad_path),
        "vad_sha256": sha256(vad_path),
        "selector_answer_blind": True,
        "full_execution_requires_dev_gate": True,
    }
    Path(args.audit_output).write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
