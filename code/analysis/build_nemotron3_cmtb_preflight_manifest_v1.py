#!/usr/bin/env python3
"""Build the frozen, prediction-blind Nemotron CMTB-291 preflight manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
JOINED = ROOT / "outputs/videollama21_standard_cmtb291_manifest_v1.jsonl"
AUDIO = ROOT / "outputs/chronusomni_cmtb_audio_manifest_v1.jsonl"
PROTOCOL = ROOT / "paper/nemotron3_nano_omni_cmtb291_nonqwen_factorial_protocol_v1.md"
ADDENDUM = ROOT / "paper/nemotron3_direct_hf_engine_addendum_v1.md"
READINESS = ROOT / "outputs/nemotron3_engine_readiness_decision_v1.md"
OUTPUT = ROOT / "data/nemotron3_cmtb291_preflight12_v1.jsonl"
AUDIT = ROOT / "outputs/nemotron3_cmtb291_preflight_manifest_audit_v1.json"

NAMESPACE = "nemotron3-omni-cmtb-preflight-v1"
JOINED_SHA256 = "dd4f9cd881310d3ca280219dbc0f789c7997d78728eded30d153054e5ad92416"
AUDIO_SHA256 = "636abdafffc18f4f84d67a0032a9ecb444930d33c154e3a686c033239af88aa9"
PROTOCOL_SHA256 = "81a548a1906f2e6a43c226ef3ea94d370a132d6ba7e88c7f7ab96f6e5a046d05"
READINESS_SHA256 = "a8355a837bc830b0b7d078dc0db7937ee778150038cf646eb13b5c735901ddf5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_int(sample_id: str) -> int:
    return int(hashlib.sha256(f"{NAMESPACE}|{sample_id}".encode()).hexdigest(), 16)


def main() -> None:
    expected_hashes = {
        JOINED: JOINED_SHA256,
        AUDIO: AUDIO_SHA256,
        PROTOCOL: PROTOCOL_SHA256,
        READINESS: READINESS_SHA256,
    }
    observed_hashes = {str(path): sha256(path) for path in expected_hashes}
    mismatches = {
        str(path): {"expected": wanted, "observed": observed_hashes[str(path)]}
        for path, wanted in expected_hashes.items()
        if observed_hashes[str(path)] != wanted
    }
    if mismatches:
        raise RuntimeError(f"frozen input hash mismatch: {mismatches}")

    rows = read_jsonl(JOINED)
    audio_rows = {row["sample_id"]: row for row in read_jsonl(AUDIO)}
    if len(rows) != 291 or len({row["sample_id"] for row in rows}) != 291:
        raise RuntimeError("unexpected joined CMTB population")
    if len(audio_rows) != 291 or any(row.get("status") != "valid" for row in audio_rows.values()):
        raise RuntimeError("unexpected audio population")

    selected = sorted(rows, key=lambda row: stable_int(row["sample_id"]))[:12]
    materialized = []
    for rank, row in enumerate(selected, start=1):
        audio = audio_rows[row["sample_id"]]
        video = Path(row["video_path"])
        wav = Path(audio["audio_path"])
        if not video.is_file() or not wav.is_file():
            raise FileNotFoundError(f"missing media for {row['sample_id']}")
        if row["source_file_sha256"] != audio["video_sha256"]:
            raise RuntimeError(f"video hash disagreement for {row['sample_id']}")
        materialized.append(
            {
                **row,
                "preflight_namespace": NAMESPACE,
                "preflight_rank": rank,
                "selection_sha256": hashlib.sha256(
                    f"{NAMESPACE}|{row['sample_id']}".encode()
                ).hexdigest(),
                "audio_path": audio["audio_path"],
                "audio_sha256": audio["audio_sha256"],
                "audio_duration_sec": audio["audio_duration_sec"],
                "video_sha256": audio["video_sha256"],
                "video_duration_sec": audio["video_duration_sec"],
            }
        )

    OUTPUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized)
    )
    audit = {
        "status": "PASS_FROZEN_PREDICTION_BLIND_PREFLIGHT_MANIFEST",
        "model_outputs_read": False,
        "selection_namespace": NAMESPACE,
        "population_n": len(rows),
        "preflight_n": len(materialized),
        "sample_ids": [row["sample_id"] for row in materialized],
        "all_media_exist": True,
        "all_audio_valid": True,
        "answer_was_not_a_selection_field": True,
        "hashes": {
            "joined_manifest": sha256(JOINED),
            "audio_manifest": sha256(AUDIO),
            "protocol": sha256(PROTOCOL),
            "addendum": sha256(ADDENDUM),
            "readiness": sha256(READINESS),
            "builder": sha256(Path(__file__)),
            "preflight_manifest": sha256(OUTPUT),
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
