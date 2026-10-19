#!/usr/bin/env python3
"""Audit separated two-row preflights before full maximal-dose inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT=Path("/workspace/svhalluc")
POINTER=ROOT/"outputs/cmtb291_maximal_displacement_pointer_v1.jsonl"
FILES={
    "qwen3":ROOT/"outputs/preflight_qwen3_maximal_pointer_dose_v1.jsonl",
    "qwen25":ROOT/"outputs/preflight_qwen25_maximal_pointer_dose_v1.jsonl",
}
RUNNERS={
    "qwen3":ROOT/"src/qwen/run_qwen3_maximal_pointer_dose_v1.py",
    "qwen25":ROOT/"src/qwen/run_qwen25_maximal_pointer_dose_v1.py",
}
OUT=ROOT/"outputs/cmtb291_maximal_pointer_preflight_audit_v1.json"


def rows(path:Path)->list[dict]:return [json.loads(line) for line in path.read_text().splitlines() if line]
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    pointers={row["sample_id"]:row for row in rows(POINTER)};details={};checks={}
    for name,path in FILES.items():
        materialized=rows(path);keys={(row["sample_id"],row["condition"]) for row in materialized}
        errors=[]
        if len(materialized)!=8 or len(keys)!=8:errors.append("expected_8_unique")
        if len({row["sample_id"] for row in materialized})!=2:errors.append("expected_2_samples")
        if {row["pointer_alpha"] for row in materialized}!={.25,.5,.75,1.0}:errors.append("alpha_set")
        for row in materialized:
            pointer=pointers[row["sample_id"]];alpha=row["pointer_alpha"]
            expected=[(1-alpha)*pointer["speech_start_sec"]+alpha*pointer["maximal_start_sec"],(1-alpha)*pointer["speech_end_sec"]+alpha*pointer["maximal_end_sec"]]
            if max(abs(a-b) for a,b in zip(expected,row["used_interval_sec"]))>1e-8:errors.append(f"interval:{row['sample_id']}:{alpha}")
            if row["audio_handling"]["use_audio_in_video"] is not False or row["decoding"]["do_sample"] is not False:errors.append(f"contract:{row['sample_id']}:{alpha}")
            if row["code_sha256"]!=sha(RUNNERS[name]):errors.append(f"code_hash:{row['sample_id']}:{alpha}")
        details[name]={"rows":len(materialized),"unique_samples":len({row["sample_id"] for row in materialized}),"parse_failures":sum(row["parse_status"]!="parsed" for row in materialized),"file_sha256":sha(path),"errors":errors}
        checks[f"{name}_pass"]=not errors
    status="PASS_AUTHORIZE_FULL_INFERENCE" if all(checks.values()) else "FAIL_FULL_INFERENCE_UNAUTHORIZED"
    result={"status":status,"checks":checks,"details":details,"pointer_sha256":sha(POINTER),"boundary":"Preflight rows are interface checks and are excluded from full shard outputs and statistical analysis."}
    OUT.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
    if status.startswith("FAIL"):raise SystemExit(1)


if __name__=="__main__":main()
