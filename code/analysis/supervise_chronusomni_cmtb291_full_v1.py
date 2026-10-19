#!/usr/bin/env python3
"""Launch the frozen four-GPU ChronusOmni full CMTB factorial after preflight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
PREFLIGHT_AUDIT = ROOT / "outputs/chronusomni_cmtb291_preflight_audit_v1.json"
ANALYSIS = ROOT / "outputs/chronusomni_cmtb291_factorial_analysis_v1.json"
TARGET_SITE = "/data4/svhalluc_envs/chronusomni_v1/site-packages"


def preflight_passed() -> bool:
    if not PREFLIGHT_AUDIT.is_file():
        return False
    try:
        return json.loads(PREFLIGHT_AUDIT.read_text()).get("status") == "PASS_MECHANICAL_PREFLIGHT"
    except Exception:
        return False


def gpu_free_memory() -> dict[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.splitlines()}


def wait_for_gates() -> None:
    while not preflight_passed():
        print("WAITING_FOR_CHRONUS_PREFLIGHT_PASS", flush=True)
        time.sleep(30)
    while True:
        free_memory = gpu_free_memory()
        # The measured preflight peak is below 43 GiB. The user explicitly
        # permits sharing GPUs 0--3 when roughly 40 GiB remains; retain a
        # conservative 45-GiB launch floor and never overcommit memory.
        if all(free_memory.get(gpu, 0) >= 45000 for gpu in range(4)):
            return
        print(f"WAITING_FOR_45G_FREE_ON_GPUS_0_1_2_3 {free_memory}", flush=True)
        time.sleep(30)


def environment(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "PYTHONPATH": TARGET_SITE,
            "HF_HOME": "/data4/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/data4/hf_cache",
            "HF_XET_CACHE": "/data4/hf_xet_cache",
            "TRANSFORMERS_CACHE": "/data4/hf_cache",
            "TORCH_HOME": "/data4/torch_cache",
            "TRITON_CACHE_DIR": f"/data4/triton_cache/chronusomni_v1_gpu{gpu}",
            "TORCH_EXTENSIONS_DIR": f"/data4/torch_extensions/chronusomni_v1_gpu{gpu}",
            "XDG_CACHE_HOME": "/data4/cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def main() -> None:
    if ANALYSIS.is_file():
        payload = json.loads(ANALYSIS.read_text())
        if payload.get("status") == "PASS_COMPLETE_FULL_POPULATION_ANALYSIS":
            print("CHRONUS_FULL_ALREADY_COMPLETE", flush=True)
            return
    wait_for_gates()
    processes = []
    for gpu in range(4):
        output = ROOT / f"outputs/chronusomni_cmtb291_factorial_v1_shard{gpu}.jsonl"
        log_path = ROOT / f"logs/chronusomni_cmtb291_factorial_v1_shard{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "src/qwen/run_chronusomni_cmtb291_factorial_v1.py",
            "--output", str(output),
            "--mode", "full",
            "--shard-id", str(gpu),
            "--shard-count", "4",
            "--physical-gpu", str(gpu),
        ]
        process = subprocess.Popen(command, cwd=ROOT, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
        processes.append((gpu, process, log))
        print(f"LAUNCHED_GPU_{gpu} PID_{process.pid}", flush=True)
    failures = []
    for gpu, process, log in processes:
        return_code = process.wait()
        log.close()
        if return_code != 0:
            failures.append({"gpu": gpu, "return_code": return_code})
    if failures:
        print(json.dumps({"status": "FULL_INFERENCE_FAILURE", "failures": failures}), flush=True)
        raise SystemExit(2)
    audited = subprocess.run(
        [sys.executable, "src/analysis/audit_chronusomni_cmtb291_full_v1.py"], cwd=ROOT
    )
    if audited.returncode != 0:
        raise SystemExit(audited.returncode)
    completed = subprocess.run(
        [sys.executable, "src/analysis/analyze_chronusomni_cmtb291_factorial_v1.py"], cwd=ROOT
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    print("PASS_CHRONUSOMNI_FULL_ANALYSIS", flush=True)


if __name__ == "__main__":
    main()
