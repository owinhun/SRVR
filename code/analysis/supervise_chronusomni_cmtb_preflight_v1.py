#!/usr/bin/env python3
"""Wait for the frozen Chronus checkpoint, run and audit the mechanical preflight."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MODEL = Path("/data4/svhalluc_models/ChronusOmni")
CHECKPOINT_AUDIT = ROOT / "outputs/chronusomni_checkpoint_audit_v1.json"
CHECKPOINT_LINK = ROOT / "external/Chronus_official_v1/checkpoints"
PREDICTIONS = ROOT / "outputs/chronusomni_cmtb291_preflight_v1.jsonl"
CACHE_AUDIT = ROOT / "outputs/chronusomni_cmtb291_preflight_cache_write_audit_v1.json"
FINAL_AUDIT = ROOT / "outputs/chronusomni_cmtb291_preflight_audit_v1.json"
LOG = ROOT / "logs/chronusomni_cmtb291_preflight_v1.log"
TARGET_SITE = "/data4/svhalluc_envs/chronusomni_v1/site-packages"
CACHE_ROOTS = (
    Path("/root/.cache/huggingface"),
    Path("/root/.cache/torch"),
    Path("/root/.triton"),
    Path("/workspace/.cache/huggingface"),
    Path("/workspace/.cache/torch"),
    Path("/workspace/svhalluc/models"),
)


def checkpoint_ready() -> bool:
    if not CHECKPOINT_AUDIT.is_file() or not MODEL.is_dir():
        return False
    try:
        return json.loads(CHECKPOINT_AUDIT.read_text()).get("status") == "PASS_FROZEN_CHECKPOINT_ON_DATA4"
    except Exception:
        return False


def wait_for_checkpoint() -> None:
    while not checkpoint_ready():
        print("WAITING_FOR_FROZEN_CHECKPOINT", flush=True)
        time.sleep(30)


def ensure_checkpoint_link() -> None:
    if os.path.lexists(CHECKPOINT_LINK):
        if not CHECKPOINT_LINK.is_symlink() or CHECKPOINT_LINK.resolve() != MODEL.resolve():
            raise RuntimeError(f"unexpected official checkpoint compatibility path: {CHECKPOINT_LINK}")
        return
    os.symlink(MODEL, CHECKPOINT_LINK, target_is_directory=True)


def recent_writes(start_ns: int) -> list[dict]:
    writes = []
    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for directory, _, files in os.walk(root):
            for name in files:
                path = Path(directory) / name
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if stat.st_mtime_ns >= start_ns:
                    writes.append({"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return sorted(writes, key=lambda row: row["path"])


def main() -> None:
    wait_for_checkpoint()
    ensure_checkpoint_link()
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": TARGET_SITE,
            "HF_HOME": "/data4/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/data4/hf_cache",
            "TRANSFORMERS_CACHE": "/data4/hf_cache",
            "TORCH_HOME": "/data4/torch_cache",
            "TRITON_CACHE_DIR": "/data4/triton_cache/chronusomni_v1",
            "TORCH_EXTENSIONS_DIR": "/data4/torch_extensions/chronusomni_v1",
            "XDG_CACHE_HOME": "/data4/cache",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    start_ns = time.time_ns()
    command = [
        sys.executable,
        "src/qwen/run_chronusomni_cmtb291_factorial_v1.py",
        "--output", str(PREDICTIONS),
        "--mode", "preflight",
        "--shard-id", "0",
        "--shard-count", "1",
        "--physical-gpu", "0",
    ]
    with LOG.open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    writes = recent_writes(start_ns)
    cache_payload = {
        "status": "PASS_NO_MODEL_CACHE_WRITE_OUTSIDE_DATA4" if not writes else "FAIL_OUTSIDE_DATA4_MODEL_CACHE_WRITE",
        "outside_data4_model_cache_writes": writes,
        "observed_cache_roots": [str(path) for path in CACHE_ROOTS],
        "run_start_ns": start_ns,
        "runner_exit_code": completed.returncode,
        "data4_environment": {key: environment[key] for key in (
            "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "TORCH_HOME",
            "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME",
        )},
        "checkpoint_link": str(CHECKPOINT_LINK),
        "checkpoint_link_target": str(CHECKPOINT_LINK.resolve()),
    }
    CACHE_AUDIT.write_text(json.dumps(cache_payload, indent=2, sort_keys=True) + "\n")
    audit = subprocess.run(
        [
            sys.executable,
            "src/analysis/audit_chronusomni_cmtb_preflight_v1.py",
            "--predictions", str(PREDICTIONS),
            "--cache-audit", str(CACHE_AUDIT),
            "--audit", str(FINAL_AUDIT),
        ],
        cwd=ROOT,
    )
    if completed.returncode != 0 or audit.returncode != 0:
        raise SystemExit(2)
    print("PASS_CHRONUSOMNI_PREFLIGHT", flush=True)


if __name__ == "__main__":
    main()
