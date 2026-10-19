#!/usr/bin/env python3
"""Run Nemotron preflight and audit runtime cache writes outside /data4."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
RUNNER = ROOT / "src/nemotron/run_nemotron3_cmtb291_factorial_v1.py"
OUTPUT = ROOT / "outputs/nemotron3_cmtb291_preflight_v8.jsonl"
LOG = ROOT / "outputs/nemotron3_cmtb291_preflight_runtime_v8.log"
AUDIT = ROOT / "outputs/nemotron3_cmtb291_preflight_cache_write_audit_v8.json"

MONITORED_ROOTS = (
    Path("/root/.cache/huggingface"),
    Path("/root/.cache/torch"),
    Path("/root/.cache/triton"),
    Path("/root/.cache/pip"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot() -> dict[str, tuple[int, int]]:
    values: dict[str, tuple[int, int]] = {}
    for root in MONITORED_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                stat = path.stat()
                values[str(path)] = (stat.st_size, stat.st_mtime_ns)
    for path in ROOT.rglob("__pycache__/*"):
        if path.is_file():
            stat = path.stat()
            values[str(path)] = (stat.st_size, stat.st_mtime_ns)
    return values


def main() -> None:
    before = snapshot()
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "2,3",
            "HF_HOME": "/data4/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/data4/hf_cache",
            "TORCH_HOME": "/data4/torch_cache",
            "TRITON_CACHE_DIR": "/data4/triton_cache/nemotron3_omni_hf_v56",
            "TORCH_EXTENSIONS_DIR": "/data4/torch_extensions/nemotron3_omni_hf_v56",
            "XDG_CACHE_HOME": "/data4/cache",
            "TMPDIR": "/data4/tmp",
            "PYTHONPYCACHEPREFIX": "/data4/pycache/nemotron3_omni_hf_v56",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "USE_HUB_KERNELS": "NO",
        }
    )
    command = [
        sys.executable,
        str(RUNNER),
        "--mode",
        "preflight",
        "--output",
        str(OUTPUT),
        "--physical-gpus",
        "2,3",
    ]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()

    after = snapshot()
    changed = sorted(
        path for path, state in after.items() if path not in before or before[path] != state
    )
    removed = sorted(path for path in before if path not in after)
    payload = {
        "status": (
            "PASS_NO_RUNTIME_CACHE_WRITE_OUTSIDE_DATA4"
            if not changed and not removed
            else "FAIL_RUNTIME_CACHE_WRITE_OUTSIDE_DATA4"
        ),
        "runner_returncode": returncode,
        "outside_data4_cache_writes": changed,
        "outside_data4_cache_removals": removed,
        "monitored_roots": [str(path) for path in MONITORED_ROOTS]
        + [str(ROOT / "**/__pycache__")],
        "environment_cache_roots": {
            key: environment[key]
            for key in (
                "HF_HOME",
                "HUGGINGFACE_HUB_CACHE",
                "TORCH_HOME",
                "TRITON_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR",
                "XDG_CACHE_HOME",
                "TMPDIR",
                "PYTHONPYCACHEPREFIX",
            )
        },
        "hashes": {
            "runner": sha256(RUNNER),
            "predictions": sha256(OUTPUT) if OUTPUT.is_file() else None,
            "runtime_log": sha256(LOG),
            "supervisor": sha256(Path(__file__)),
        },
    }
    AUDIT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if returncode != 0 or changed or removed:
        raise SystemExit(returncode or 2)


if __name__ == "__main__":
    main()
