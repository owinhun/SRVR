#!/usr/bin/env python3
"""Resume the frozen prompt factorial on GPUs 2/3 and analyze only when complete."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
RUNNER = ROOT / "src/qwen/run_cmtb291_prompt_component_factorial_v1.py"
ANALYZER = ROOT / "src/analysis/analyze_cmtb291_prompt_component_factorial_v1.py"
SHARD0 = ROOT / "outputs/prompt_ablation/qwen25_prompt_components_v1_shard0.jsonl"
SHARD1 = ROOT / "outputs/prompt_ablation/qwen25_prompt_components_v1_shard1.jsonl"
QWEN3 = ROOT / "outputs/prompt_ablation/qwen3_prompt_components_v1.jsonl"
FINAL = ROOT / "outputs/cmtb291_prompt_component_factorial_analysis_v1.json"
LOCK = ROOT / "outputs/prompt_ablation/orchestrator_gpu23_v1.lock"
STATUS = ROOT / "outputs/prompt_ablation/orchestrator_gpu23_v1_status.json"
CONDITIONS = {
    "timestamp_only_audio_off", "timestamp_only_audio_on",
    "relevant_timestamp_only_audio_off", "relevant_timestamp_only_audio_on",
    "relevant_timestamp_visual_focus_audio_off", "relevant_timestamp_visual_focus_audio_on",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rows(path: Path) -> tuple[int, int]:
    if not path.is_file():
        return 0, 0
    parsed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parsed.append(json.loads(line))
    keys = {(row["sample_id"], row["condition"]) for row in parsed}
    if any(row.get("condition") not in CONDITIONS for row in parsed):
        raise RuntimeError(f"Unexpected condition in {path}")
    if len(keys) != len(parsed):
        raise RuntimeError(f"Duplicate rows in {path}: lines={len(parsed)} unique={len(keys)}")
    return len(parsed), len(keys)


def running(fragment: str) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "run_cmtb291_prompt_component_factorial_v1.py" in cmdline and fragment in cmdline:
            return True
    return False


def launch(model: str, output: Path, physical_gpus: str, visible: str, shard: int | None = None) -> None:
    command = [
        sys.executable, str(RUNNER), "--model", model, "--output", str(output),
        "--physical-gpus", physical_gpus,
    ]
    if shard is not None:
        command.extend(["--shard-id", str(shard)])
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = visible
    # Operational allocator setting only: preserves every frozen model/input/
    # decoding parameter while allowing the Qwen3 visual encoder to reuse
    # fragmented blocks between its six fixed cells.  The OOM/restart event is
    # retained in the supervisor log and no sample or condition is skipped.
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = output.with_suffix(output.suffix + ".supervisor.log")
    log = log_path.open("a", encoding="utf-8")
    log.write(f"\n[{now()}] launch {' '.join(command)} CUDA_VISIBLE_DEVICES={visible}\n")
    log.flush()
    subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[{now()}] launched {model} {output.name}", flush=True)


def write_status(stage: str, counts: dict[str, int]) -> None:
    STATUS.write_text(json.dumps({
        "status": stage, "updated_at_utc": now(), "counts": counts,
        "result_exists": FINAL.is_file(),
    }, indent=2) + "\n")


def main() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another prompt-factorial orchestrator is already active") from exc
        while True:
            n0, _ = rows(SHARD0)
            n1, _ = rows(SHARD1)
            nq3, _ = rows(QWEN3)
            counts = {"qwen25_shard0": n0, "qwen25_shard1": n1, "qwen3": nq3}
            if n0 < 822 and not running(SHARD0.name):
                launch("qwen25", SHARD0, "2", "2", shard=0)
            if n1 < 924 and not running(SHARD1.name):
                launch("qwen25", SHARD1, "3", "3", shard=1)
            if n0 < 822 or n1 < 924:
                write_status("RUNNING_QWEN25", counts)
                time.sleep(30)
                continue
            if running(SHARD0.name) or running(SHARD1.name):
                write_status("WAITING_FOR_QWEN25_PROCESS_EXIT", counts)
                time.sleep(10)
                continue
            if nq3 < 1746:
                if not running(QWEN3.name):
                    launch("qwen3", QWEN3, "2,3", "2,3")
                write_status("RUNNING_QWEN3", counts)
                time.sleep(30)
                continue
            if running(QWEN3.name):
                write_status("WAITING_FOR_QWEN3_PROCESS_EXIT", counts)
                time.sleep(10)
                continue
            write_status("ANALYZING_FROZEN_COMPLETE_OUTPUT", counts)
            subprocess.run([sys.executable, str(ANALYZER)], cwd=ROOT, check=True)
            write_status("PASS_COMPLETE", counts)
            print(f"[{now()}] PASS_COMPLETE {FINAL}", flush=True)
            return


if __name__ == "__main__":
    main()
