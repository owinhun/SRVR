#!/usr/bin/env python3
"""Wait for both frozen native-generation runs and analyze only after completion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
Q25 = ROOT / "outputs/cmtb_native_generation_dev48_qwen25_v1.jsonl"
Q3 = ROOT / "outputs/cmtb_native_generation_dev48_qwen3_v1.jsonl"
STATUS = ROOT / "outputs/cmtb_native_generation_dev48_supervisor_v1.json"
ANALYSIS = ROOT / "outputs/cmtb_native_generation_dev48_analysis_v1.json"
EXPECTED = 96


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def runner_alive(family: str) -> bool:
    needle = f"--model-family\x00{family}".encode()
    own = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        try:
            command = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"run_cmtb_native_generation_dev48_v1.py" in command and needle in command:
            return True
    return False


def write_status(state: str, **extra: object) -> None:
    value = {
        "pid": os.getpid(),
        "state": state,
        "qwen25_rows": line_count(Q25),
        "qwen3_rows": line_count(Q3),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main() -> None:
    while True:
        q25, q3 = line_count(Q25), line_count(Q3)
        if q25 == EXPECTED and q3 == EXPECTED:
            break
        if q25 > EXPECTED or q3 > EXPECTED:
            write_status("FAILED_DUPLICATE_OR_EXCESS_ROWS")
            raise SystemExit(2)
        failures = []
        if q25 < EXPECTED and not runner_alive("qwen25"):
            failures.append("qwen25_runner_not_alive")
        if q3 < EXPECTED and not runner_alive("qwen3"):
            failures.append("qwen3_runner_not_alive")
        if failures:
            write_status("FAILED_INCOMPLETE_RUN", failures=failures)
            raise SystemExit(3)
        write_status("RUNNING_FROZEN_NATIVE_GENERATION")
        time.sleep(30)
    write_status("ANALYZING_COMPLETE_OUTPUTS")
    subprocess.run(
        [sys.executable, "src/analysis/analyze_cmtb_native_generation_dev48_v1.py"],
        cwd=ROOT,
        check=True,
    )
    report = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    write_status(report["status"], analysis=str(ANALYSIS))


if __name__ == "__main__":
    main()

