#!/usr/bin/env python3
"""Fail-closed full-denominator option-role audit for all existing CMTB factorials."""

from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/svhalluc")
OUT = ROOT / "outputs"
ROLE_MAP = OUT / "nor_annotation_validated_v1.jsonl"
Q25 = OUT / "qwen25_ter_replication_v1.jsonl"
Q25_AUDIO = str(OUT / "v45_audio/qwen25_ter_av_auto_shard*.jsonl")
Q3 = OUT / "qwen3_ter_large_v1.jsonl"
CHRONUS = str(OUT / "chronusomni_cmtb291_factorial_v1_shard*.jsonl")
JSON_OUT = OUT / "cmtb291_option_role_factorial_allmodels_v1.json"
REPORT = OUT / "cmtb291_option_role_factorial_allmodels_v1.md"
SEED = 20260814
BOOT = 20_000
LETTERS = {"A", "B", "C"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(rows: list[dict], condition: str) -> dict[str, dict]:
    chosen = [
        row for row in rows
        if row.get("condition") == condition and row.get("task", "CMTB") == "CMTB"
    ]
    result = {row["sample_id"]: row for row in chosen}
    if len(chosen) != len(result) or len(result) != 291:
        raise RuntimeError(f"{condition}: expected 291 unique rows, got {len(chosen)}/{len(result)}")
    return result


def classify(row: dict, role: dict | None) -> str:
    prediction = row.get("prediction")
    answer = row.get("answer", row.get("official_answer"))
    if prediction == answer:
        return "CURRENT"
    if prediction not in LETTERS:
        return "PARSE_FAIL"
    if role is None:
        return "UNMAPPED"
    if prediction == role["offtime_narrated_option"]:
        return "NARRATED"
    if prediction == role["unrelated_option"]:
        return "UNRELATED"
    raise RuntimeError(f"role partition failure: {row['sample_id']}")


def boot_ci(values: np.ndarray, seed_offset: int) -> list[float]:
    rng = np.random.default_rng(SEED + seed_offset)
    draws = values[rng.integers(0, len(values), size=(BOOT, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def model_result(label: str, cells: dict[str, dict[str, dict]], ids: list[str], roles: dict[str, dict], seed_offset: int) -> dict:
    assigned = {
        condition: [classify(cells[condition][sample_id], roles.get(sample_id)) for sample_id in ids]
        for condition in cells
    }
    counts = {condition: dict(Counter(assigned[condition])) for condition in cells}
    off, on = assigned["correct_pointer_audio_off"], assigned["correct_pointer_audio_on"]
    narrated_off = np.asarray([int(value == "NARRATED") for value in off])
    narrated_on = np.asarray([int(value == "NARRATED") for value in on])
    unrelated_off = np.asarray([int(value == "UNRELATED") for value in off])
    unrelated_on = np.asarray([int(value == "UNRELATED") for value in on])
    selectivity = (narrated_on - narrated_off) - (unrelated_on - unrelated_off)
    transition = Counter(zip(off, on))
    return {
        "full_denominator_n": len(ids),
        "role_mapped_n": len(roles),
        "roles_by_condition": counts,
        "correct_pointer_audio_transition": {
            "current_to_narrated": transition[("CURRENT", "NARRATED")],
            "narrated_to_current": transition[("NARRATED", "CURRENT")],
            "current_to_unrelated": transition[("CURRENT", "UNRELATED")],
            "unrelated_to_current": transition[("UNRELATED", "CURRENT")],
            "all_pairs": {f"{left}->{right}": count for (left, right), count in sorted(transition.items())},
        },
        "fail_closed_full_denominator_selectivity": {
            "estimate": float(selectivity.mean()),
            "paired_bootstrap_95ci": boot_ci(selectivity, seed_offset),
            "definition": "[I(d_n|audio on)-I(d_n|audio off)]-[I(d_u|audio on)-I(d_u|audio off)]; unmapped roles contribute zero to both indicators.",
            "interpretation": "A fail-closed lower-bound directional diagnostic, not a prevalence estimate because only 184/291 roles have human labels.",
        },
    }


def main() -> None:
    role_rows = read(ROLE_MAP)
    role_map_all = {row["sample_id"]: row for row in role_rows}
    if len(role_map_all) != len(role_rows) or len(role_map_all) != 184:
        raise RuntimeError("expected exact validated 184-item role map")
    roles = {
        sample_id: row for sample_id, row in role_map_all.items()
        if row.get("offtime_narrated_option") in LETTERS
        and row.get("unrelated_option") in LETTERS
        and row["offtime_narrated_option"] != row["unrelated_option"]
    }
    q25_rows = read(Q25)
    q25_audio_paths = [Path(path) for path in sorted(glob.glob(Q25_AUDIO))]
    q25_audio_rows = [row for path in q25_audio_paths for row in read(path)]
    q3_rows = read(Q3)
    chronus_paths = [Path(path) for path in sorted(glob.glob(CHRONUS))]
    chronus_rows = [row for path in chronus_paths for row in read(path)]
    raw = {
        "Qwen2.5-Omni": {
            "no_pointer_audio_off": select(q25_rows, "V"),
            "no_pointer_audio_on": select(q25_rows, "O_raw"),
            "correct_pointer_audio_off": select(q25_rows, "TER_V_auto"),
            "correct_pointer_audio_on": select(q25_audio_rows, "TER_AV_auto"),
        },
        "Qwen3-Omni": {
            "no_pointer_audio_off": select(q3_rows, "V"),
            "no_pointer_audio_on": select(q3_rows, "O_raw"),
            "correct_pointer_audio_off": select(q3_rows, "TER_V_auto"),
            "correct_pointer_audio_on": select(q3_rows, "TER_AV_auto"),
        },
        "ChronusOmni": {
            "no_pointer_audio_off": select(chronus_rows, "no_pointer_audio_off"),
            "no_pointer_audio_on": select(chronus_rows, "no_pointer_audio_on"),
            "correct_pointer_audio_off": select(chronus_rows, "correct_pointer_audio_off"),
            "correct_pointer_audio_on": select(chronus_rows, "correct_pointer_audio_on"),
        },
    }
    ids = sorted(next(iter(raw.values()))["correct_pointer_audio_off"])
    if any(set(cell) != set(ids) for model in raw.values() for cell in model.values()):
        raise RuntimeError("factorial item IDs differ")
    models = {
        label: model_result(label, cells, ids, roles, offset)
        for offset, (label, cells) in enumerate(raw.items())
    }
    inputs = [ROLE_MAP, Q25, Q3, *q25_audio_paths, *chronus_paths]
    result = {
        "status": "PASS_EXPLORATORY_ALL_MODEL_OPTION_ROLE_AUDIT",
        "scope": "Post-hoc, role-map-limited directional audit. It does not make the interaction family-general or prove lexical causality.",
        "human_role_map": {"all_labeled_n": len(role_map_all), "determinate_three_role_n": len(roles)},
        "models": models,
        "bootstrap": {"repetitions": BOOT, "seed": SEED, "unit": "CMTB video"},
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in inputs},
        "analysis_sha256": sha(Path(__file__)),
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# CMTB-291 option-role factorial audit across existing models", "", "Status: **exploratory post-hoc audit**", "", result["scope"], "", "| Model | $d_n$ off→on | $d_u$ off→on | Selectivity | 95% CI | $y\to d_n$ / $d_n\to y$ |", "|---|---:|---:|---:|---:|---:|"]
    for label, item in models.items():
        counts_off = item["roles_by_condition"]["correct_pointer_audio_off"]
        counts_on = item["roles_by_condition"]["correct_pointer_audio_on"]
        selectivity = item["fail_closed_full_denominator_selectivity"]
        transition = item["correct_pointer_audio_transition"]
        lines.append(
            f"| {label} | {counts_off.get('NARRATED', 0)}→{counts_on.get('NARRATED', 0)} | "
            f"{counts_off.get('UNRELATED', 0)}→{counts_on.get('UNRELATED', 0)} | "
            f"{100 * selectivity['estimate']:+.1f} pp | "
            f"[{100 * selectivity['paired_bootstrap_95ci'][0]:+.1f}, {100 * selectivity['paired_bootstrap_95ci'][1]:+.1f}] | "
            f"{transition['current_to_narrated']} / {transition['narrated_to_current']} |"
        )
    lines.extend(["", "All role counts use the full 291 denominator; absent human role annotations remain `UNMAPPED` rather than being imputed or dropped.", ""])
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
