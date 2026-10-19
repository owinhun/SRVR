#!/usr/bin/env python3
"""Analyze the frozen CMTB-291 prompt-component Audio factorial."""

from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path("/workspace/svhalluc")
OUT = ROOT / "outputs"
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
ANNOTATION = OUT / "nor_annotation_validated_v1.jsonl"
FREEZE = OUT / "cmtb291_prompt_component_factorial_freeze_v1.json"
Q3_EXISTING = OUT / "qwen3_ter_large_v1.jsonl"
Q25_EXISTING = OUT / "qwen25_ter_replication_v1.jsonl"
Q25_FULL_ON = str(OUT / "v45_audio/qwen25_ter_av_auto_shard*.jsonl")
Q3_NEW = OUT / "prompt_ablation/qwen3_prompt_components_v1.jsonl"
Q25_NEW = str(OUT / "prompt_ablation/qwen25_prompt_components_v1_shard*.jsonl")
JSON_OUT = OUT / "cmtb291_prompt_component_factorial_analysis_v1.json"
REPORT_OUT = OUT / "cmtb291_prompt_component_factorial_report_kr_v1.md"
REPS = 20_000
SEED = 20260814
LETTERS = {"A", "B", "C"}
PROMPTS = (
    "no_reference", "timestamp_only", "relevant_timestamp_only",
    "relevant_timestamp_visual_focus", "full_srvr",
)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(rows: list[dict], condition: str) -> dict[str, dict]:
    chosen = [r for r in rows if r.get("task") == "CMTB" and r.get("condition") == condition]
    result = {r["sample_id"]: r for r in chosen}
    if len(chosen) != 291 or len(result) != 291:
        raise RuntimeError(f"{condition}: expected 291 unique rows, got {len(chosen)}/{len(result)}")
    return result


def load_model(model: str) -> tuple[dict[str, dict[str, dict]], list[Path]]:
    if model == "qwen3":
        existing_paths = [Q3_EXISTING]
        existing = read(Q3_EXISTING)
        new_paths = [Q3_NEW]
        new = read(Q3_NEW)
    else:
        full_on_paths = [Path(p) for p in sorted(glob.glob(Q25_FULL_ON))]
        new_paths = [Path(p) for p in sorted(glob.glob(Q25_NEW))]
        if len(full_on_paths) != 4 or len(new_paths) != 2:
            raise RuntimeError("Qwen2.5 requires four existing full-on and two new prompt shards")
        existing_paths = [Q25_EXISTING, *full_on_paths]
        existing = read(Q25_EXISTING) + [r for p in full_on_paths for r in read(p)]
        new = [r for p in new_paths for r in read(p)]
    conditions = {
        "no_reference": ("V", "O_raw"),
        "timestamp_only": ("timestamp_only_audio_off", "timestamp_only_audio_on"),
        "relevant_timestamp_only": ("relevant_timestamp_only_audio_off", "relevant_timestamp_only_audio_on"),
        "relevant_timestamp_visual_focus": ("relevant_timestamp_visual_focus_audio_off", "relevant_timestamp_visual_focus_audio_on"),
        "full_srvr": ("TER_V_auto", "TER_AV_auto"),
    }
    cells = {}
    for prompt_name, (off_name, on_name) in conditions.items():
        source = existing if prompt_name in {"no_reference", "full_srvr"} else new
        cells[prompt_name] = {"off": unique(source, off_name), "on": unique(source, on_name)}
    return cells, [*existing_paths, *new_paths]


def ci(values: np.ndarray) -> list[float]:
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def role(row: dict, annotation: dict | None) -> str:
    pred, answer = row.get("prediction"), row["answer"]
    if pred == answer:
        return "CURRENT"
    if pred not in LETTERS:
        return "PARSE_FAIL"
    if annotation is None:
        return "UNMAPPED"
    narrated, unrelated = annotation["offtime_narrated_option"], annotation["unrelated_option"]
    if narrated in LETTERS and pred == narrated:
        return "NARRATED"
    if unrelated in LETTERS and pred == unrelated:
        return "UNRELATED"
    if narrated == "NONE":
        return "NO_NARRATED_ROLE"
    if narrated == "AMBIGUOUS":
        return "AMBIGUOUS"
    return "UNMAPPED"


def gee_interaction(ids: list[str], no_ref: dict, prompt: dict) -> dict:
    rows = []
    for sid in ids:
        for reference, group in ((0, no_ref), (1, prompt)):
            for audio, name in ((0, "off"), (1, "on")):
                rows.append({"item": sid, "correct": int(group[name][sid]["correct"]), "audio": audio, "reference": reference})
    frame = pd.DataFrame(rows)
    fit = smf.gee(
        "correct ~ audio * reference", groups="item", data=frame,
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    term = "audio:reference"
    return {
        "log_odds_coefficient": float(fit.params[term]),
        "odds_ratio": float(np.exp(fit.params[term])),
        "coefficient_95ci": [float(x) for x in fit.conf_int().loc[term]],
        "odds_ratio_95ci": [float(x) for x in np.exp(fit.conf_int().loc[term])],
        "p": float(fit.pvalues[term]),
    }


def analyze_model(model: str, cells: dict, ids: list[str], annotations: dict[str, dict]) -> dict:
    correct = {
        p: {a: np.asarray([int(cells[p][a][sid]["correct"]) for sid in ids]) for a in ("off", "on")}
        for p in PROMPTS
    }
    accuracy = {p: {a: float(correct[p][a].mean()) for a in ("off", "on")} for p in PROMPTS}
    audio_effect = {p: correct[p]["on"] - correct[p]["off"] for p in PROMPTS}
    rng = np.random.default_rng(SEED + (0 if model == "qwen3" else 1))
    draw = rng.integers(0, len(ids), size=(REPS, len(ids)))
    interactions = {}
    for p in PROMPTS[1:]:
        item = audio_effect[p] - audio_effect["no_reference"]
        interactions[p] = {
            "estimate": float(item.mean()),
            "paired_bootstrap_95ci": ci(item[draw].mean(axis=1)),
            "logistic_gee": gee_interaction(ids, cells["no_reference"], cells[p]),
        }
    component_changes = {}
    for left, right in zip(PROMPTS[1:-1], PROMPTS[2:]):
        item = audio_effect[right] - audio_effect[left]
        component_changes[f"{left}_to_{right}"] = {
            "audio_effect_change": float(item.mean()),
            "paired_bootstrap_95ci": ci(item[draw].mean(axis=1)),
        }

    roles = {}
    for p in PROMPTS:
        off = [role(cells[p]["off"][sid], annotations.get(sid)) for sid in ids]
        on = [role(cells[p]["on"][sid], annotations.get(sid)) for sid in ids]
        transitions = Counter(zip(off, on))
        roles[p] = {
            "audio_off": dict(Counter(off)), "audio_on": dict(Counter(on)),
            "current_to_narrated": transitions[("CURRENT", "NARRATED")],
            "narrated_to_current": transitions[("NARRATED", "CURRENT")],
            "unmapped_either_n": sum(x == "UNMAPPED" for x in off) + sum(x == "UNMAPPED" for x in on),
        }
    parse_failures = {
        p: {a: sum(cells[p][a][sid].get("parse_status") == "parse_failure" for sid in ids) for a in ("off", "on")}
        for p in PROMPTS
    }
    return {
        "n": len(ids), "accuracy": accuracy,
        "audio_effect": {p: float(v.mean()) for p, v in audio_effect.items()},
        "interaction_vs_no_reference": interactions,
        "adjacent_component_changes": component_changes,
        "option_roles_fail_closed_full_denominator": roles,
        "parse_failures_counted_wrong": parse_failures,
    }


def main() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_BEFORE_NEW_MODEL_PREDICTIONS":
        raise RuntimeError("Invalid freeze status")
    rows = [r for r in read(MANIFEST) if r["task"] == "CMTB"]
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)) or len(ids) != 291:
        raise RuntimeError("Manifest is not unique CMTB-291")
    annotations = {r["sample_id"]: r for r in read(ANNOTATION)}
    if len(annotations) != 184:
        raise RuntimeError("Frozen role map changed")
    models, input_paths = {}, [FREEZE, MANIFEST, ANNOTATION]
    for model in ("qwen25", "qwen3"):
        cells, paths = load_model(model)
        models[model] = analyze_model(model, cells, ids, annotations)
        input_paths.extend(paths)

    result = {
        "status": "PASS_COMPLETE_CMTB291_PROMPT_COMPONENT_FACTORIAL",
        "experiment": "cmtb291_prompt_component_factorial_v1",
        "scope": "Prospective prompt-component diagnostic; existing no-reference and full-SRVR cells reused exactly.",
        "models": models,
        "bootstrap": {"repetitions": REPS, "seed": SEED, "unit": "CMTB item"},
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(input_paths))},
        "analysis_sha256": sha(Path(__file__)),
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# CMTB-291 prompt-component factorial", "", f"Status: **{result['status']}**", ""]
    for model, r in models.items():
        lines.extend([f"## {model}", "", "| Prompt | Audio off | Audio on | Audio effect | Interaction vs no reference | 95% CI |", "|---|---:|---:|---:|---:|---:|"])
        for p in PROMPTS:
            interaction = r["interaction_vs_no_reference"].get(p)
            est = "--" if interaction is None else f"{interaction['estimate']:+.1%}"
            cint = "--" if interaction is None else str([round(100*x, 1) for x in interaction["paired_bootstrap_95ci"]])
            lines.append(f"| {p} | {r['accuracy'][p]['off']:.1%} | {r['accuracy'][p]['on']:.1%} | {r['audio_effect'][p]:+.1%} | {est} | {cint} |")
        lines.extend(["", "Narrated transitions (audio off -> on):"])
        for p in PROMPTS:
            rr = r["option_roles_fail_closed_full_denominator"][p]
            lines.append(f"- {p}: current→narrated {rr['current_to_narrated']}, narrated→current {rr['narrated_to_current']}, unmapped {rr['unmapped_either_n']}")
        lines.append("")
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
