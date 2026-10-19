#!/usr/bin/env python3
"""Audit the maximum CMTB headroom of a per-sample audio gate.

This is a kill test, not a deployable selector: the oracle uses correctness only
to quantify the ceiling available above fixed audio-off SRVR.  A frozen human
role map is used solely for a directional diagnostic on its annotated subset.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/svhalluc")
OUT = ROOT / "outputs"
Q25 = OUT / "qwen25_ter_replication_v1.jsonl"
Q3 = OUT / "qwen3_ter_large_v1.jsonl"
Q25_AUDIO_GLOB = str(OUT / "v45_audio/qwen25_ter_av_auto_shard*.jsonl")
HUMAN = OUT / "nor_annotations_annotator_2.jsonl"
JSON_OUT = OUT / "cmtb_selective_audio_router_headroom_v1.json"
MD_OUT = OUT / "cmtb_selective_audio_router_headroom_v1.md"


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select(rows: list[dict[str, Any]], condition: str) -> dict[str, dict[str, Any]]:
    chosen = [
        row for row in rows
        if row.get("task") == "CMTB" and row.get("condition") == condition
    ]
    result = {row["sample_id"]: row for row in chosen}
    if len(chosen) != 291 or len(result) != 291:
        raise RuntimeError(
            f"{condition}: expected 291 unique rows, got {len(chosen)}/{len(result)}"
        )
    return result


def model_audit(
    audio_off: dict[str, dict[str, Any]],
    audio_on: dict[str, dict[str, Any]],
    human: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ids = sorted(audio_off)
    if set(ids) != set(audio_on):
        raise RuntimeError("audio-off/audio-on sample sets differ")

    off_correct = {sid for sid in ids if bool(audio_off[sid]["correct"])}
    on_correct = {sid for sid in ids if bool(audio_on[sid]["correct"])}
    union = off_correct | on_correct
    gains = on_correct - off_correct
    harms = off_correct - on_correct

    determinate_ids = sorted(
        sid for sid in set(ids) & set(human)
        if human[sid].get("offtime_narrated_option") in {"A", "B", "C"}
    )
    directional = {
        "n": len(determinate_ids),
        "audio_off_narrated": 0,
        "audio_on_narrated": 0,
        "audio_induced_narrated_flip": 0,
        "reverse_narrated_flip": 0,
        "gain_n": 0,
        "harm_n": 0,
        "harm_to_narrated_n": 0,
        "gain_from_narrated_n": 0,
    }
    for sid in determinate_ids:
        narrated = human[sid]["offtime_narrated_option"]
        pred_off = audio_off[sid].get("prediction")
        pred_on = audio_on[sid].get("prediction")
        off_is_narrated = pred_off == narrated
        on_is_narrated = pred_on == narrated
        directional["audio_off_narrated"] += int(off_is_narrated)
        directional["audio_on_narrated"] += int(on_is_narrated)
        directional["audio_induced_narrated_flip"] += int(
            (not off_is_narrated) and on_is_narrated
        )
        directional["reverse_narrated_flip"] += int(
            off_is_narrated and (not on_is_narrated)
        )
        is_gain = sid in gains
        is_harm = sid in harms
        directional["gain_n"] += int(is_gain)
        directional["harm_n"] += int(is_harm)
        directional["harm_to_narrated_n"] += int(is_harm and on_is_narrated)
        directional["gain_from_narrated_n"] += int(is_gain and off_is_narrated)

    n = len(ids)
    return {
        "n": n,
        "audio_off_correct": len(off_correct),
        "audio_on_correct": len(on_correct),
        "oracle_union_correct": len(union),
        "audio_off_accuracy": len(off_correct) / n,
        "audio_on_accuracy": len(on_correct) / n,
        "oracle_union_accuracy": len(union) / n,
        "oracle_gain_over_audio_off_pp": 100 * (len(union) - len(off_correct)) / n,
        "wrong_to_correct_available": len(gains),
        "correct_to_wrong_to_avoid": len(harms),
        "unchanged_correct": len(off_correct & on_correct),
        "unchanged_wrong": n - len(union),
        "switch_positive_prevalence": len(gains) / n,
        "directional_human_subset": directional,
    }


def main() -> None:
    q25_rows = read(Q25)
    q3_rows = read(Q3)
    q25_audio_paths = [Path(path) for path in sorted(glob.glob(Q25_AUDIO_GLOB))]
    if len(q25_audio_paths) != 4:
        raise RuntimeError("expected four Qwen2.5 audio-on shards")
    q25_audio_rows = [row for path in q25_audio_paths for row in read(path)]
    human_rows = read(HUMAN)
    human = {row["sample_id"]: row for row in human_rows}
    if len(human_rows) != 100 or len(human) != 100:
        raise RuntimeError("expected 100 unique human annotations")

    models = {
        "qwen25": model_audit(
            select(q25_rows, "TER_V_auto"),
            select(q25_audio_rows, "TER_AV_auto"),
            human,
        ),
        "qwen3": model_audit(
            select(q3_rows, "TER_V_auto"),
            select(q3_rows, "TER_AV_auto"),
            human,
        ),
    }

    result = {
        "status": "STOP_ACCURACY_ROUTER_LOW_HEADROOM",
        "design": (
            "Oracle per-sample selector between independently frozen correct-pointer "
            "audio-off and audio-on predictions; correctness is used only for ceiling analysis."
        ),
        "models": models,
        "decision": {
            "promote_accuracy_router": False,
            "reason": (
                "Even a label-oracle selector can improve fixed audio-off SRVR by less "
                "than 4 percentage points in either Qwen generation; the deployable "
                "selector would need to find rare gains while avoiding substantially "
                "more harmful switches."
            ),
            "human_scope": (
                "Single prediction-blind annotator subset; directional diagnostic only, "
                "not independent-human agreement or population prevalence."
            ),
        },
    }
    JSON_OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# CMTB selective-audio router headroom kill test",
        "",
        "Status: **STOP_ACCURACY_ROUTER_LOW_HEADROOM**",
        "",
        "The oracle chooses between already-frozen correct-pointer audio-off and audio-on "
        "predictions using the gold correctness label only to measure the maximum possible "
        "accuracy ceiling. It is not a deployable method result.",
        "",
        "| Model | Audio off | Audio on | Oracle union | Max gain | W→C available | C→W to avoid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in models.items():
        lines.append(
            f"| {name} | {row['audio_off_correct']}/291 "
            f"({100*row['audio_off_accuracy']:.1f}%) | "
            f"{row['audio_on_correct']}/291 ({100*row['audio_on_accuracy']:.1f}%) | "
            f"{row['oracle_union_correct']}/291 ({100*row['oracle_union_accuracy']:.1f}%) | "
            f"+{row['oracle_gain_over_audio_off_pp']:.1f}%p | "
            f"{row['wrong_to_correct_available']} | {row['correct_to_wrong_to_avoid']} |"
        )
    lines.extend([
        "",
        "## Frozen human-role directional subset",
        "",
        "This subset comes from one prediction-blind human annotator and is used only to "
        "inspect direction, not to claim inter-annotator agreement or prevalence.",
        "",
        "| Model | Determinate N | Narrated off→on | Audio-induced narrated | Reverse | Harms to narrated |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, row in models.items():
        d = row["directional_human_subset"]
        lines.append(
            f"| {name} | {d['n']} | {d['audio_off_narrated']}→{d['audio_on_narrated']} | "
            f"{d['audio_induced_narrated_flip']} | {d['reverse_narrated_flip']} | "
            f"{d['harm_to_narrated_n']}/{d['harm_n']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "Do not train a per-sample audio gate merely to raise CMTB accuracy. The oracle "
        "headroom is below 4%p, while useful switches are rare and harmful switches are "
        "much more common. This branch remains a mechanism diagnostic, not a promoted "
        "algorithm contribution.",
        "",
    ])
    MD_OUT.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
