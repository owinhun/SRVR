#!/usr/bin/env python3
"""Freeze a prediction-blind 60-item CMTB oracle-interval annotation package."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
OFFICIAL = ROOT / "data/SVHalluc/samples.json"
GEOMETRY = ROOT / "outputs/vad_envelope_geometry_cmtb291_v1.jsonl"
OUT_JSONL = ROOT / "outputs/cmtb60_oracle_interval_annotation_manifest_v1.jsonl"
OUT_CSV = ROOT / "outputs/cmtb60_oracle_interval_annotation_sheet_v1.csv"
OUT_FREEZE = ROOT / "outputs/cmtb60_oracle_interval_annotation_freeze_v1.json"
SEED = 20260814


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_choices(question: str) -> dict[str, str]:
    match = re.search(
        r"\(A\)\s*(.*?)\s*\(B\)\s*(.*?)\s*\(C\)\s*(.*)$",
        question.strip(),
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"Cannot parse choices: {question}")
    return {"A": match.group(1).strip(), "B": match.group(2).strip(), "C": match.group(3).strip()}


def main() -> None:
    official = json.loads(OFFICIAL.read_text(encoding="utf-8"))
    geometry = [json.loads(line) for line in GEOMETRY.read_text(encoding="utf-8").splitlines()]
    if len(geometry) != 291 or len({row["sample_id"] for row in geometry}) != 291:
        raise RuntimeError("Expected the locked 291-item CMTB geometry population")

    enriched = []
    for row in geometry:
        index = int(row["sample_id"].split("_")[-1])
        meta = official[index]
        if meta["task_type_abbr"] != "CMTB" or meta["video_path"] != row["video_id"]:
            raise RuntimeError(f"Official metadata mismatch for {row['sample_id']}")
        choices = parse_choices(meta["question"])
        answer = meta["answer"]
        enriched.append({
            **row,
            "official_index": index,
            "video_path": str(ROOT / "data/SVHalluc" / meta["video_path"]),
            "gold_option": answer,
            "current_action_text": choices[answer],
        })

    ranked = sorted(enriched, key=lambda x: (x["envelope_coverage"], x["sample_id"]))
    for rank, row in enumerate(ranked):
        row["coverage_quartile"] = min(4, (4 * rank) // len(ranked) + 1)

    selected = []
    quartile_population = {}
    for quartile in range(1, 5):
        pool = [row for row in ranked if row["coverage_quartile"] == quartile]
        quartile_population[str(quartile)] = len(pool)
        rng = random.Random(SEED + quartile)
        selected.extend(rng.sample(pool, 15))
    selected.sort(key=lambda x: (x["coverage_quartile"], x["sample_id"]))

    annotation_fields = {
        "annotator_1_start_sec": None,
        "annotator_1_end_sec": None,
        "annotator_1_status": None,
        "annotator_1_notes": None,
        "annotator_2_start_sec": None,
        "annotator_2_end_sec": None,
        "annotator_2_status": None,
        "annotator_2_notes": None,
        "adjudicated_start_sec": None,
        "adjudicated_end_sec": None,
        "adjudicated_status": None,
        "adjudication_notes": None,
    }
    output_rows = [{**row, **annotation_fields} for row in selected]
    OUT_JSONL.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    freeze = {
        "status": "FROZEN_AWAITING_TWO_HUMAN_ANNOTATORS",
        "purpose": "CMTB visual-action interval oracle upper-bound audit; not an SRVR input and not a generalization experiment.",
        "selection": {
            "prediction_blind": True,
            "model_outputs_read": False,
            "seed": SEED,
            "population_n": 291,
            "sample_n": 60,
            "rule": "15 uniform-random items within each VAD-envelope coverage quartile.",
            "quartile_population": quartile_population,
            "quartile_selected": {str(q): 15 for q in range(1, 5)},
        },
        "annotation_boundary": "Human annotators may see the muted video, current-action label, and automatic speech interval, but not distractor text, model predictions, or correctness. The adjudicated visual span is an oracle analysis only.",
        "source_hashes": {
            str(OFFICIAL.relative_to(ROOT)): sha(OFFICIAL),
            str(GEOMETRY.relative_to(ROOT)): sha(GEOMETRY),
        },
        "output_hashes": {
            str(OUT_JSONL.relative_to(ROOT)): sha(OUT_JSONL),
            str(OUT_CSV.relative_to(ROOT)): sha(OUT_CSV),
        },
    }
    OUT_FREEZE.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
