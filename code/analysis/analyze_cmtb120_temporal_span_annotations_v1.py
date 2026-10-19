#!/usr/bin/env python3
"""Audit CMTB-120 temporal-span agreement and VAD geometry before adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/cmtb120_temporal_span_annotation_blind_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
DEFAULT_A1 = ROOT / "outputs/cmtb120_temporal_spans_annotator_1_v1.jsonl"
DEFAULT_A2 = ROOT / "outputs/cmtb120_temporal_spans_annotator_2_v1.jsonl"
REPORT = ROOT / "outputs/cmtb120_temporal_span_agreement_v1.json"
QUEUE = ROOT / "data/cmtb120_temporal_span_adjudication_queue_v1.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator-1", type=Path, default=DEFAULT_A1)
    parser.add_argument("--annotator-2", type=Path, default=DEFAULT_A2)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def merge(spans: list[list[float]]) -> list[tuple[float, float]]:
    result: list[list[float]] = []
    for start, end in sorted(spans):
        if not result or start > result[-1][1]:
            result.append([float(start), float(end)])
        else:
            result[-1][1] = max(result[-1][1], float(end))
    return [(start, end) for start, end in result]


def length(spans: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in spans)


def intersection(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    total = 0.0
    left = right = 0
    while left < len(a) and right < len(b):
        total += max(0.0, min(a[left][1], b[right][1]) - max(a[left][0], b[right][0]))
        if a[left][1] <= b[right][1]:
            left += 1
        else:
            right += 1
    return total


def geometry(a: list[list[float]], b: list[list[float]]) -> dict[str, float | None]:
    aa, bb = merge(a), merge(b)
    inter = intersection(aa, bb)
    la, lb = length(aa), length(bb)
    union = la + lb - inter
    return {
        "intersection_sec": inter,
        "union_iou": inter / union if union else None,
        "a_recall_by_b": inter / la if la else None,
        "b_recall_by_a": inter / lb if lb else None,
    }


def kappa(left: list[str], right: list[str]) -> float | None:
    if not left:
        return None
    labels = set(left) | set(right)
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    ca, cb = Counter(left), Counter(right)
    expected = sum(ca[x] * cb[x] for x in labels) / (len(left) ** 2)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None, "mean": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values), "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)), "q3": float(np.quantile(array, 0.75)),
        "mean": float(array.mean()),
    }


def per_annotator(rows: dict[str, dict], items: dict[str, dict], cues: dict[str, dict]) -> dict:
    current_iou, current_recall, narrated_overlap = [], [], []
    for sample_id, annotation in rows.items():
        item = items[sample_id]
        cue = cues[item["video_id"]]
        vad = [[cue["speech_start_sec"], cue["speech_end_sec"]]]
        current = geometry(annotation["current_spans"], vad)
        current_iou.append(current["union_iou"])
        current_recall.append(current["a_recall_by_b"])
        if annotation["narrated_status"] == "spans":
            narrated = geometry(annotation["narrated_spans"], vad)
            narrated_overlap.append(narrated["a_recall_by_b"])
    return {
        "n": len(rows),
        "vad_current_union_iou": summarize([x for x in current_iou if x is not None]),
        "current_support_recall_by_vad": summarize([x for x in current_recall if x is not None]),
        "narrated_support_recall_by_vad_when_present": summarize(
            [x for x in narrated_overlap if x is not None]
        ),
        "narrated_option_counts": dict(Counter(r["narrated_option"] for r in rows.values())),
        "narrated_status_counts": dict(Counter(r["narrated_status"] for r in rows.values())),
    }


def main() -> None:
    args = parse_args()
    manifest = read(MANIFEST)
    items = {row["sample_id"]: row for row in manifest}
    cues = {row["video_id"]: row for row in read(CUES)}
    a1 = {row["sample_id"]: row for row in read(args.annotator_1)}
    a2 = {row["sample_id"]: row for row in read(args.annotator_2)}
    expected = set(items)
    if not args.allow_incomplete and (set(a1) != expected or set(a2) != expected):
        raise RuntimeError(f"annotations incomplete: expected={len(expected)} a1={len(a1)} a2={len(a2)}")
    if (set(a1) | set(a2)) - expected:
        raise RuntimeError("annotation contains unknown sample IDs")
    paired = sorted(set(a1) & set(a2))
    categorical = {}
    for field in ("narrated_option", "narrated_status", "confidence"):
        left = [str(a1[sample_id][field]) for sample_id in paired]
        right = [str(a2[sample_id][field]) for sample_id in paired]
        categorical[field] = {
            "raw_agreement": sum(x == y for x, y in zip(left, right)) / len(paired) if paired else None,
            "cohen_kappa": kappa(left, right),
        }
    current_iou = [
        geometry(a1[sample_id]["current_spans"], a2[sample_id]["current_spans"])["union_iou"]
        for sample_id in paired
    ]
    narrated_iou = [
        geometry(a1[sample_id]["narrated_spans"], a2[sample_id]["narrated_spans"])["union_iou"]
        for sample_id in paired
        if a1[sample_id]["narrated_status"] == a2[sample_id]["narrated_status"] == "spans"
    ]
    agreement_ids = [
        sample_id for sample_id in paired
        if a1[sample_id]["narrated_option"] == a2[sample_id]["narrated_option"]
        and a1[sample_id]["narrated_status"] == a2[sample_id]["narrated_status"]
    ]
    disagreement_ids = [sample_id for sample_id in paired if sample_id not in agreement_ids]
    queue = [{
        **items[sample_id], "annotator_1": a1[sample_id], "annotator_2": a2[sample_id],
        "adjudication": None, "model_outputs_seen": False,
    } for sample_id in disagreement_ids]
    QUEUE.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in queue
    ))
    report = {
        "status": (
            "COMPLETE_READY_FOR_HUMAN_ADJUDICATION"
            if set(a1) == expected and set(a2) == expected else "INCOMPLETE_DESCRIPTIVE_ONLY"
        ),
        "target_n": len(expected), "annotator_1_n": len(a1), "annotator_2_n": len(a2),
        "paired_n": len(paired), "categorical_agreement": categorical,
        "current_span_union_iou_between_annotators": summarize([x for x in current_iou if x is not None]),
        "narrated_span_union_iou_between_annotators": summarize([x for x in narrated_iou if x is not None]),
        "annotator_1_geometry": per_annotator(a1, items, cues),
        "annotator_2_geometry": per_annotator(a2, items, cues),
        "category_agreement_n": len(agreement_ids), "adjudication_queue_n": len(queue),
        "automatic_consensus_used": False,
        "hashes": {
            "manifest": sha(MANIFEST), "cues": sha(CUES),
            "annotator_1": sha(args.annotator_1), "annotator_2": sha(args.annotator_2),
            "adjudication_queue": sha(QUEUE),
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
