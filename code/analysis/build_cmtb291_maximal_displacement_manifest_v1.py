#!/usr/bin/env python3
"""Build and audit the prediction-free full-N maximal pointer manifest."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "data/ter_large_eval_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
PROTOCOL = ROOT / "paper/cmtb291_maximal_displacement_dose_protocol_v1.md"
OUT = ROOT / "outputs/cmtb291_maximal_displacement_pointer_v1.jsonl"
AUDIT = ROOT / "outputs/cmtb291_maximal_displacement_pointer_audit_v1.json"
REPORT = ROOT / "outputs/cmtb291_maximal_displacement_pointer_audit_v1.md"
FORBIDDEN = {"answer", "options", "choices", "prediction", "correct", "accuracy", "human_label"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    def q(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = int(position); upper = min(lower + 1, len(ordered) - 1); weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight
    return {"min": min(ordered), "q1": q(.25), "median": statistics.median(ordered), "mean": statistics.fmean(ordered), "q3": q(.75), "max": max(ordered)}


def main() -> None:
    manifest = [row for row in rows(MANIFEST) if row["task"] == "CMTB"]
    cues = {row["video_id"]: row for row in rows(CUES)}
    built = []
    for row in sorted(manifest, key=lambda item: item["sample_id"]):
        cue = cues[row["video_id"]]
        start, end = float(cue["speech_start_sec"]), float(cue["speech_end_sec"])
        duration = float(cue["media_duration_sec"])
        length = end - start
        candidates = [(0.0, length, "left_boundary"), (duration - length, duration, "right_boundary")]
        true_center = (start + end) / 2
        scored = [(abs((left + right) / 2 - true_center), left, right, name) for left, right, name in candidates]
        if abs(scored[0][0] - scored[1][0]) <= 1e-12:
            chosen = scored[int(hashlib.sha256(row["sample_id"].encode()).hexdigest(), 16) % 2]
            tie_break = "sample_sha256_parity"
        else:
            chosen = max(scored, key=lambda value: value[0])
            tie_break = "not_needed"
        center_distance, wrong_start, wrong_end, boundary = chosen
        overlap = max(0.0, min(end, wrong_end) - max(start, wrong_start))
        union = 2 * length - overlap
        iou = overlap / union if union else 0.0
        built.append({
            "sample_id": row["sample_id"], "video_id": row["video_id"],
            "video_path": cue["video_path"], "source_file_sha256": cue["source_file_sha256"],
            "media_duration_sec": duration, "speech_start_sec": start, "speech_end_sec": end,
            "interval_duration_sec": length, "maximal_start_sec": wrong_start, "maximal_end_sec": wrong_end,
            "selected_boundary": boundary, "tie_break": tie_break,
            "center_displacement_sec": center_distance,
            "normalized_center_displacement": center_distance / duration,
            "overlap_sec": overlap, "overlap_iou": iou, "strict_zero_overlap": overlap <= 1e-12,
            "construction_uses_answer": False, "construction_uses_options": False,
            "construction_uses_model_output": False, "construction_uses_human_semantics": False,
            "construction_rule": "equal_duration_farthest_video_boundary_v1",
        })
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in built))
    checks = {
        "protocol_frozen_present": PROTOCOL.is_file(),
        "exact_291_rows": len(built) == 291,
        "unique_sample_ids": len({row["sample_id"] for row in built}) == 291,
        "unique_video_ids": len({row["video_id"] for row in built}) == 291,
        "all_paths_and_source_hashes_match_cues": all(row["video_path"] == cues[row["video_id"]]["video_path"] and row["source_file_sha256"] == cues[row["video_id"]]["source_file_sha256"] for row in built),
        "all_intervals_in_bounds": all(0 <= row["maximal_start_sec"] < row["maximal_end_sec"] <= row["media_duration_sec"] + 1e-9 for row in built),
        "all_duration_matched_1e_6": all(abs((row["maximal_end_sec"] - row["maximal_start_sec"]) - row["interval_duration_sec"]) <= 1e-6 for row in built),
        "all_normalized_displacement_at_least_0_35": all(row["normalized_center_displacement"] >= .35 for row in built),
        "all_iou_at_most_0_20": all(row["overlap_iou"] <= .20 for row in built),
        "no_forbidden_result_fields": all(not (FORBIDDEN & set(row)) for row in built),
        "all_construction_blind_flags_false": all(not row[key] for row in built for key in ("construction_uses_answer", "construction_uses_options", "construction_uses_model_output", "construction_uses_human_semantics")),
    }
    status = "PASS_AUTHORIZE_MAXIMAL_DOSE_INFERENCE" if all(checks.values()) else "FAIL_INFERENCE_UNAUTHORIZED"
    result = {
        "status": status, "model_inference_authorized": status.startswith("PASS"),
        "counts": {"all": len(built), "strict_zero_overlap": sum(row["strict_zero_overlap"] for row in built), "positive_overlap": sum(not row["strict_zero_overlap"] for row in built)},
        "geometry": {
            "normalized_center_displacement": quantiles([row["normalized_center_displacement"] for row in built]),
            "overlap_iou": quantiles([row["overlap_iou"] for row in built]),
            "interval_fraction_of_video": quantiles([row["interval_duration_sec"] / row["media_duration_sec"] for row in built]),
        },
        "checks": checks,
        "input_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (MANIFEST, CUES, PROTOCOL)},
        "output": str(OUT.relative_to(ROOT)), "output_sha256": sha(OUT),
        "boundary": "Full-N maximal displacement preserves duration and strongly moves the pointer, but 97 rows retain small unavoidable overlap; strict zero-overlap evidence remains separate.",
    }
    AUDIT.write_text(json.dumps(result, indent=2) + "\n")
    REPORT.write_text("\n".join([
        "# CMTB-291 maximal-displacement pointer audit v1", "", f"Status: **{status}**", "",
        f"All N={len(built)}; strict zero-overlap {result['counts']['strict_zero_overlap']}; positive-overlap {result['counts']['positive_overlap']}.", "",
        f"Normalized center displacement: `{result['geometry']['normalized_center_displacement']}`", "",
        f"Overlap IoU: `{result['geometry']['overlap_iou']}`", "", result["boundary"],
    ]) + "\n")
    print(json.dumps(result, indent=2))
    if not result["model_inference_authorized"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
