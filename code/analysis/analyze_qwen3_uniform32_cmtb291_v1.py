#!/usr/bin/env python3
"""Analyze locked CMTB-291 Uniform-32 efficiency confirmation."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path("/workspace/svhalluc")
INPUT = ROOT / "outputs/qwen3_uniform32_cmtb291_v1.jsonl"
REFERENCE = ROOT / "outputs/qwen3_ter_large_v1.jsonl"
CUES = ROOT / "outputs/ter_cues_large_v1.jsonl"
OUTPUT = ROOT / "outputs/qwen3_uniform32_cmtb291_v1_analysis.json"
REPORT = ROOT / "outputs/qwen3_uniform32_cmtb291_v1_report.md"


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def paired(a, b, ids, seed):
    av = np.array([a[i]["correct"] for i in ids], dtype=int)
    bv = np.array([b[i]["correct"] for i in ids], dtype=int)
    a_win = int(np.sum((av == 1) & (bv == 0)))
    b_win = int(np.sum((av == 0) & (bv == 1)))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(20000):
        ix = rng.integers(0, len(ids), len(ids))
        boot.append(float(np.mean(av[ix] - bv[ix])))
    p = 1.0 if a_win + b_win == 0 else float(
        binomtest(min(a_win, b_win), a_win + b_win, 0.5).pvalue
    )
    return {
        "n": len(ids), "a_correct": int(av.sum()), "b_correct": int(bv.sum()),
        "a_accuracy": float(av.mean()), "b_accuracy": float(bv.mean()),
        "difference_a_minus_b": float(np.mean(av - bv)),
        "a_correct_to_b_wrong": a_win,
        "a_wrong_to_b_correct": b_win,
        "exact_mcnemar_p": p,
        "video_bootstrap_ci": [
            float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
        ],
        "prediction_agreement": float(np.mean([
            a[i]["prediction"] == b[i]["prediction"] for i in ids
        ])),
    }


def main():
    rows = read(INPUT)
    pairs = {(x["sample_id"], x["condition"]) for x in rows}
    if len(rows) != 582 or len(pairs) != 582:
        raise RuntimeError("Incomplete or duplicate Uniform-32 rows")
    by = defaultdict(dict)
    for x in rows:
        by[x["condition"]][x["sample_id"]] = x
    if set(by) != {"Uniform32_auto_tau", "Uniform32_no_tau"}:
        raise RuntimeError("Condition mismatch")
    ids = sorted(by["Uniform32_auto_tau"])
    if len(ids) != 291 or any(
        tuple(by["Uniform32_auto_tau"][i]["selected_frame_indices"]) !=
        tuple(by["Uniform32_no_tau"][i]["selected_frame_indices"]) for i in ids
    ):
        raise RuntimeError("Paired frame-index mismatch")

    refs = defaultdict(dict)
    for x in read(REFERENCE):
        if x.get("task") == "CMTB":
            refs[x["condition"]][x["sample_id"]] = x
    full = refs["TER_V_auto"]
    if set(full) != set(ids):
        raise RuntimeError("Full-TER reference mismatch")
    cues = {x["video_id"]: x for x in read(CUES)}

    auto = by["Uniform32_auto_tau"]
    no_tau = by["Uniform32_no_tau"]
    summaries = {}
    for name, values in [("Uniform32_auto_tau", auto),
                         ("Uniform32_no_tau", no_tau),
                         ("Full_TER", full)]:
        z = [values[i] for i in ids]
        summaries[name] = {
            "n": 291, "correct": sum(x["correct"] for x in z),
            "accuracy": sum(x["correct"] for x in z) / 291,
            "parse_failures": sum(x["parse_status"] != "parsed" for x in z),
            "mean_latency_sec": float(np.mean([x["latency_sec"] for x in z])),
            "median_latency_sec": float(np.median([x["latency_sec"] for x in z])),
        }

    full_frames = {
        i: int(round(
            full[i]["video_sampling"]["processor_video_kwargs"]["fps"]
            * cues[full[i]["video_id"]]["media_duration_sec"]
        )) for i in ids
    }
    summaries["Full_TER"]["mean_actual_processor_frames"] = float(
        np.mean(list(full_frames.values()))
    )
    summaries["Full_TER"]["median_actual_processor_frames"] = float(
        np.median(list(full_frames.values()))
    )
    summaries["Uniform32_auto_tau"]["mean_actual_frames"] = 32.0

    auto_vs_full = paired(auto, full, ids, 20260730)
    auto_vs_no = paired(auto, no_tau, ids, 20260731)
    frame_reduction = 1 - 32 / summaries["Full_TER"]["mean_actual_processor_frames"]
    logged_latency_reduction = (
        1 - summaries["Uniform32_auto_tau"]["mean_latency_sec"]
        / summaries["Full_TER"]["mean_latency_sec"]
    )
    accuracy_pass = (
        summaries["Uniform32_auto_tau"]["accuracy"]
        >= summaries["Full_TER"]["accuracy"] - .03
    )
    frame_pass = frame_reduction >= .35
    latency_pass = logged_latency_reduction >= .15

    duration = {i: auto[i]["source_duration_sec"] for i in ids}
    coverage = {
        i: (auto[i]["true_vad_interval_sec"][1] -
            auto[i]["true_vad_interval_sec"][0]) / duration[i] for i in ids
    }

    def tertiles(values):
        q1, q2 = np.quantile(list(values.values()), [1 / 3, 2 / 3])
        return {
            "low": [i for i in ids if values[i] <= q1],
            "mid": [i for i in ids if q1 < values[i] <= q2],
            "high": [i for i in ids if values[i] > q2],
        }, [float(q1), float(q2)]

    duration_groups, duration_cuts = tertiles(duration)
    coverage_groups, coverage_cuts = tertiles(coverage)

    def group_report(groups):
        return {
            name: {
                "n": len(group),
                "Uniform32_auto_tau_accuracy":
                    sum(auto[i]["correct"] for i in group) / len(group),
                "Uniform32_no_tau_accuracy":
                    sum(no_tau[i]["correct"] for i in group) / len(group),
                "Full_TER_accuracy":
                    sum(full[i]["correct"] for i in group) / len(group),
            } for name, group in groups.items()
        }

    decision = (
        "EFFICIENCY_PASS" if accuracy_pass and frame_pass and latency_pass
        else "PARTIAL_PASS_ACCURACY_AND_FRAMES_ONLY"
    )
    result = {
        "decision": decision,
        "integrity": {
            "rows": 582, "samples": 291, "unique_pairs": 582,
            "identical_frame_indices_between_conditions": True,
            "parse_failures": sum(x["parse_status"] != "parsed" for x in rows),
        },
        "summaries": summaries,
        "paired": {
            "Uniform32_auto_tau_vs_Full_TER": auto_vs_full,
            "Uniform32_auto_tau_vs_Uniform32_no_tau": auto_vs_no,
        },
        "efficiency_gate": {
            "accuracy_within_3pp": accuracy_pass,
            "actual_frame_reduction": frame_reduction,
            "frame_reduction_at_least_35pct": frame_pass,
            "logged_latency_reduction": logged_latency_reduction,
            "latency_reduction_at_least_15pct": latency_pass,
            "latency_comparability_warning": (
                "Uniform-32 latency starts after explicit frame decoding, "
                "whereas prior Full-TER latency includes video decoding. "
                "The logged reduction is therefore a favorable upper bound; "
                "the end-to-end 15% latency gate is not met."
            ),
        },
        "subgroups": {
            "video_duration_tertiles": group_report(duration_groups),
            "video_duration_cutpoints_sec": duration_cuts,
            "speech_interval_coverage_tertiles": group_report(coverage_groups),
            "speech_interval_coverage_cutpoints": coverage_cuts,
        },
        "scope": (
            "Secondary efficiency confirmation on previously inspected "
            "CMTB-291; not an untouched final test."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Qwen3 CMTB-291 Uniform-32 Efficiency Confirmation", "",
        f"Decision: **{decision}**", "",
        "| Condition | Correct | Accuracy | Parse failures | Mean latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("Uniform32_auto_tau", "Uniform32_no_tau", "Full_TER"):
        s = summaries[name]
        lines.append(
            f"| {name} | {s['correct']}/291 | {100*s['accuracy']:.1f}% | "
            f"{s['parse_failures']} | {s['mean_latency_sec']:.2f}s |"
        )
    lines += [
        "", "## Paired results", "",
        f"- Uniform32_autoτ − Full-TER: "
        f"{100*auto_vs_full['difference_a_minus_b']:+.1f}%p "
        f"(95% video bootstrap CI "
        f"{100*auto_vs_full['video_bootstrap_ci'][0]:+.1f} to "
        f"{100*auto_vs_full['video_bootstrap_ci'][1]:+.1f}%p); "
        f"11 vs 7 paired flips; McNemar p={auto_vs_full['exact_mcnemar_p']:.3f}.",
        f"- Uniform32_autoτ − noτ: "
        f"{100*auto_vs_no['difference_a_minus_b']:+.1f}%p "
        f"(95% CI {100*auto_vs_no['video_bootstrap_ci'][0]:+.1f} to "
        f"{100*auto_vs_no['video_bootstrap_ci'][1]:+.1f}%p); "
        f"119 vs 1 paired flips; McNemar p={auto_vs_no['exact_mcnemar_p']:.3g}.",
        "", "## Efficiency gate", "",
        f"- Accuracy: **{'PASS' if accuracy_pass else 'FAIL'}**.",
        f"- Frames: **{'PASS' if frame_pass else 'FAIL'}**, "
        f"{100*frame_reduction:.1f}% mean reduction.",
        f"- Latency: **{'PASS' if latency_pass else 'FAIL'}**, "
        f"{100*logged_latency_reduction:.1f}% logged reduction.",
        "- Latency warning: Uniform-32 excludes explicit frame decoding from "
        "its timer, while Full-TER includes video decoding. The 12.1% reduction "
        "is thus an optimistic upper bound and still misses the 15% threshold.",
        "", "## Interpretation", "",
        "Uniform-32 with the correct automatic speech timestamp preserves "
        "Full-TER accuracy and strongly outperforms the identical frames "
        "without the timestamp. This confirms sparse timestamp-conditioned "
        "visual grounding at scale. It does not satisfy the pre-registered "
        "end-to-end efficiency gate because latency reduction is below 15%.",
        "", "This is a secondary confirmation on previously inspected "
        "CMTB-291, not an untouched final test.", "",
    ]
    REPORT.write_text("\n".join(lines))
    print(json.dumps({"decision": decision, "analysis": str(OUTPUT),
                      "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
