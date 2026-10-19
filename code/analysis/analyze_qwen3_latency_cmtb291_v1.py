#!/usr/bin/env python3
"""Analyze fair end-to-end CMTB-291 latency benchmark."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("/workspace/svhalluc")
SHARDS = [
    ROOT / "outputs/qwen3_latency_cmtb291_v1_shard0.jsonl",
    ROOT / "outputs/qwen3_latency_cmtb291_v1_shard1.jsonl",
]
MERGED = ROOT / "outputs/qwen3_latency_cmtb291_v1.jsonl"
OUTPUT = ROOT / "outputs/qwen3_latency_cmtb291_v1_analysis.json"
REPORT = ROOT / "outputs/qwen3_latency_cmtb291_v1_report.md"
CONDITIONS = ("Full_TER", "Uniform32_auto_tau")


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    rows = [x for p in SHARDS for x in read(p)]
    keys = {(x["sample_id"], x["condition"], x["repeat"]) for x in rows}
    if len(rows) != 1746 or len(keys) != 1746:
        raise RuntimeError("Incomplete or duplicate latency rows")
    if any(x["timer_scope"] !=
           "reader_init+decode+preprocess+generation+parse" for x in rows):
        raise RuntimeError("Timer-scope mismatch")
    by = defaultdict(lambda: defaultdict(list))
    for x in rows:
        by[x["condition"]][x["sample_id"]].append(x)
    ids = sorted(by["Full_TER"])
    if len(ids) != 291 or any(
        len(by[c][i]) != 3 for c in CONDITIONS for i in ids
    ):
        raise RuntimeError("Three-repeat coverage mismatch")
    rows.sort(key=lambda x: (x["sample_id"], x["condition"], x["repeat"]))
    MERGED.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))

    per_sample = {}
    summaries = {}
    for condition in CONDITIONS:
        per_sample[condition] = {
            i: {
                "median_latency_sec": float(np.median(
                    [x["latency_sec"] for x in by[condition][i]]
                )),
                "median_peak_memory_bytes": float(np.median([
                    sum(x["peak_gpu_memory_bytes_visible"])
                    for x in by[condition][i]
                ])),
                "median_processed_frames": float(np.median([
                    x["processed_frame_count"] for x in by[condition][i]
                ])),
            } for i in ids
        }
        lat = np.array([
            per_sample[condition][i]["median_latency_sec"] for i in ids
        ])
        mem = np.array([
            per_sample[condition][i]["median_peak_memory_bytes"] for i in ids
        ])
        frames = np.array([
            per_sample[condition][i]["median_processed_frames"] for i in ids
        ])
        condition_rows = [x for x in rows if x["condition"] == condition]
        summaries[condition] = {
            "samples": 291, "measurements": 873,
            "sample_median_latency_sec": float(np.median(lat)),
            "sample_p90_latency_sec": float(np.quantile(lat, .9)),
            "sample_mean_latency_sec": float(np.mean(lat)),
            "median_peak_gpu_memory_sum_bytes": float(np.median(mem)),
            "mean_processed_frames": float(np.mean(frames)),
            "median_processed_frames": float(np.median(frames)),
            "parse_failures": sum(
                x["parse_status"] != "parsed" for x in condition_rows
            ),
        }

    full = np.array([
        per_sample["Full_TER"][i]["median_latency_sec"] for i in ids
    ])
    uniform = np.array([
        per_sample["Uniform32_auto_tau"][i]["median_latency_sec"] for i in ids
    ])
    diff = uniform - full
    rng = np.random.default_rng(20260730)
    boot_diff, boot_reduction = [], []
    for _ in range(20000):
        ix = rng.integers(0, len(ids), len(ids))
        f, u = full[ix], uniform[ix]
        boot_diff.append(float(np.median(u - f)))
        boot_reduction.append(float(1 - np.median(u) / np.median(f)))
    latency_reduction = 1 - np.median(uniform) / np.median(full)
    full_mem = summaries["Full_TER"]["median_peak_gpu_memory_sum_bytes"]
    uniform_mem = summaries["Uniform32_auto_tau"][
        "median_peak_gpu_memory_sum_bytes"
    ]
    memory_reduction = 1 - uniform_mem / full_mem
    frame_reduction = 1 - (
        summaries["Uniform32_auto_tau"]["mean_processed_frames"]
        / summaries["Full_TER"]["mean_processed_frames"]
    )
    test = wilcoxon(diff)
    decision = "END_TO_END_LATENCY_FAIL"
    result = {
        "decision": decision,
        "integrity": {
            "rows": 1746, "samples": 291, "repeats_per_condition": 3,
            "warmup_excluded": True,
            "timer_scope": "reader_init+decode+preprocess+generation+parse",
            "counterbalanced_condition_order": True,
        },
        "conditions": summaries,
        "paired_latency": {
            "median_uniform_minus_full_sec": float(np.median(diff)),
            "mean_uniform_minus_full_sec": float(np.mean(diff)),
            "median_difference_bootstrap_ci": [
                float(np.quantile(boot_diff, .025)),
                float(np.quantile(boot_diff, .975)),
            ],
            "median_latency_reduction": float(latency_reduction),
            "latency_reduction_bootstrap_ci": [
                float(np.quantile(boot_reduction, .025)),
                float(np.quantile(boot_reduction, .975)),
            ],
            "uniform_faster_fraction": float(np.mean(diff < 0)),
            "wilcoxon_statistic": float(test.statistic),
            "wilcoxon_p": float(test.pvalue),
        },
        "efficiency": {
            "mean_processed_frame_reduction": float(frame_reduction),
            "median_peak_memory_reduction": float(memory_reduction),
            "latency_reduction_at_least_15pct": bool(latency_reduction >= .15),
        },
        "interpretation": (
            "Uniform32 preserves accuracy and reduces processed visual frames, "
            "but its explicit sparse-frame decoding/list construction does not "
            "produce end-to-end speedup in this implementation."
        ),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    f, u = summaries["Full_TER"], summaries["Uniform32_auto_tau"]
    ci = result["paired_latency"]["median_difference_bootstrap_ci"]
    rci = result["paired_latency"]["latency_reduction_bootstrap_ci"]
    lines = [
        "# Qwen3 CMTB-291 Fair End-to-End Latency", "",
        f"Decision: **{decision}**", "",
        "| Condition | Median latency | p90 latency | Mean frames | "
        "Median peak GPU memory |",
        "|---|---:|---:|---:|---:|",
        f"| Full-TER | {f['sample_median_latency_sec']:.2f}s | "
        f"{f['sample_p90_latency_sec']:.2f}s | "
        f"{f['mean_processed_frames']:.1f} | "
        f"{f['median_peak_gpu_memory_sum_bytes']/2**30:.2f} GiB |",
        f"| Uniform32+τ | {u['sample_median_latency_sec']:.2f}s | "
        f"{u['sample_p90_latency_sec']:.2f}s | "
        f"{u['mean_processed_frames']:.1f} | "
        f"{u['median_peak_gpu_memory_sum_bytes']/2**30:.2f} GiB |",
        "", "## Paired result", "",
        f"- Median Uniform32 − Full-TER: {np.median(diff):+.2f}s "
        f"(95% video bootstrap CI {ci[0]:+.2f} to {ci[1]:+.2f}s).",
        f"- Median latency reduction: {100*latency_reduction:+.1f}% "
        f"(95% CI {100*rci[0]:+.1f} to {100*rci[1]:+.1f}%).",
        f"- Uniform32 is faster on {100*np.mean(diff < 0):.1f}% of videos.",
        f"- Paired Wilcoxon p={test.pvalue:.3g}.",
        f"- Processed-frame reduction: {100*frame_reduction:.1f}%.",
        f"- Median peak-memory reduction: {100*memory_reduction:.1f}%.",
        "", "## Interpretation", "",
        "Uniform32 retains the previously established accuracy and visual-frame "
        "reduction, but it is slower end to end in the measured implementation. "
        "Direct construction and decoding of a sparse PIL-frame list offsets "
        "the smaller visual-token budget. The pre-registered 15% latency gate "
        "fails; no end-to-end speedup should be claimed.", "",
    ]
    REPORT.write_text("\n".join(lines))
    print(json.dumps({"decision": decision, "analysis": str(OUTPUT),
                      "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
