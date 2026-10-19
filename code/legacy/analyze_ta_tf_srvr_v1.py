#!/usr/bin/env python3
"""Analyze the TA+TF SRVR replication (v2, per-condition jobs, no merge-collision).

Merges test_results.json (predictions, in dataset order) with the
corresponding *_map.jsonl (sample_id/task/answer, in the same order) for
each condition x shard, computes per-condition / per-task accuracy, and
compares VAD_localized_clip against the strict-duration-matched random
control on the same N=390 subset (mirrors the CMTB-291 primary comparison).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/workspace/svhalluc/claude_track")
OUT = ROOT / "outputs"
CONDITIONS = ["raw_av", "video_only", "vad_localized_clip", "matched_random_0"]
COND_LABEL = {
    "raw_av": "Raw_AV",
    "video_only": "Video_only",
    "vad_localized_clip": "VAD_localized_clip",
    "matched_random_0": "Matched_random_0",
}
N_SHARDS = 7


def parse_pred(text: str) -> str | None:
    m = re.search(r"\b([ABC])\b", text.strip())
    return m.group(1) if m else None


def load_condition(cond: str) -> list[dict]:
    rows = []
    for shard_id in range(N_SHARDS):
        result_path = OUT / f"ta_tf_srvr_v2_{cond}_output_shard{shard_id}/test_results.json"
        map_path = OUT / f"ta_tf_srvr_v2_{cond}_shard{shard_id}_map.jsonl"
        results = json.loads(result_path.read_text())
        maps = [json.loads(l) for l in map_path.read_text().splitlines() if l.strip()]
        assert len(results) == len(maps), f"{cond} shard{shard_id}: {len(results)} vs {len(maps)}"
        for r, m in zip(results, maps):
            pred_letter = parse_pred(r.get("pred", ""))
            rows.append({
                "sample_id": m["sample_id"],
                "task": m["task"],
                "condition": COND_LABEL[cond],
                "answer": m["answer"],
                "prediction": pred_letter,
                "correct": pred_letter == m["answer"],
                "raw_output": r.get("pred", ""),
            })
    return rows


def accuracy(rows: list[dict]) -> tuple[int, int, float]:
    n = len(rows)
    c = sum(1 for r in rows if r["correct"])
    return c, n, (c / n if n else float("nan"))


def bootstrap_ci(rows: list[dict], n_boot: int = 2000, seed: int = 20260802) -> tuple[float, float]:
    import random
    rng = random.Random(seed)
    n = len(rows)
    correct = [1 if r["correct"] else 0 for r in rows]
    means = []
    for _ in range(n_boot):
        s = sum(correct[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return lo, hi


def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    by_id_a = {r["sample_id"]: r["correct"] for r in rows_a}
    by_id_b = {r["sample_id"]: r["correct"] for r in rows_b}
    common = sorted(set(by_id_a) & set(by_id_b))
    b_wrong_a_right = sum(1 for s in common if by_id_a[s] and not by_id_b[s])
    a_wrong_b_right = sum(1 for s in common if by_id_b[s] and not by_id_a[s])
    n = b_wrong_a_right + a_wrong_b_right
    if n == 0:
        return {"n_pairs": len(common), "discordant": 0, "p_value": 1.0}
    # exact binomial two-sided test against p=0.5
    from math import comb
    k = min(a_wrong_b_right, b_wrong_a_right)
    p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
    p = min(p, 1.0)
    return {
        "n_pairs": len(common),
        "a_correct_b_wrong": b_wrong_a_right,
        "b_correct_a_wrong": a_wrong_b_right,
        "discordant": n,
        "p_value": p,
    }


def main() -> None:
    all_rows: dict[str, list[dict]] = {}
    for cond in CONDITIONS:
        all_rows[COND_LABEL[cond]] = load_condition(cond)

    report: dict = {"design": "TA+TF SRVR v2 (Claude track), per-condition jobs, no collision", "conditions": {}}

    for label, rows in all_rows.items():
        c, n, acc = accuracy(rows)
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
        task_acc = {}
        for task, trows in by_task.items():
            tc, tn, tacc = accuracy(trows)
            lo, hi = bootstrap_ci(trows)
            task_acc[task] = {"correct": tc, "n": tn, "accuracy": round(tacc, 4),
                               "bootstrap_ci95": [round(lo, 4), round(hi, 4)]}
        lo, hi = bootstrap_ci(rows)
        report["conditions"][label] = {
            "correct": c, "n": n, "accuracy": round(acc, 4),
            "bootstrap_ci95": [round(lo, 4), round(hi, 4)],
            "by_task": task_acc,
        }

    # Primary comparison on the strict N=390 matched-random-feasible subset:
    # VAD_localized_clip vs Matched_random_0, restricted to sample_ids present in Matched_random_0.
    mr_ids = {r["sample_id"] for r in all_rows["Matched_random_0"]}
    vad_matched = [r for r in all_rows["VAD_localized_clip"] if r["sample_id"] in mr_ids]
    raw_matched = [r for r in all_rows["Raw_AV"] if r["sample_id"] in mr_ids]
    vo_matched = [r for r in all_rows["Video_only"] if r["sample_id"] in mr_ids]

    def summarize(rows, name):
        c, n, acc = accuracy(rows)
        lo, hi = bootstrap_ci(rows)
        return {"label": name, "correct": c, "n": n, "accuracy": round(acc, 4),
                "bootstrap_ci95": [round(lo, 4), round(hi, 4)]}

    report["primary_comparison_strict_n390"] = {
        "Raw_AV": summarize(raw_matched, "Raw_AV"),
        "Video_only": summarize(vo_matched, "Video_only"),
        "VAD_localized_clip": summarize(vad_matched, "VAD_localized_clip"),
        "Matched_random_0": summarize(all_rows["Matched_random_0"], "Matched_random_0"),
        "mcnemar_vad_vs_matched_random": mcnemar(vad_matched, all_rows["Matched_random_0"]),
        "mcnemar_vad_vs_raw_av": mcnemar(vad_matched, raw_matched),
        "mcnemar_vad_vs_video_only": mcnemar(vad_matched, vo_matched),
    }

    # Paired correction pattern: Raw_AV -> VAD_localized_clip (full 570, matching CMTB's headline table)
    raw_by_id = {r["sample_id"]: r["correct"] for r in all_rows["Raw_AV"]}
    vad_by_id = {r["sample_id"]: r["correct"] for r in all_rows["VAD_localized_clip"]}
    common = sorted(set(raw_by_id) & set(vad_by_id))
    both_wrong = sum(1 for s in common if not raw_by_id[s] and not vad_by_id[s])
    both_right = sum(1 for s in common if raw_by_id[s] and vad_by_id[s])
    fixed = sum(1 for s in common if not raw_by_id[s] and vad_by_id[s])
    broken = sum(1 for s in common if raw_by_id[s] and not vad_by_id[s])
    report["correction_pattern_raw_to_vad_n570"] = {
        "n": len(common), "both_wrong": both_wrong, "both_right": both_right,
        "raw_wrong_vad_correct": fixed, "raw_correct_vad_wrong": broken,
    }

    out_path = OUT / "ta_tf_srvr_analysis_v1.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
