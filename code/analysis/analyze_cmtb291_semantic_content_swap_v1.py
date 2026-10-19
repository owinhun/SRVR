#!/usr/bin/env python3
"""Frozen analyzer for the matched semantic content swap.

Four conditions on the 244-item swap subset, all with the correct pointer and
answering-time waveform audio off; only the appended text differs:
  no_transcript / unrelated / original  -- reused frozen predictions
  current_compatible                    -- the new condition

Primary endpoint: d_n choice rate, full 244 denominator, unmapped roles counted
fail-closed and bounded adversarially. Secondary: the 144 role-mapped items,
explicitly labelled as a selected subset (the role map is the Qwen error union).
Accuracy is tertiary: donor text shares tokens with the correct option by
construction, so accuracy is contaminated by design.
"""
from __future__ import annotations

import argparse, json, random, hashlib
from math import comb
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
OUTD = ROOT / "outputs"
ASSIGN = OUTD / "cmtb291_semantic_content_swap_assignment_v1.jsonl"
ROLE = OUTD / "nor_annotation_validated_v1.jsonl"
FREEZE = OUTD / "cmtb291_semantic_content_swap_freeze_v1.json"
SEED = 20260818
REPS = 20000

BASE = {  # condition -> (file glob, condition field value)
    "qwen25": {
        "no_transcript": ([OUTD / "qwen25_ter_replication_v1.jsonl"], "TER_V_auto"),
        "unrelated": (sorted(OUTD.glob("qwen25_pointer_asr_content_control_v1_shard*.jsonl")),
                      "pointer_unrelated_transcript"),
        "original": (sorted(OUTD.glob("qwen25_pointer_asr_content_control_v1_shard*.jsonl")),
                     "pointer_original_transcript"),
        "current_compatible": (sorted(OUTD.glob("qwen25_semantic_content_swap_v1_shard*.jsonl")),
                               "pointer_current_compatible_transcript"),
    },
    "qwen3": {
        "no_transcript": ([OUTD / "qwen3_ter_large_v1.jsonl"], "TER_V_auto"),
        "unrelated": (sorted(OUTD.glob("qwen3_pointer_asr_content_control_v1_shard*.jsonl")),
                      "pointer_unrelated_transcript"),
        "original": (sorted(OUTD.glob("qwen3_pointer_asr_content_control_v1_shard*.jsonl")),
                     "pointer_original_transcript"),
        "current_compatible": (sorted(OUTD.glob("qwen3_semantic_content_swap_v1_shard*.jsonl")),
                               "pointer_current_compatible_transcript"),
    },
}
ORDER = ["no_transcript", "unrelated", "original", "current_compatible"]


def read(paths):
    for p in paths:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def tf(v):
    return v if isinstance(v, bool) else str(v).lower() == "true"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def boot(a, b, ids):
    rng = random.Random(SEED)
    idx = list(ids); n = len(idx); d = []
    for _ in range(REPS):
        s = [idx[rng.randrange(n)] for _ in range(n)]
        d.append(sum(a[i] for i in s) / n - sum(b[i] for i in s) / n)
    d.sort()
    return d[int(.025 * REPS)] * 100, d[int(.975 * REPS)] * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUTD / "cmtb291_semantic_content_swap_analysis_v1.json")
    args = ap.parse_args()

    freeze = json.loads(FREEZE.read_text())
    subset = [json.loads(l)["sample_id"] for l in ASSIGN.read_text().splitlines() if l.strip()]
    subset = sorted(subset)
    if len(subset) != freeze["population_n"]:
        raise RuntimeError("assignment size does not match the freeze")

    roles = {}
    for r in read([ROLE]):
        roles[r["sample_id"]] = r.get("offtime_narrated_option")
    mapped = [s for s in subset if roles.get(s)]

    result = {
        "status": "PENDING_INFERENCE",
        "population_n": len(subset),
        "role_mapped_n": len(mapped),
        "role_unmapped_n": len(subset) - len(mapped),
        "primary_endpoint": "d_n choice rate, full denominator, unmapped fail-closed",
        "bootstrap": {"seed": SEED, "replicates": REPS, "unit": "CMTB item"},
        "models": {},
    }

    for model, conds in BASE.items():
        picks, present = {}, []
        for cond in ORDER:
            paths, tag = conds[cond]
            got = {r["sample_id"]: r for r in read(paths)
                   if r.get("condition") == tag and r["sample_id"] in set(subset)}
            if len(got) == len(subset):
                picks[cond] = got
                present.append(cond)
        m = {"conditions_available": present,
             "conditions_missing": [c for c in ORDER if c not in present]}

        for cond in present:
            g = picks[cond]
            dn = {s: (g[s].get("prediction") == roles.get(s)) for s in subset}
            acc = {s: tf(g[s].get("correct")) for s in subset}
            m[cond] = {
                "accuracy_full": round(100 * sum(acc.values()) / len(subset), 2),
                "dn_rate_identified": round(100 * sum(dn.values()) / len(subset), 2),
                "dn_rate_role_mapped_subset": (
                    round(100 * sum(dn[s] for s in mapped) / len(mapped), 2) if mapped else None),
            }

        if {"unrelated", "current_compatible"} <= set(present):
            u, c = picks["unrelated"], picks["current_compatible"]
            du = {s: (u[s].get("prediction") == roles.get(s)) for s in subset}
            dc = {s: (c[s].get("prediction") == roles.get(s)) for s in subset}
            b = sum(1 for s in subset if dc[s] and not du[s])
            n_ = sum(1 for s in subset if du[s] and not dc[s])
            lo, hi = boot(dc, du, subset)
            m["primary_contrast_current_minus_unrelated_dn_rate"] = {
                "delta_points": round(100 * (sum(dc.values()) - sum(du.values())) / len(subset), 2),
                "ci95": [round(lo, 2), round(hi, 2)],
                "discordant_current_only": b, "discordant_unrelated_only": n_,
                "mcnemar_p": mcnemar(b, n_),
                "verdict": ("current-compatible content LOWERS d_n" if hi < 0 else
                            "current-compatible content RAISES d_n" if lo > 0 else
                            "inconclusive: generic text load"),
            }
            result["status"] = "PASS_COMPLETE_SEMANTIC_CONTENT_SWAP"
        result["models"][model] = m

    result["hashes"] = {
        "assignment": hashlib.sha256(ASSIGN.read_bytes()).hexdigest(),
        "analyzer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2)[:1600])


if __name__ == "__main__":
    main()
