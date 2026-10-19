#!/usr/bin/env python3
"""Localized-delivery Audio x Reference interaction for Qwen3, all 291 items.

  numeric   interaction = (TER_AV_auto - TER_V_auto) - (O_raw - V)
  localized interaction = (AV_VADclip  - V_VADclip ) - (O_raw - V)

Separates "the reversal is lineage-specific" from "the reversal is
numeric-delivery-specific". See the frozen protocol for the decision rule.
"""
import json, random, hashlib
from math import comb
from pathlib import Path

ROOT = Path("/workspace/svhalluc")
OUTD = ROOT / "outputs"
NUMERIC_CELL_NAMES = {
    "no_pointer_audio_off": "V",
    "no_pointer_audio_on": "O_raw",
    "correct_pointer_audio_off": "TER_V_auto",
    "correct_pointer_audio_on": "TER_AV_auto",
}
# per-model sources: (no-reference + numeric cells, localized-off cell, new localized-on cell)
SOURCES = {
    "qwen3": {
        "base": [OUTD / "qwen3_ter_large_v1.jsonl"],
        "clip_off": [OUTD / "qwen3_ter_large_v1.jsonl"],
        "clip_on": [OUTD / "qwen3_localized_av_clip_v1.jsonl"],
    },
    "qwen25": {
        "base": ([OUTD / "qwen25_ter_replication_v1.jsonl"]
                 + sorted((OUTD / "v45_audio").glob("qwen25_ter_av_auto_shard*.jsonl"))),
        "clip_off": [OUTD / "qwen25_temporal_controls_v1.jsonl"],
        "clip_on": sorted(OUTD.glob("qwen25_localized_av_clip_v1_shard*.jsonl")),
    },
    # ChronusOmni and Phi-4 name the same four cells differently in their frozen
    # numeric runners; the localized cells use the shared V_VADclip/AV_VADclip names.
    # amendment 1: exact geometry is primary (matches what the two null systems
    # received); padded is the sensitivity (matches the Qwen localized cells)
    "chronusomni_exact": {
        "base": sorted(OUTD.glob("chronusomni_cmtb291_factorial_v1_shard*.jsonl")),
        "clip_off": sorted(OUTD.glob("chronusomni_localized_av_clip_exact_v1_shard*.jsonl")),
        "clip_on": sorted(OUTD.glob("chronusomni_localized_av_clip_exact_v1_shard*.jsonl")),
        "rename": NUMERIC_CELL_NAMES,
    },
    "chronusomni_padded": {
        "base": sorted(OUTD.glob("chronusomni_cmtb291_factorial_v1_shard*.jsonl")),
        "clip_off": sorted(OUTD.glob("chronusomni_localized_av_clip_v1_shard*.jsonl")),
        "clip_on": sorted(OUTD.glob("chronusomni_localized_av_clip_v1_shard*.jsonl")),
        "rename": NUMERIC_CELL_NAMES,
    },
    "phi4mm_exact": {
        "base": sorted(OUTD.glob("phi4mm_cmtb291_factorial_full8_shard*_v1.jsonl")),
        "clip_off": sorted(OUTD.glob("phi4mm_localized_av_clip_exact_v1_shard*.jsonl")),
        "clip_on": sorted(OUTD.glob("phi4mm_localized_av_clip_exact_v1_shard*.jsonl")),
        "rename": NUMERIC_CELL_NAMES,
    },
    "phi4mm_padded": {
        "base": sorted(OUTD.glob("phi4mm_cmtb291_factorial_full8_shard*_v1.jsonl")),
        "clip_off": sorted(OUTD.glob("phi4mm_localized_av_clip_padded_v1_shard*.jsonl")),
        "clip_on": sorted(OUTD.glob("phi4mm_localized_av_clip_padded_v1_shard*.jsonl")),
        "rename": NUMERIC_CELL_NAMES,
    },
}
OUT = OUTD / "cmtb291_localized_delivery_interaction_analysis_v1.json"
SEED, REPS = 20260818, 20000


def cells(paths, want, rename=None):
    rename = rename or {}
    got = {c: {} for c in want}
    for path in (paths if isinstance(paths, list) else [paths]):
        if not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            c = rename.get(d.get("condition"), d.get("condition"))
            if c in got and str(d.get("sample_id", "")).startswith("cmtb_"):
                v = d.get("correct")
                got[c][d["sample_id"]] = v if isinstance(v, bool) else str(v).lower() == "true"
    return got


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2)


def boot_int(on1, off1, on0, off0, ids):
    """bootstrap the difference-in-differences"""
    rng = random.Random(SEED)
    idx = list(ids); n = len(idx); d = []
    for _ in range(REPS):
        s = [idx[rng.randrange(n)] for _ in range(n)]
        a = sum(on1[i] for i in s) / n - sum(off1[i] for i in s) / n
        b = sum(on0[i] for i in s) / n - sum(off0[i] for i in s) / n
        d.append(a - b)
    d.sort()
    return d[int(.025 * REPS)] * 100, d[int(.975 * REPS)] * 100


def main():
    out = {"bootstrap": {"seed": SEED, "replicates": REPS}, "models": {}}
    complete = True
    for model, src in SOURCES.items():
        rn = src.get("rename")
        g = cells(src["base"], ["V", "O_raw", "TER_V_auto", "TER_AV_auto"], rn)
        g.update(cells(src["clip_off"], ["V_VADclip"], rn))
        g.update(cells(src["clip_on"], ["AV_VADclip"], rn))

        def interaction(on1, off1):
            ids = sorted(set(g.get(on1, {})) & set(g.get(off1, {})) & set(g["O_raw"]) & set(g["V"]))
            if not ids:
                return None
            n = len(ids)
            eff1 = (sum(g[on1][i] for i in ids) - sum(g[off1][i] for i in ids)) / n
            eff0 = (sum(g["O_raw"][i] for i in ids) - sum(g["V"][i] for i in ids)) / n
            lo, hi = boot_int(g[on1], g[off1], g["O_raw"], g["V"], ids)
            b = sum(1 for i in ids if g[on1][i] and not g[off1][i])
            c = sum(1 for i in ids if g[off1][i] and not g[on1][i])
            return {"n": n,
                    "audio_effect_with_reference": round(100 * eff1, 2),
                    "audio_effect_no_reference": round(100 * eff0, 2),
                    "interaction_points": round(100 * (eff1 - eff0), 2),
                    "ci95": [round(lo, 2), round(hi, 2)],
                    "paired_flips_on_only": b, "paired_flips_off_only": c,
                    "mcnemar_p": mcnemar(b, c)}

        m = {"cell_n": {k: len(v) for k, v in g.items()},
             "cell_accuracy": {k: (round(100 * sum(v.values()) / len(v), 2) if v else None)
                               for k, v in g.items()},
             "numeric_delivery": interaction("TER_AV_auto", "TER_V_auto"),
             "localized_delivery": interaction("AV_VADclip", "V_VADclip")}
        if m["localized_delivery"] is None:
            complete = False
        else:
            lo, hi = m["localized_delivery"]["ci95"]
            m["verdict"] = ("reversal survives a delivery change within one lineage" if hi < 0 else
                            "delivery-specific; withdraw the lineage reading" if lo <= 0 <= hi else
                            "positive interaction under localized delivery; report the contradiction")
        out["models"][model] = m

    out["status"] = ("PASS_COMPLETE_LOCALIZED_DELIVERY_INTERACTION" if complete
                     else "PENDING_INFERENCE")
    out["hashes"] = {"analyzer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
