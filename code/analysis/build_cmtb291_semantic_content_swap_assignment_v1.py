#!/usr/bin/env python3
"""Frozen, prediction-blind donor assignment for the current-compatible ASR condition.

Mirrors build of the existing unrelated-donor assignment
(outputs/qwen25_pointer_asr_content_control_assignment_v1.jsonl) with the
matching criterion inverted: the donor must share content tokens with the
CORRECT option and share none with either distractor.

Permitted inputs : CMTB manifest, frozen ASR cue text, official answer labels.
Forbidden inputs : model predictions, correctness, any new-condition outcome.

The answer label is used to CONSTRUCT the stimulus. That is inherent to the
design -- a "current-compatible content" condition cannot be defined without
knowing which option is current. The measured endpoint stays prediction-blind.
"""
import json, re, hashlib, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/qwen25_pointer_asr_content_control_assignment_v1.jsonl"
OUT = ROOT / "outputs/cmtb291_semantic_content_swap_assignment_v1.jsonl"
AUDIT = ROOT / "outputs/cmtb291_semantic_content_swap_assignment_audit_v1.json"

# frozen length band, identical to the existing unrelated-donor condition
WC_LO, WC_HI = 0.80, 1.34

STOP = set("""a an the and or of to in on at for with without from by is are was were be been being
this that these those it its as into onto over under up down out off then than so if you your we
our they them he she his her i me my will would can could should may might do does did done have
has had not no yes there here when what which who whom how why all any both each few more most
other some such only own same too very s t don now just also get got go goes going put puts
""".split())


def toks(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2}


def parse_options(q):
    parts = re.split(r"\(([A-C])\)\s*", q)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]

targets = []
for r in rows:
    opts = parse_options(r["question"])
    if len(opts) != 3:
        raise SystemExit(f"option parse failed for {r['sample_id']}")
    ans = r["answer_evaluation_only"]
    targets.append({
        "sample_id": r["sample_id"], "video_id": r["video_id"],
        "asr": r["original_asr_transcript"], "wc": r["original_asr_word_count"],
        "answer": ans, "options": opts,
        "question": r["question"],
        "speech_start_sec": r["speech_start_sec"], "speech_end_sec": r["speech_end_sec"],
        "ans_toks": toks(opts[ans]),
        "distractor_toks": set().union(*[toks(v) for k, v in opts.items() if k != ans]),
    })

pool = [{"sample_id": t["sample_id"], "video_id": t["video_id"], "asr": t["asr"],
         "wc": t["wc"], "toks": toks(t["asr"])} for t in targets]

assigned, skipped = [], []
for t in targets:
    cands = []
    for d in pool:
        if d["video_id"] == t["video_id"] or d["sample_id"] == t["sample_id"]:
            continue
        ratio = d["wc"] / max(t["wc"], 1)
        if not (WC_LO <= ratio <= WC_HI):
            continue
        hit = d["toks"] & t["ans_toks"]
        if not hit:
            continue
        if d["toks"] & t["distractor_toks"]:
            continue
        cands.append((len(hit), -abs(ratio - 1.0), d["sample_id"], d, sorted(hit), ratio))
    if not cands:
        skipped.append(t["sample_id"])
        continue
    # deterministic: most answer-token overlap, then closest length, then sample-ID
    cands.sort(key=lambda c: (-c[0], -c[1], c[2]))
    _, _, _, d, hit, ratio = cands[0]
    assigned.append({
        "sample_id": t["sample_id"], "video_id": t["video_id"],
        "answer_evaluation_only": t["answer"],
        "question": t["question"],
        "speech_start_sec": t["speech_start_sec"], "speech_end_sec": t["speech_end_sec"],
        "original_asr_transcript": t["asr"], "original_asr_word_count": t["wc"],
        "donor_sample_id": d["sample_id"], "donor_video_id": d["video_id"],
        "donor_asr_transcript": d["asr"], "donor_asr_word_count": d["wc"],
        "shared_answer_content_tokens": hit,
        "donor_distractor_overlap": [],
        "word_count_ratio": ratio,
        "selection": ("different sample/video; shares >=1 correct-option content token; "
                      "zero distractor-option content overlap; word-count ratio in "
                      f"[{WC_LO},{WC_HI}]; max overlap then closest length then sample-ID"),
        "selection_used_model_prediction_or_correctness": False,
        "selection_used_answer_label": True,
    })

OUT.write_text("".join(json.dumps(a, sort_keys=True) + "\n" for a in assigned))

ratios = sorted(a["word_count_ratio"] for a in assigned)
ov = sorted(len(a["shared_answer_content_tokens"]) for a in assigned)
audit = {
    "status": "PASS_PREDICTION_BLIND_CURRENT_COMPATIBLE_ASSIGNMENT" if assigned else "FAIL",
    "targets_considered": len(targets),
    "assigned_n": len(assigned),
    "skipped_n": len(skipped),
    "skipped_sample_ids": skipped,
    "self_donor_n": 0,
    "same_video_donor_n": 0,
    "donor_distractor_overlap_n": 0,
    "word_count_band": [WC_LO, WC_HI],
    "word_count_ratio_min": ratios[0],
    "word_count_ratio_median": ratios[len(ratios) // 2],
    "word_count_ratio_max": ratios[-1],
    "answer_token_overlap_min": ov[0],
    "answer_token_overlap_median": ov[len(ov) // 2],
    "answer_token_overlap_max": ov[-1],
    "permitted_inputs": ["CMTB manifest", "frozen ASR cue text", "official answer label"],
    "forbidden_inputs": ["model predictions", "correctness", "new-condition outcomes"],
    "source_assignment_sha256": sha(SRC),
    "assignment_sha256": sha(OUT),
    "code_sha256": sha(__file__),
}
AUDIT.write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps({k: v for k, v in audit.items() if k != "skipped_sample_ids"}, indent=2))
