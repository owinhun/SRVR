#!/usr/bin/env python3
"""Analyze the prospectively frozen choice-free CMTB Dev-48 kill test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest
from transformers import AutoModel, AutoTokenizer


ROOT = Path("/workspace/svhalluc")
PROTOCOL = ROOT / "paper/cmtb_native_generation_dev48_protocol_v1.md"
PROTOCOL_SHA = "8b45d7156c6ec792a4668ab402a5faf10fa28f34b598c496539a5b523b0da5a0"
MODEL_PATH = (
    ROOT.parent
    / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CONDITIONS = ("native_raw_av", "native_srvr")
LETTERS = ("A", "B", "C")
BOOTSTRAP_DRAWS = 20_000
SEED = 20260811


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen25", type=Path, default=ROOT / "outputs/cmtb_native_generation_dev48_qwen25_v1.jsonl")
    parser.add_argument("--qwen3", type=Path, default=ROOT / "outputs/cmtb_native_generation_dev48_qwen3_v1.jsonl")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs/cmtb_native_generation_dev48_analysis_v1.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "outputs/cmtb_native_generation_dev48_analysis_v1.md")
    parser.add_argument("--mapped-output", type=Path, default=ROOT / "outputs/cmtb_native_generation_dev48_mapped_v1.jsonl")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(a: str, b: str) -> float:
    left, right = Counter(tokens(a)), Counter(tokens(b))
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def top_letter(scores: dict[str, float]) -> tuple[str, float]:
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return ranked[0][0], ranked[0][1] - ranked[1][1]


def paired_bootstrap(raw: np.ndarray, srvr: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    n = len(raw)
    indices = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    differences = (srvr[indices] - raw[indices]).mean(axis=1) * 100
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def encode_minilm(texts: list[str]) -> np.ndarray:
    """Sentence-Transformers mean pooling for the frozen all-MiniLM snapshot."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
    model.eval()
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), 128):
        encoded = tokenizer(
            texts[start:start + 128], padding=True, truncation=True,
            max_length=256, return_tensors="pt",
        )
        with torch.inference_mode():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(pooled.cpu().numpy())
    return np.concatenate(batches, axis=0)


def validate_file(path: Path, family: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    keys = [(row.get("sample_id"), row.get("condition")) for row in rows]
    if len(rows) != 96 or len(set(keys)) != 96:
        raise RuntimeError(f"{family}: expected complete unique 48x2 rows, got {len(rows)}")
    if {row.get("model_family") for row in rows} != {family}:
        raise RuntimeError(f"{family}: model-family mismatch")
    if {row.get("condition") for row in rows} != set(CONDITIONS):
        raise RuntimeError(f"{family}: condition mismatch")
    counts = Counter(row["condition"] for row in rows)
    if any(counts[condition] != 48 for condition in CONDITIONS):
        raise RuntimeError(f"{family}: incomplete condition counts {counts}")
    if any(row.get("protocol_sha256") != PROTOCOL_SHA for row in rows):
        raise RuntimeError(f"{family}: protocol provenance mismatch")
    return rows


def main() -> None:
    args = cli()
    if sha256(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("protocol hash drift")
    rows = validate_file(args.qwen25, "qwen25") + validate_file(args.qwen3, "qwen3")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(MODEL_PATH)

    strings: list[str] = []
    for row in rows:
        strings.append(row["native_output"])
        strings.extend(row["choices_evaluation_only"][letter] for letter in LETTERS)
    unique = sorted(set(strings))
    embeddings = encode_minilm(unique)
    vector = {text: embeddings[index] for index, text in enumerate(unique)}

    mapped: list[dict[str, Any]] = []
    prompt_audit_pass = True
    for row in rows:
        output = row["native_output"]
        choices = row["choices_evaluation_only"]
        cosine_scores = {
            letter: float(np.dot(vector[output], vector[choices[letter]])) for letter in LETTERS
        }
        token_scores = {letter: token_f1(output, choices[letter]) for letter in LETTERS}
        cosine_letter, cosine_margin = top_letter(cosine_scores)
        token_letter, token_margin = top_letter(token_scores)
        leaked = any(choice.lower() in row["prompt"].lower() for choice in choices.values())
        prompt_audit_pass = prompt_audit_pass and not leaked and not row.get("prompt_contains_choices", True)
        mapped.append({
            **row,
            "cosine_scores": cosine_scores,
            "cosine_prediction": cosine_letter,
            "cosine_margin": cosine_margin,
            "cosine_correct": cosine_letter == row["answer_evaluation_only"],
            "token_f1_scores": token_scores,
            "token_f1_prediction": token_letter,
            "token_f1_margin": token_margin,
            "token_f1_correct": token_letter == row["answer_evaluation_only"],
            "mapping_agreement": cosine_letter == token_letter,
            "choice_leaked_into_prompt": leaked,
            "mapping_model_revision": MODEL_REVISION,
        })

    by_key = {(row["model_family"], row["sample_id"], row["condition"]): row for row in mapped}
    report_models: dict[str, Any] = {}
    all_gates: list[bool] = [prompt_audit_pass]
    for family in ("qwen25", "qwen3"):
        sample_ids = sorted({row["sample_id"] for row in mapped if row["model_family"] == family})
        raw_rows = [by_key[(family, sample_id, "native_raw_av")] for sample_id in sample_ids]
        srvr_rows = [by_key[(family, sample_id, "native_srvr")] for sample_id in sample_ids]
        raw = np.array([row["cosine_correct"] for row in raw_rows], dtype=float)
        srvr = np.array([row["cosine_correct"] for row in srvr_rows], dtype=float)
        token_raw = np.array([row["token_f1_correct"] for row in raw_rows], dtype=float)
        token_srvr = np.array([row["token_f1_correct"] for row in srvr_rows], dtype=float)
        wrong_to_correct = int(np.sum((raw == 0) & (srvr == 1)))
        correct_to_wrong = int(np.sum((raw == 1) & (srvr == 0)))
        discordant = wrong_to_correct + correct_to_wrong
        mcnemar = float(binomtest(wrong_to_correct, discordant, 0.5).pvalue) if discordant else 1.0
        cosine_delta = float((srvr.mean() - raw.mean()) * 100)
        token_delta = float((token_srvr.mean() - token_raw.mean()) * 100)
        margin_coverage = {
            condition: float(np.mean([
                row["cosine_margin"] >= 0.05
                for row in mapped
                if row["model_family"] == family and row["condition"] == condition
            ]))
            for condition in CONDITIONS
        }
        empty = {
            condition: sum(
                bool(row["output_empty"])
                for row in mapped
                if row["model_family"] == family and row["condition"] == condition
            )
            for condition in CONDITIONS
        }
        gates = {
            "complete_nonempty": len(sample_ids) == 48 and sum(empty.values()) == 0,
            "cosine_gain_at_least_10pp": cosine_delta >= 10.0 - 1e-9,
            "net_correction_at_least_5": wrong_to_correct - correct_to_wrong >= 5,
            "token_f1_nonnegative_direction": token_delta >= -1e-9,
            "margin_005_coverage_at_least_70pct_each": all(value >= 0.70 for value in margin_coverage.values()),
        }
        all_gates.extend(gates.values())
        report_models[family] = {
            "n": len(sample_ids),
            "cosine_mapping": {
                "raw_accuracy": float(raw.mean()),
                "srvr_accuracy": float(srvr.mean()),
                "delta_percentage_points": cosine_delta,
                "paired_bootstrap_95ci_percentage_points": paired_bootstrap(raw, srvr),
                "wrong_to_correct": wrong_to_correct,
                "correct_to_wrong": correct_to_wrong,
                "exact_mcnemar_p": mcnemar,
            },
            "token_f1_mapping": {
                "raw_accuracy": float(token_raw.mean()),
                "srvr_accuracy": float(token_srvr.mean()),
                "delta_percentage_points": token_delta,
            },
            "mapping_agreement": {
                condition: float(np.mean([
                    row["mapping_agreement"]
                    for row in mapped
                    if row["model_family"] == family and row["condition"] == condition
                ]))
                for condition in CONDITIONS
            },
            "margin_005_coverage": margin_coverage,
            "empty_outputs": empty,
            "gates": gates,
        }

    status = "PASS_AUTHORIZE_SEPARATELY_FROZEN_FULL_N_NATIVE_PROTOCOL" if all(all_gates) else "STOP_NATIVE_DEV48_GATE"
    report = {
        "schema": "cmtb_native_generation_dev48_analysis_v1",
        "status": status,
        "scope": "prospective choice-free Dev-48 kill test; not an official benchmark score",
        "protocol_sha256": PROTOCOL_SHA,
        "mapping_model_revision": MODEL_REVISION,
        "prompt_audit_pass": prompt_audit_pass,
        "models": report_models,
        "full_n_inference_authorized": status.startswith("PASS_"),
        "human_semantic_claim_authorized": False,
        "thresholds_retuned": False,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "input_hashes": {"qwen25": sha256(args.qwen25), "qwen3": sha256(args.qwen3)},
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.mapped_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in mapped),
        encoding="utf-8",
    )
    lines = [
        "# CMTB native-generation Dev-48 analysis v1",
        "",
        f"Status: **{status}**",
        "",
        "This is a prospective choice-free Dev-48 diagnostic, not an official SVHalluc score.",
        "",
        "| Model | Cosine Raw | Cosine SRVR | Delta | W→C/C→W | Token-F1 delta | margin≥.05 Raw/SRVR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family, value in report_models.items():
        cosine = value["cosine_mapping"]
        lines.append(
            f"| {family} | {100*cosine['raw_accuracy']:.1f}% | {100*cosine['srvr_accuracy']:.1f}% | "
            f"{cosine['delta_percentage_points']:+.1f} pp | {cosine['wrong_to_correct']}/{cosine['correct_to_wrong']} | "
            f"{value['token_f1_mapping']['delta_percentage_points']:+.1f} pp | "
            f"{100*value['margin_005_coverage']['native_raw_av']:.1f}%/"
            f"{100*value['margin_005_coverage']['native_srvr']:.1f}% |"
        )
    lines.extend([
        "",
        f"- Prompt leakage audit: `{prompt_audit_pass}`",
        f"- Full-N native inference authorized: `{report['full_n_inference_authorized']}`",
        "- Human semantic headline authorized: `false`",
        "- No wording, mapper, threshold, or subset was retuned after outputs.",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
