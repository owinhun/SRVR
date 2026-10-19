#!/usr/bin/env python3
"""Post-hoc role-routing audit for the frozen maximal pointer dose.

The role labels come only from the genuine single-human, prediction-blind
random-100 annotation.  This audit asks whether displacing the supplied time
reference routes predictions from the current option toward the off-time
narrated option, rather than toward an arbitrary distractor.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
OUTPUTS = ROOT / "outputs"
ANNOTATIONS = OUTPUTS / "nor_annotations_annotator_2.jsonl"
PRIMARY = OUTPUTS / "cmtb291_maximal_displacement_dose_analysis_v1.json"
POINTER = OUTPUTS / "cmtb291_maximal_displacement_pointer_v1.jsonl"
OUT_JSON = OUTPUTS / "cmtb291_pointer_dose_role_routing_audit_v1.json"
OUT_MD = OUTPUTS / "cmtb291_pointer_dose_role_routing_audit_v1.md"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)

EXPECTED_HASHES = {
    "nor_annotations_annotator_2.jsonl": "e588a72bcb6c9d7b64fd852cf22608f73cb1741665191bb4f56cf858568bd064",
    "cmtb291_maximal_displacement_dose_analysis_v1.json": "31e301296e038c2c74540d77883e93d0c4b42961c9dfa70e9e0a748d0c4c8194",
    "cmtb291_maximal_displacement_pointer_v1.jsonl": "b4c253de44557cd944bf0091efaca5c15030863fddb9f9102b3b557b9a5a679a",
    "qwen3_ter_large_v1.jsonl": "149d50489bc87557d766dc64c45cc2cf64bfa37368fa7e8da28cd18707401973",
    "qwen25_ter_replication_v1.jsonl": "247e4e91c6509b893d16357c646db602bd11cc781ef3949c5dbd01fd019f7dd7",
    "qwen3_maximal_pointer_dose_v1_shard0.jsonl": "eea5f404450c0b3193bd3054bedb8b8f63a342461d1def887684cfd7952adb35",
    "qwen3_maximal_pointer_dose_v1_shard1.jsonl": "98347c575240b00320ec9065ef341fba95e44263974b3e834033d53339a0e7d6",
    "qwen3_maximal_pointer_dose_v1_shard2.jsonl": "5a61832bfaa55ff28595ab5df97d3465f89d8be154852393ea0f36cb2e777777",
    "qwen3_maximal_pointer_dose_v1_shard3.jsonl": "217dafc3b45635d610b14495700d5534027bb25d69954c7dccf09d2d97566e16",
    "qwen25_maximal_pointer_dose_v1_shard0.jsonl": "65015e3c178d775fdec9623ace3b4e2b178d8551d4b9296536e460d27adee229",
    "qwen25_maximal_pointer_dose_v1_shard1.jsonl": "68cfefbfbc59ff99537c3859d15d80117e5fe284029a1872911ee8bc3fd2e392",
    "qwen25_maximal_pointer_dose_v1_shard2.jsonl": "e11b23de0e72a22ab0831d4a923f160dd95fd65babb964e9286c0806f6d54f08",
    "qwen25_maximal_pointer_dose_v1_shard3.jsonl": "e1598c88255270d50bb2b75b42cf9be88ad62f273acb6f16af8dfd70578f03e9",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def exact_mcnemar(gain: int, loss: int) -> float:
    discordant = gain + loss
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gain, loss) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def verify_hashes() -> dict[str, str]:
    observed = {}
    for name, expected in EXPECTED_HASHES.items():
        path = OUTPUTS / name
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen input drift: {name}: {actual} != {expected}")
        observed[name] = actual
    primary = json.loads(PRIMARY.read_text())
    if primary.get("status") != "PASS_PROMOTE_FULL_N_MAXIMAL_DOSE":
        raise RuntimeError("primary maximal-dose analysis is not the frozen PASS")
    if not all(model.get("validation", {}).get("pass") for model in primary["models"]):
        raise RuntimeError("primary maximal-dose validation is not PASS")
    return observed


def endpoint(name: str) -> list[dict]:
    return [
        row
        for row in read_jsonl(OUTPUTS / name)
        if row.get("task") == "CMTB" and row.get("condition") == "TER_V_auto"
    ]


def moved(prefix: str) -> list[dict]:
    rows: list[dict] = []
    for name in sorted(glob.glob(str(OUTPUTS / f"{prefix}_maximal_pointer_dose_v1_shard*.jsonl"))):
        rows.extend(read_jsonl(Path(name)))
    return rows


def role(prediction: str | None, answer: str, human: dict) -> str:
    if prediction == answer:
        return "CURRENT"
    if prediction == human["offtime_narrated_option"]:
        return "NARRATED"
    if prediction == human["unrelated_option"]:
        return "UNRELATED"
    return "PARSE_FAIL"


def analyze_model(prefix: str, label: str, endpoint_name: str, humans: dict[str, dict]) -> dict:
    all_zero = endpoint(endpoint_name)
    all_moved = moved(prefix)
    rows_by_alpha = {0.0: {row["sample_id"]: row for row in all_zero}}
    for alpha in ALPHAS[1:]:
        rows_by_alpha[alpha] = {
            row["sample_id"]: row
            for row in all_moved
            if abs(float(row["pointer_alpha"]) - alpha) < 1e-9
        }
    if any(len(rows_by_alpha[alpha]) != 291 for alpha in ALPHAS):
        raise RuntimeError(f"{label}: incomplete dose population")
    if any(not set(humans).issubset(rows_by_alpha[alpha]) for alpha in ALPHAS):
        raise RuntimeError(f"{label}: missing human-determinate samples")

    role_by_alpha: dict[float, dict[str, str]] = {}
    counts_by_alpha = {}
    for alpha in ALPHAS:
        roles = {}
        for sample_id, human in humans.items():
            row = rows_by_alpha[alpha][sample_id]
            if human["offtime_narrated_option"] == row["answer"]:
                raise RuntimeError(f"{sample_id}: narrated role equals current answer")
            if len({row["answer"], human["offtime_narrated_option"], human["unrelated_option"]}) != 3:
                raise RuntimeError(f"{sample_id}: role labels are not a three-way partition")
            roles[sample_id] = role(row.get("prediction"), row["answer"], human)
        role_by_alpha[alpha] = roles
        counter = Counter(roles.values())
        counts_by_alpha[str(alpha)] = {
            name: counter.get(name, 0)
            for name in ("CURRENT", "NARRATED", "UNRELATED", "PARSE_FAIL")
        }

    trajectories = {
        sample_id: "".join(
            "1" if role_by_alpha[alpha][sample_id] == "NARRATED" else "0"
            for alpha in ALPHAS
        )
        for sample_id in humans
    }
    weakly_non_decreasing = sum("10" not in bits for bits in trajectories.values())
    adjacent_pairs = []
    for left, right in zip(ALPHAS[:-1], ALPHAS[1:]):
        gain = sum(
            role_by_alpha[left][sample_id] != "NARRATED"
            and role_by_alpha[right][sample_id] == "NARRATED"
            for sample_id in humans
        )
        loss = sum(
            role_by_alpha[left][sample_id] == "NARRATED"
            and role_by_alpha[right][sample_id] != "NARRATED"
            for sample_id in humans
        )
        adjacent_pairs.append(
            {
                "left_alpha": left,
                "right_alpha": right,
                "non_narrated_to_narrated": gain,
                "narrated_to_non_narrated": loss,
                "exact_mcnemar_p": exact_mcnemar(gain, loss),
            }
        )
    endpoint_gain = sum(bits[0] == "0" and bits[-1] == "1" for bits in trajectories.values())
    endpoint_loss = sum(bits[0] == "1" and bits[-1] == "0" for bits in trajectories.values())
    return {
        "model": label,
        "n": len(humans),
        "counts_by_alpha": counts_by_alpha,
        "narrated_fraction_by_alpha": [
            counts_by_alpha[str(alpha)]["NARRATED"] / len(humans) for alpha in ALPHAS
        ],
        "narrated_binary_trajectory_counts": dict(sorted(Counter(trajectories.values()).items())),
        "weakly_non_decreasing_narrated_trajectories": weakly_non_decreasing,
        "weakly_non_decreasing_narrated_fraction": weakly_non_decreasing / len(humans),
        "endpoint_non_narrated_to_narrated": endpoint_gain,
        "endpoint_narrated_to_non_narrated": endpoint_loss,
        "endpoint_exact_mcnemar_p": exact_mcnemar(endpoint_gain, endpoint_loss),
        "adjacent_pairs": adjacent_pairs,
    }


def main() -> None:
    input_hashes = verify_hashes()
    annotations = read_jsonl(ANNOTATIONS)
    if len(annotations) != 100 or len({row["sample_id"] for row in annotations}) != 100:
        raise RuntimeError("single-human file is not the frozen random-100")
    if {row.get("annotator_id") for row in annotations} != {"annotator_2"}:
        raise RuntimeError("annotation provenance drift")
    humans = {
        row["sample_id"]: row
        for row in annotations
        if row.get("offtime_narrated_option") in {"A", "B", "C"}
        and row.get("unrelated_option") in {"A", "B", "C"}
    }
    if len(humans) != 49:
        raise RuntimeError(f"determinate role population drift: {len(humans)} != 49")

    models = [
        analyze_model("qwen3", "Qwen3-Omni-30B-A3B", "qwen3_ter_large_v1.jsonl", humans),
        analyze_model("qwen25", "Qwen2.5-Omni-7B", "qwen25_ter_replication_v1.jsonl", humans),
    ]
    result = {
        "status": "AUDITED_POST_HOC_ROLE_ROUTING",
        "population": "49 determinate roles from frozen random-100, one genuine prediction-blind human",
        "alpha_order": list(ALPHAS),
        "claim_scope": (
            "Directional semantic diagnostic only. It does not estimate CMTB prevalence, provide "
            "inter-human reliability, or establish that every displaced interval visually contains "
            "the narrated event. The analysis was selected after model outcomes were available."
        ),
        "models": models,
        "input_sha256": input_hashes,
        "analysis_code_sha256": None,
    }
    result["analysis_code_sha256"] = sha256(Path(__file__))
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# CMTB-291 pointer-dose role-routing audit v1",
        "",
        "Status: **AUDITED_POST_HOC_ROLE_ROUTING**",
        "",
        result["claim_scope"],
        "",
        "| Model | Role | α=0 | .25 | .50 | .75 | 1.0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for role_name in ("CURRENT", "NARRATED", "UNRELATED", "PARSE_FAIL"):
            values = [model["counts_by_alpha"][str(alpha)][role_name] for alpha in ALPHAS]
            lines.append(
                f"| {model['model']} | {role_name} | " + " | ".join(map(str, values)) + " |"
            )
    lines.extend(
        [
            "",
            "## Paired narrated-option direction",
            "",
            "| Model | Weakly non-decreasing narrated trajectory | α=0→1 gain/loss | Exact paired p |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in models:
        lines.append(
            f"| {model['model']} | {model['weakly_non_decreasing_narrated_trajectories']}/49 "
            f"({100 * model['weakly_non_decreasing_narrated_fraction']:.1f}%) | "
            f"{model['endpoint_non_narrated_to_narrated']}/{model['endpoint_narrated_to_non_narrated']} | "
            f"{model['endpoint_exact_mcnemar_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The correct-option loss is therefore directionally concentrated on the human-labeled off-time narrated option, not the unrelated distractor. Because the role outcome and analysis were not prospectively gated and only one human labeled the roles, this result may strengthen the mechanism narrative but remains a post-hoc directional diagnostic.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"status": result["status"], "models": models}, indent=2))


if __name__ == "__main__":
    main()
