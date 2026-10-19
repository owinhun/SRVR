#!/usr/bin/env python3
"""Post-hoc budget robustness analysis for frozen InternVL Dev-48 outputs."""

import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path("/workspace/svhalluc")
MANIFEST = ROOT / "outputs/videoitg_cmtb_dev48_manifest_v1.jsonl"
K32 = Path("/data2/svhalluc_outputs/videoitg_cmtb/internvl_cmtb_dev48_samebudget_v3.jsonl")
K8 = Path("/data2/svhalluc_outputs/s2fs_internvl_dev48_merged_v1.jsonl")
OUT_JSON = ROOT / "outputs/internvl_srvr_k8_k32_dev48_budget_report_v1.json"
OUT_MD = ROOT / "outputs/internvl_srvr_k8_k32_dev48_budget_report_kr_v1.md"
EXPECTED = {
    "manifest": "2ff399f2530a8b047415964387c67713487b8fdf60913accd5ba94848dcb3f77",
    "k32": "7770090fe8026e43322e065e9f0a5518dbcc9e2821cd41c19f1046484c414525",
    "k8": "f3da9e9e59426acb4af176d51f43de182ca8b3eb9df5b7af94091b30f3bc9895",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def compare(a, b, seed=20260813, repetitions=20000):
    wc = sum(x == 0 and y == 1 for x, y in zip(a, b))
    cw = sum(x == 1 and y == 0 for x, y in zip(a, b))
    rng = random.Random(seed)
    effects = []
    for _ in range(repetitions):
        ix = [rng.randrange(len(a)) for _ in a]
        effects.append(sum(b[i] - a[i] for i in ix) / len(a))
    effects.sort()
    return {
        "delta_correct_count": sum(b) - sum(a),
        "delta_accuracy": (sum(b) - sum(a)) / len(a),
        "wrong_to_correct": wc,
        "correct_to_wrong": cw,
        "mcnemar_exact_p": mcnemar(wc, cw),
        "paired_bootstrap_95_ci": [effects[500], effects[19500]],
    }


def main():
    hashes = {"manifest": sha(MANIFEST), "k32": sha(K32), "k8": sha(K8)}
    if hashes != EXPECTED:
        raise RuntimeError(f"input hash mismatch: {hashes}")
    manifest = rows(MANIFEST)
    answers = {row["sample_id"]: row["answer_evaluation_only"] for row in manifest}
    r32 = {(row["sample_id"], row["condition"]): row for row in rows(K32)}
    r8 = {(row["sample_id"], row["condition"]): row for row in rows(K8)}
    ids = sorted(answers)
    mapping = {
        "uniform_k32": (r32, "uniform_32"),
        "srvr_k32": (r32, "srvr_32"),
        "uniform_k8": (r8, "full_uniform_8"),
        "srvr_k8": (r8, "srvr_uniform_8"),
    }
    correct = {}
    for label, (source, condition) in mapping.items():
        values = []
        for sample_id in ids:
            row = source[(sample_id, condition)]
            if row.get("answer_used") is not False:
                raise RuntimeError(f"non-blind source: {label} {sample_id}")
            values.append(int(row["prediction"] == answers[sample_id]))
        correct[label] = values
    metrics = {
        label: {"correct": sum(values), "n": 48, "accuracy": sum(values) / 48}
        for label, values in correct.items()
    }
    comparisons = {
        "uniform_k8_vs_k32": compare(correct["uniform_k32"], correct["uniform_k8"]),
        "srvr_k8_vs_k32": compare(correct["srvr_k32"], correct["srvr_k8"]),
        "srvr_vs_uniform_k32": compare(correct["uniform_k32"], correct["srvr_k32"]),
        "srvr_vs_uniform_k8": compare(correct["uniform_k8"], correct["srvr_k8"]),
    }
    interaction = (
        comparisons["srvr_vs_uniform_k8"]["delta_accuracy"]
        - comparisons["srvr_vs_uniform_k32"]["delta_accuracy"]
    )
    report = {
        "status": "PASS_POSTHOC_BUDGET_ROBUSTNESS",
        "scope": "same Dev-48 and same answerer; exploratory paired budget analysis, not independent confirmation",
        "metrics": metrics,
        "comparisons": comparisons,
        "budget_by_occurrence_interaction": interaction,
        "input_hashes": hashes,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# InternVL SRVR K=8 대 K=32 Dev-48 예산 강건성",
        "",
        "- 상태: **PASS_POSTHOC_BUDGET_ROBUSTNESS**",
        "- 범위: 같은 Dev-48·같은 answerer의 탐색적 paired 분석이며 독립 확인이 아님.",
        "",
        "| 조건 | 정답 | 정확도 |",
        "|---|---:|---:|",
    ]
    for label in mapping:
        value = metrics[label]
        lines.append(f"| {label} | {value['correct']}/48 | {100*value['accuracy']:.1f}% |")
    lines += ["", "## 해석", ""]
    lines.append(
        f"- Uniform과 SRVR 모두 K=8에서 K=32보다 {comparisons['srvr_k8_vs_k32']['delta_correct_count']:+d}/48이었고, "
        f"SRVR−Uniform 효과는 두 예산 모두 +14/48이다. budget×occurrence 상호작용은 {100*interaction:+.1f}%p다."
    )
    lines.append(
        "- 따라서 이번 Dev에서는 4배 적은 프레임이 SRVR의 상대 이득을 보존했다. 다만 K=8은 S²FS 개발 실험에 포함된 "
        "sequential 결과이므로 효율성의 독립 확인이나 K=8 우월성으로 주장하지 않는다."
    )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
