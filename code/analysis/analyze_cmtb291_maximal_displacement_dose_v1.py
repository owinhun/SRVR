#!/usr/bin/env python3
"""Fail-closed analysis of the full-N maximal-displacement pointer dose."""

from __future__ import annotations

import glob
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/svhalluc")
ALPHAS = (0.0, .25, .50, .75, 1.0)
BOOTSTRAP = 20_000
SEED = 20260802
POINTER = ROOT / "outputs/cmtb291_maximal_displacement_pointer_v1.jsonl"
PROTOCOL = ROOT / "paper/cmtb291_maximal_displacement_dose_protocol_v1.md"
EVAL = ROOT / "data/ter_large_eval_v1.jsonl"
POINTER_HASH = "b4c253de44557cd944bf0091efaca5c15030863fddb9f9102b3b557b9a5a679a"
PROTOCOL_HASH = "d8b0236cace2c43bf8ee3397710e6a4ef56ed952a1fb348ad60f7e96a10821b0"
EVAL_HASH = "9221a8af1f67e9da22bedbb2bca9c68333ec6a38c5f614219c05376b2b451f9e"
MODEL_CONTRACTS = {
    "qwen3": {
        "model_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
    },
    "qwen25": {
        "model_id": "Qwen/Qwen2.5-Omni-7B",
        "revision": "ae9e1690543ffd5c0221dc27f79834d0294cba00",
    },
}
OUT = ROOT / "outputs/cmtb291_maximal_displacement_dose_analysis_v1.json"
REPORT = ROOT / "outputs/cmtb291_maximal_displacement_dose_report_v1.md"
TABLE = ROOT / "paper/tables/maximal_pointer_dose_full291_v1.tex"


def read(path: Path) -> list[dict]: return [json.loads(line) for line in path.read_text().splitlines() if line]
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_prompt(question: str, start: float, end: float) -> str:
    return (
        f"The relevant speech activity occurs from {start:.2f} to {end:.2f} seconds. "
        "Answer using only the visual action during that interval. "
        "Do not infer or use transcript content.\n\n" + question
    )


def exact_pair(left: list[dict], right: list[dict], seed_offset: int) -> dict:
    one={row["sample_id"]:bool(row["correct"]) for row in left};two={row["sample_id"]:bool(row["correct"]) for row in right}
    if set(one)!=set(two):raise RuntimeError("paired IDs mismatch")
    ids=sorted(one);left_only=sum(one[sid] and not two[sid] for sid in ids);right_only=sum(two[sid] and not one[sid] for sid in ids);discordant=left_only+right_only
    p=1.0 if not discordant else min(1.0,2*sum(math.comb(discordant,k) for k in range(min(left_only,right_only)+1))/(2**discordant))
    delta=np.array([int(one[sid])-int(two[sid]) for sid in ids],dtype=float);rng=np.random.default_rng(SEED+seed_offset)
    draws=delta[rng.integers(0,len(ids),size=(BOOTSTRAP,len(ids)))].mean(axis=1)
    return {"n":len(ids),"left_accuracy":float(np.mean(list(one.values()))),"right_accuracy":float(np.mean(list(two.values()))),"left_minus_right":float(delta.mean()),"left_only_correct":left_only,"right_only_correct":right_only,"exact_mcnemar_p":p,"bootstrap_95ci":[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]}


def endpoint(model: str) -> list[dict]:
    if model.startswith("Qwen3"):
        rows=read(ROOT/"outputs/qwen3_ter_large_v1.jsonl")
    else:
        rows=read(ROOT/"outputs/qwen25_ter_replication_v1.jsonl")
    return [row for row in rows if row.get("task")=="CMTB" and row.get("condition")=="TER_V_auto"]


def new_rows(prefix: str) -> list[dict]:
    result=[]
    for name in sorted(glob.glob(str(ROOT/f"outputs/{prefix}_maximal_pointer_dose_v1_shard*.jsonl"))):result.extend(read(Path(name)))
    return result


def validate_new(prefix: str, rows: list[dict], pointers: dict[str,dict], evaluation: dict[str,dict], runner: Path) -> dict:
    expected=291*4; keys={(row["sample_id"],float(row["pointer_alpha"])) for row in rows};errors=[]
    if len(rows)!=expected or len(keys)!=expected:errors.append(f"row_or_key_count:{len(rows)}/{len(keys)}")
    if {float(row["pointer_alpha"]) for row in rows}!={.25,.5,.75,1.0}:errors.append("alpha_set")
    runner_hash=sha(runner)
    contract=MODEL_CONTRACTS[prefix]
    for row in rows:
        sid=row.get("sample_id");pointer=pointers.get(sid);item=evaluation.get(sid);alpha=float(row.get("pointer_alpha",-1))
        if pointer is None or item is None:errors.append(f"unknown:{sid}");continue
        expected_start=(1-alpha)*pointer["speech_start_sec"]+alpha*pointer["maximal_start_sec"]
        expected_end=(1-alpha)*pointer["speech_end_sec"]+alpha*pointer["maximal_end_sec"]
        prompt=expected_prompt(item["question"],expected_start,expected_end)
        expected_condition=f"MaxDose_alpha{int(round(100*alpha)):03d}"
        if max(abs(row["used_interval_sec"][0]-expected_start),abs(row["used_interval_sec"][1]-expected_end))>1e-8:errors.append(f"interval:{sid}:{alpha}")
        if row.get("condition")!=expected_condition or row.get("task")!="CMTB":errors.append(f"condition:{sid}:{alpha}")
        if row.get("video_id")!=pointer["video_id"] or row.get("answer")!=item["answer"]:errors.append(f"item:{sid}:{alpha}")
        if row.get("prompt")!=prompt or row.get("prompt_sha256")!=hashlib.sha256(prompt.encode()).hexdigest():errors.append(f"prompt:{sid}:{alpha}")
        if row.get("pointer_manifest_sha256")!=POINTER_HASH or row.get("protocol_sha256")!=PROTOCOL_HASH or row.get("manifest_sha256")!=EVAL_HASH or row.get("code_sha256")!=runner_hash:errors.append(f"hash:{sid}:{alpha}")
        if row.get("model_id")!=contract["model_id"] or row.get("model_revision")!=contract["revision"] or row.get("processor_revision")!=contract["revision"]:errors.append(f"model:{sid}:{alpha}")
        if row.get("audio_handling",{}).get("use_audio_in_video") is not False:errors.append(f"audio:{sid}:{alpha}")
        if row.get("decoding",{}).get("do_sample") is not False:errors.append(f"decode:{sid}:{alpha}")
        sampling=row.get("video_sampling",{})
        if sampling.get("requested_fps")!=2.0 or sampling.get("max_frames")!=768:errors.append(f"sampling:{sid}:{alpha}")
        if row.get("strict_zero_overlap")!=pointer["strict_zero_overlap"] or abs(row.get("maximal_overlap_iou",-1)-pointer["overlap_iou"])>1e-12:errors.append(f"geometry:{sid}:{alpha}")
    return {"expected_rows":expected,"observed_rows":len(rows),"unique_keys":len(keys),"runner_sha256":runner_hash,"errors":errors,"pass":not errors}


def analyze_model(model: str,prefix: str,pointers:dict[str,dict],evaluation:dict[str,dict],runner:Path)->dict:
    zero=endpoint(model);new=new_rows(prefix);validation=validate_new(prefix,new,pointers,evaluation,runner)
    if not validation["pass"]:raise RuntimeError(f"{model} validation failed: {validation['errors'][:10]}")
    if len(zero)!=291 or len({row["sample_id"] for row in zero})!=291:raise RuntimeError(f"{model} endpoint incomplete")
    endpoint_errors=[];contract=MODEL_CONTRACTS[prefix]
    for row in zero:
        sid=row["sample_id"];pointer=pointers.get(sid);item=evaluation.get(sid)
        if pointer is None or item is None:endpoint_errors.append(f"unknown:{sid}");continue
        prompt=expected_prompt(item["question"],pointer["speech_start_sec"],pointer["speech_end_sec"])
        if row.get("prompt")!=prompt or row.get("prompt_sha256")!=hashlib.sha256(prompt.encode()).hexdigest():endpoint_errors.append(f"prompt:{sid}")
        if row.get("answer")!=item["answer"] or row.get("task")!="CMTB" or row.get("condition")!="TER_V_auto":endpoint_errors.append(f"item:{sid}")
        if row.get("manifest_sha256")!=EVAL_HASH or row.get("model_id")!=contract["model_id"] or row.get("model_revision")!=contract["revision"] or row.get("processor_revision")!=contract["revision"]:endpoint_errors.append(f"contract:{sid}")
        if abs(row.get("timestamp_start_sec",-1)-pointer["speech_start_sec"])>1e-12 or abs(row.get("timestamp_end_sec",-1)-pointer["speech_end_sec"])>1e-12:endpoint_errors.append(f"interval:{sid}")
        sampling=row.get("video_sampling",{})
        if sampling.get("requested_fps")!=2.0 or sampling.get("max_frames")!=768 or row.get("decoding",{}).get("do_sample") is not False:endpoint_errors.append(f"sampling_decode:{sid}")
    if endpoint_errors:raise RuntimeError(f"{model} frozen endpoint contract failed: {endpoint_errors[:10]}")
    validation["frozen_alpha0_contract"]={"observed_rows":len(zero),"errors":endpoint_errors,"pass":True}
    by={0.0:zero}
    for alpha in (.25,.5,.75,1.0):by[alpha]=[row for row in new if abs(float(row["pointer_alpha"])-alpha)<1e-9]
    populations={"all":set(pointers),"strict_zero_overlap":{sid for sid,row in pointers.items() if row["strict_zero_overlap"]},"positive_overlap":{sid for sid,row in pointers.items() if not row["strict_zero_overlap"]}}
    summaries={}
    for index,(name,ids) in enumerate(populations.items()):
        per={};accuracies=[];zero_predictions={row["sample_id"]:row["prediction"] for row in by[0.0] if row["sample_id"] in ids}
        for alpha in ALPHAS:
            subset=[row for row in by[alpha] if row["sample_id"] in ids];accuracy=float(np.mean([bool(row["correct"]) for row in subset]));accuracies.append(accuracy)
            per[str(alpha)]={"n":len(subset),"correct":sum(bool(row["correct"]) for row in subset),"accuracy":accuracy,"parse_failures":sum(row.get("parse_status")!="parsed" for row in subset),"prediction_agreement_with_alpha0":float(np.mean([row["prediction"]==zero_predictions[row["sample_id"]] for row in subset]))}
        summaries[name]={"alpha":per,"accuracy_sequence":accuracies,"non_increasing":all(accuracies[i]>=accuracies[i+1] for i in range(4)),"alpha025_vs_alpha075":exact_pair([row for row in by[.25] if row["sample_id"] in ids],[row for row in by[.75] if row["sample_id"] in ids],index),"alpha0_vs_alpha1":exact_pair([row for row in by[0] if row["sample_id"] in ids],[row for row in by[1] if row["sample_id"] in ids],100+index)}
    primary=summaries["all"]
    gates={"all291_accuracy_non_increasing":primary["non_increasing"],"all291_alpha025_minus_alpha075_at_least_15pp":primary["alpha025_vs_alpha075"]["left_minus_right"]>=.15,"all291_alpha025_vs_alpha075_mcnemar_below_005":primary["alpha025_vs_alpha075"]["exact_mcnemar_p"]<.05,"all291_alpha0_minus_alpha1_at_least_20pp":primary["alpha0_vs_alpha1"]["left_minus_right"]>=.20}
    gates["model_gate_pass"]=all(gates.values())
    return {"model":model,"validation":validation,"populations":summaries,"gates":gates}


def main()->None:
    if sha(POINTER)!=POINTER_HASH or sha(PROTOCOL)!=PROTOCOL_HASH or sha(EVAL)!=EVAL_HASH:raise RuntimeError("frozen pointer/protocol/evaluation drift")
    pointer_rows=read(POINTER);pointers={row["sample_id"]:row for row in pointer_rows}
    if len(pointers)!=291:raise RuntimeError("pointer population mismatch")
    evaluation={row["sample_id"]:row for row in read(EVAL) if row.get("task")=="CMTB"}
    if len(evaluation)!=291 or set(evaluation)!=set(pointers):raise RuntimeError("evaluation population mismatch")
    models=[analyze_model("Qwen3-Omni-30B-A3B","qwen3",pointers,evaluation,ROOT/"src/qwen/run_qwen3_maximal_pointer_dose_v1.py"),analyze_model("Qwen2.5-Omni-7B","qwen25",pointers,evaluation,ROOT/"src/qwen/run_qwen25_maximal_pointer_dose_v1.py")]
    status="PASS_PROMOTE_FULL_N_MAXIMAL_DOSE" if all(model["gates"]["model_gate_pass"] for model in models) else "FAIL_OR_MIXED_DO_NOT_PROMOTE"
    result={"status":status,"protocol":str(PROTOCOL.relative_to(ROOT)),"prospective_boundary":"alpha 0 reused frozen output; alpha .25/.50/.75/1 were unseen at freeze","sample_counts":{"all":291,"strict_zero_overlap":194,"positive_overlap":97},"models":models,"bootstrap":{"repetitions":BOOTSTRAP,"seed":SEED},"input_sha256":{"pointer":POINTER_HASH,"protocol":PROTOCOL_HASH,"evaluation":EVAL_HASH},"claim_boundary":"Full-N maximal displacement reduces strict-subset dependence but does not turn the 97 low-overlap rows into strict zero-overlap controls or add a second dataset."}
    OUT.write_text(json.dumps(result,indent=2)+"\n")
    lines=["# CMTB-291 full-N maximal-displacement dose report v1","",f"Decision: **{status}**","","Alpha 0 was frozen previously; alpha .25/.50/.75/1 were prospective under the frozen protocol.",""]
    table=["\\begin{tabular}{lrrrrr}","\\toprule","Model & $0$ & $.25$ & $.50$ & $.75$ & $1$ \\\\","\\midrule"]
    for model in models:
        sequence=model["populations"]["all"]["accuracy_sequence"]
        lines += [f"## {model['model']}","","| Alpha | All N=291 | Strict N=194 | Overlap N=97 |","|---:|---:|---:|---:|"]
        for alpha in ALPHAS:
            values=[model["populations"][population]["alpha"][str(alpha)]["accuracy"] for population in ("all","strict_zero_overlap","positive_overlap")]
            lines.append(f"| {alpha:.2f} | {100*values[0]:.1f}% | {100*values[1]:.1f}% | {100*values[2]:.1f}% |")
        pair=model["populations"]["all"]["alpha025_vs_alpha075"]
        lines += ["",f"All-291 alpha .25 - .75: {100*pair['left_minus_right']:+.1f} pp, 95% CI [{100*pair['bootstrap_95ci'][0]:.1f}, {100*pair['bootstrap_95ci'][1]:.1f}], p={pair['exact_mcnemar_p']:.4g}.","",f"Gates: `{model['gates']}`",""]
        label="Qwen3" if model["model"].startswith("Qwen3") else "Qwen2.5"
        table.append(label+" & "+" & ".join(f"{100*x:.1f}" for x in sequence)+r" \\")
    lines += ["## Claim boundary","",result["claim_boundary"],""]
    REPORT.write_text("\n".join(lines));TABLE.parent.mkdir(parents=True,exist_ok=True);TABLE.write_text("\n".join(table+["\\bottomrule","\\end{tabular}"])+"\n")
    print(json.dumps({"status":status,"models":[{"model":model["model"],"sequence":model["populations"]["all"]["accuracy_sequence"],"gates":model["gates"]} for model in models]},indent=2))


if __name__=="__main__":main()
