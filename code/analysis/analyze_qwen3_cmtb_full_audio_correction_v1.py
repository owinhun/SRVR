#!/usr/bin/env python3
"""Recompute manuscript statistics from a complete, lineage-checked correction."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests

ROOT = Path('/workspace/svhalluc')
OUT = ROOT / 'outputs/qwen3_cmtb_full_audio_correction_v1'
sys.path.insert(0, str(ROOT / 'src/analysis'))
import analyze_icassp_v60_reviewer_stats_v1 as stats
import analyze_v116_hybrid_consensus_option_roles_v1 as role
import analyze_cmtb291_prompt_component_factorial_v1 as component
import audit_icassp_v116_localized_factorial_contract_v1 as localized

CELLS = ('P0A0', 'P0A1', 'P1A0', 'P1A1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def write(path, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if Path(path).exists():
        assert Path(path).read_text() == text, f'refuse overwrite {path}'
    else:
        with Path(path).open('x') as f:
            f.write(text)


def evaluate(ids, cells, seed=20260813):
    draws = np.random.default_rng(seed).integers(0, len(ids), (20000, len(ids)))
    result = stats.analyze_factorial(ids, cells, draws)
    result['n'] = len(ids)
    result['cell_correct'] = {c: sum(bool(cells[c][sid]['correct']) for sid in ids) for c in CELLS}
    result['parse_failures'] = {c: sum(cells[c][sid].get('prediction') not in {'A', 'B', 'C'} for sid in ids) for c in CELLS}
    result['headroom_max_interaction_pp'] = 100 * (1 - result['cell_accuracy']['P1A0'] - result['probability_scale']['audio_no_pointer']['delta_accuracy'])
    common = [sid for sid in ids if all(cells[c][sid].get('prediction') in {'A', 'B', 'C'} for c in CELLS)]
    if common:
        cd = np.random.default_rng(seed).integers(0, len(common), (20000, len(common)))
        result['common_valid'] = {'n': len(common), **stats.analyze_factorial(common, cells, cd)}
    else:
        result['common_valid'] = {'n': 0, 'status': 'unavailable_no_common_valid_sources'}
    return result


def family(models, nine=False):
    tests = {}
    for name, value in models.items():
        if nine:
            for term in ('audio_no_pointer', 'audio_correct_pointer'):
                tests[f'{name}:{term}'] = value['probability_scale'][term]['mcnemar_exact_p']
        tests[f'{name}:logit_interaction'] = value['logit_scale_interaction']['gee_p']
    adjusted = multipletests(list(tests.values()), method='holm')[1]
    return {name: {'raw_p': float(p), 'holm_p': float(q)} for (name, p), q in zip(tests.items(), adjusted)}


def main():
    freeze = read(OUT / 'freeze.json')
    for path, digest in freeze['protected_sha256'].items():
        assert sha(path) == digest, path
    paths = [OUT / f'shard{s}of2_predictions.jsonl' for s in range(2)]
    for shard, path in enumerate(paths):
        complete = read(OUT / f'shard{shard}of2_complete.json')
        assert complete['predictions_sha256'] == sha(path)
        assert complete['freeze_sha256'] == sha(OUT / 'freeze.json')
        assert complete['n'] == len(rows(path))
    resume = read(OUT / 'parallel_resume_record.json')
    assert sha(Path(resume['initial_prefix'])) == resume['initial_prefix_sha256']
    corrected = [r for p in paths for r in rows(p)]
    ids = [r['sample_id'] for r in rows(ROOT / 'data/ter_large_eval_v1.jsonl') if r['task'] == 'CMTB']
    expected = {(sid, c) for sid in ids for c in freeze['cells']}
    assert len(corrected) == len(expected) == 1164
    assert {(r['sample_id'], r['condition']) for r in corrected} == expected
    current_by_key = {(r['sample_id'], r['condition']): r for r in corrected}
    for r in rows(Path(resume['initial_prefix'])):
        assert current_by_key[r['sample_id'], r['condition']] == r
    assert all(r['correction']['freeze_sha256'] == sha(OUT / 'freeze.json') for r in corrected)
    audits = Counter()
    for r in corrected:
        c = r['correction']
        same = c['legacy_input_signature'] == c['corrected_input_signature']
        assert same == (c['status'] == 'REUSED_EQUAL_RECONSTRUCTED_INPUT')
        assert c['visual_tensors_equal']
        a = c['audio_support']
        assert abs(a['valid_mel_frames'] - int(np.ceil(a['samples'] / 160.))) <= 1
        assert r['correct'] == (r.get('prediction') == r['answer'])
        audits[c['status']] += 1
    o = ROOT / 'outputs'
    q3old = rows(o / 'qwen3_ter_large_v1.jsonl')
    q25 = rows(o / 'qwen25_ter_replication_v1.jsonl')
    q25on = [r for p in sorted((o / 'v45_audio').glob('qwen25_ter_av_auto_shard*.jsonl')) for r in rows(p)]
    chronus = [r for p in sorted(o.glob('chronusomni_cmtb291_factorial_v1_shard*.jsonl')) for r in rows(p)]
    select = stats.select
    main_cells = {
        'Qwen2.5-Omni': dict(zip(CELLS, [select(q25, 'V'), select(q25, 'O_raw'), select(q25, 'TER_V_auto'), select(q25on, 'TER_AV_auto')])),
        'Qwen3-Omni': dict(zip(CELLS, [select(q3old, 'V'), select(corrected, 'O_raw'), select(q3old, 'TER_V_auto'), select(corrected, 'TER_AV_auto')])),
        'ChronusOmni': dict(zip(CELLS, [select(chronus, c) for c in ('no_pointer_audio_off', 'no_pointer_audio_on', 'correct_pointer_audio_off', 'correct_pointer_audio_on')]))}
    models = {name: evaluate(ids, cells) for name, cells in main_cells.items()}
    historical_six_path = o / 'icassp_v90_table1_six_interaction_holm_v1.json'
    historical_six = read(historical_six_path)
    six_p = {name: models[name]['logit_scale_interaction']['gee_p'] if name in models else r['gee_raw_p']
             for name, r in historical_six['models'].items()}
    six_q = multipletests(list(six_p.values()), method='holm')[1]
    old_cells = dict(zip(CELLS, [select(q3old, c) for c in ('V', 'O_raw', 'TER_V_auto', 'TER_AV_auto')]))
    old_result = evaluate(ids, old_cells)
    assert list(old_result['cell_correct'].values()) == [137, 177, 262, 252]
    assert list(models['Qwen2.5-Omni']['cell_correct'].values()) == [136, 154, 216, 172]
    assert list(models['ChronusOmni']['cell_correct'].values()) == [137, 199, 252, 235]
    bare_models = {}
    for key, name in (('qwen25', 'Qwen2.5-Omni'), ('qwen3', 'Qwen3-Omni')):
        comp, _ = component.load_model(key)
        cells = {'P0A0': main_cells[name]['P0A0'], 'P0A1': main_cells[name]['P0A1'],
                 'P1A0': comp['timestamp_only']['off'],
                 'P1A1': select(corrected, 'timestamp_only_audio_on') if key == 'qwen3' else comp['timestamp_only']['on']}
        bare_models[name] = evaluate(ids, cells, 20260814)
    localized_models = {}
    for name, source in localized.SOURCES.items():
        cells = {'P0A0': localized.load(source['base'], 'V', rename=True),
                 'P0A1': localized.load(source['base'], 'O_raw', rename=True),
                 'P1A0': localized.load(source['localized_off'], 'V_VADclip'),
                 'P1A1': localized.load(source['localized_on'], 'AV_VADclip')}
        if name == 'Qwen3-Omni':
            cells.update(P0A1=select(corrected, 'O_raw'), P1A1=select(corrected, 'AV_VADclip'))
        localized_models[name] = evaluate(ids, cells, 20260818)
    annotations = {r['sample_id']: r for r in rows(role.HYBRID_OUT)}
    role_results = role.model_result(sorted(ids), main_cells['Qwen3-Omni']['P1A0'], main_cells['Qwen3-Omni']['P1A1'], annotations, 0)
    old_roles = role.model_result(sorted(ids), old_cells['P1A0'], old_cells['P1A1'], annotations, 0)
    by_condition = {}
    old_sources = {c: select(rows(Path(next(r['correction']['source_predictions'] for r in corrected if r['condition'] == c))), c) for c in freeze['cells']}
    for c in freeze['cells']:
        rr = [r for r in corrected if r['condition'] == c]
        by_condition[c] = {'status_counts': dict(Counter(r['correction']['status'] for r in rr)),
            'old_correct': sum(bool(r['correct']) for r in old_sources[c].values()),
            'corrected_correct': sum(bool(r['correct']) for r in rr),
            'prediction_changed_n': sum(r['prediction'] != old_sources[c][r['sample_id']]['prediction'] for r in rr)}
    result = {'status': 'COMPLETE_LINEAGE_AND_INPUT_CHECKS_PASS', 'n': 291,
              'input_audit': dict(audits), 'correction_by_condition': by_condition,
              'models': models, 'central_nine_test_family': family(models, nine=True),
              'six_table1_interaction_sensitivity': {name: {'raw_p': float(p), 'holm_six_p': float(q)} for (name, p), q in zip(six_p.items(), six_q)},
              'legacy_qwen3': old_result, 'qwen3_roles': role_results, 'legacy_qwen3_roles': old_roles,
              'bare_models': bare_models, 'bare_two_test_family': family(bare_models),
              'localized_models': localized_models, 'localized_four_test_family': family(localized_models),
              'example_cmtb_001423': {c: select(corrected, c)['cmtb_001423']['prediction'] for c in freeze['cells']},
              'provenance': {'freeze_sha256': sha(OUT / 'freeze.json'), 'analyzer_sha256': sha(__file__),
                             'raw_sha256': {str(p): sha(p) for p in paths}, 'role_map_sha256': sha(role.HYBRID_OUT),
                             'historical_unchanged_scope_statistics_sha256': sha(historical_six_path)},
              'limits': ['Retrospective input correction, not independent replication.',
                         'Reuse is based on current reconstruction and recorded historical inputs; old tensors were not retained.',
                         'Bare-prompt historical model_revision was local_snapshot; same hard-coded path and download metadata, weaker provenance.',
                         'Role map sampled from historical errors; new unmapped errors remain unresolved.']}
    write(OUT / 'results.json', result)
    lines = ['# Qwen3 CMTB full-audio correction', '', 'All 291 locked sources retained; no outcome selection.', '',
             '| Cell | Old correct | Corrected correct | Rerun | Input-identical reuse |', '|---|---:|---:|---:|---:|']
    for c, r in by_condition.items():
        lines.append(f"| {c} | {r['old_correct']} | {r['corrected_correct']} | {r['status_counts'].get('RERUN_FULL_AUDIO', 0)} | {r['status_counts'].get('REUSED_EQUAL_RECONSTRUCTED_INPUT', 0)} |")
    lines += ['', '| Model | Audio without reference | Audio with reference | Interaction [95% CI] | GEE Holm p (9 tests) |', '|---|---:|---:|---|---:|']
    for name, r in models.items():
        s = r['probability_scale']; d = s['interaction']; ci = d['bootstrap_95ci']
        p = result['central_nine_test_family'][f'{name}:logit_interaction']['holm_p']
        lines.append(f"| {name} | {s['audio_no_pointer']['delta_accuracy']*100:+.1f} pp | {s['audio_correct_pointer']['delta_accuracy']*100:+.1f} pp | {d['estimate']*100:+.1f} [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] pp | {p:.6g} |")
    lines += ['', 'See results.json for all bare/localized families, role counts, common-valid sensitivity and provenance.', '', *['- ' + x for x in result['limits']], '',
              'Scientific Critical Thinking informed this correction and interpretation. Procedural reference:',
              'Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026).',
              '[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065).', '']
    report = '\n'.join(lines)
    dest = OUT / 'results.md'
    if dest.exists():
        assert dest.read_text() == report
    else:
        with dest.open('x') as f:
            f.write(report)
    print(report)


if __name__ == '__main__':
    main()
