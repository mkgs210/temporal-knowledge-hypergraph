"""Coverage regressions: do not hide a material or let one adjacent paper name a group."""
import json
from copy import deepcopy

import pytest

from test_gemini_labels import fixture
from test_scientific_labels import graph
from tkh.scientific_labels import build_label_input


def test_full_scope_includes_unseen_and_shared_names_but_withholds_claim_sentences(tmp_path):
    from tkh.gemini_labels import prepare_jobs
    run, prompt, _ = fixture(tmp_path)
    view = graph()
    view['nodes'][0]['provenance']['articles'] = [0, 1, 2, 3]
    view['nodes'][0]['surface_form'] = 'Carbon structures'
    view['nodes'][2]['surface_form'] = 'Silicon nitride structures'
    for n in view['nodes']:
        if n['type'] == 'claim':
            n['surface_form'] = f'Private scientific statement {n["id"]}.'
    (run/'snapshots/2022/input.json').write_text(json.dumps(view))
    jobs = prepare_jobs(run, prompt, evidence_policy='full_scope')
    evidence = jobs[0]['evidence']
    assert {n['id'] for n in evidence['members']} == {'n0', 'n2', 'n4', 'n6'}
    assert 'Carbon structures' in jobs[0]['prompt']
    assert 'Silicon nitride structures' in jobs[0]['prompt']
    assert evidence['scope_summary']['visible_nonclaim_members'] == 4
    held = jobs[0]['input']['support']['nodes']
    for n in held:
        if n['type'] == 'claim':
            assert n['surface_form'] not in jobs[0]['prompt']
    assert evidence['scope_summary']['withheld_claim_members'] > 0
    assert all(n['type'] == 'claim' for n in evidence['generation_claims'])


def test_context_is_bounded_per_source_deduplicated_and_separate_from_members():
    from tkh.label_scope import build_scope_evidence
    view = graph()
    inp = build_label_input(view, [n['id'] for n in view['nodes']], 2022)
    source = inp['generation']['source_ids'][0]
    linked = inp['generation']['member_ids'][0]
    inp['generation']['claim_context'] = [
        {'node': {'id': f'context-{i}', 'type': 'claim', 'surface_form': f'Adjacent result {i}.',
                  'observed_at': 2020, 'provenance': {'articles': [source]}},
         'edge_id': f'edge-{i}', 'supporting_member_ids': [linked]}
        for i in range(20)
    ]
    inp['generation']['claim_context'].append(deepcopy(inp['generation']['claim_context'][0]))
    evidence = build_scope_evidence(view, inp, level=0, group_id='test')
    assert len(evidence['claim_context']) == 2
    assert len({c['node']['id'] for c in evidence['claim_context']}) == 2
    assert evidence['context_summary']['available_unique_claims'] == 20
    assert evidence['context_summary']['omitted_unique_claims'] == 18
    assert not {c['node']['id'] for c in evidence['claim_context']} & {n['id'] for n in evidence['members']}


def test_scope_validation_rejects_future_names_even_if_not_in_generation_half(tmp_path):
    from tkh.gemini_labels import prepare_jobs
    run, prompt, _ = fixture(tmp_path)
    view = graph(); view['nodes'][2]['observed_at'] = 2023
    (run/'snapshots/2022/input.json').write_text(json.dumps(view))
    with pytest.raises(ValueError, match='future'):
        prepare_jobs(run, prompt, evidence_policy='full_scope')


def test_selective_repair_generates_only_requested_groups_and_rejects_missing_ids(tmp_path):
    from tkh.gemini_labels import prepare_jobs
    run, prompt, _ = fixture(tmp_path)
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps([{'year': 2022, 'level': 1, 'group_id': 'fine'}]))
    jobs = prepare_jobs(run, prompt, evidence_policy='full_scope', groups_file=selection)
    assert [(j['year'], j['level'], j['id']) for j in jobs] == [(2022, 1, 'fine')]
    selection.write_text(json.dumps([{'year': 2022, 'level': 1, 'group_id': 'missing'}]))
    with pytest.raises(ValueError, match='selection|missing'):
        prepare_jobs(run, prompt, evidence_policy='full_scope', groups_file=selection)


def test_full_scope_preparation_is_offline_and_records_changed_evidence_contract(tmp_path, monkeypatch):
    from tkh.gemini_labels import run_labels
    run, prompt, _ = fixture(tmp_path)
    def forbidden(*a, **kw):
        raise AssertionError('Preparation used an API or credentials')
    monkeypatch.setattr('tkh.gemini_labels.load_api_key', forbidden)
    out = tmp_path/'out'
    run_labels(run, out, prompt_path=prompt, prepare_only=True, evidence_policy='full_scope')
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['evidence_policy_id'] == 'full_scope'
    assert protocol['independence']['scope_names_hidden'] is False
    assert protocol['independence']['support_claim_texts_hidden'] is True
    assert protocol['generation']['maxOutputTokens'] == 512


def test_pilot_cannot_export_a_full_hierarchy_with_unrepaired_old_labels(tmp_path, monkeypatch):
    from tkh.gemini_labels import run_labels
    from test_gemini_labels import reply
    run, prompt, original = fixture(tmp_path)
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps([{'year': 2022, 'level': 1, 'group_id': 'fine'}]))
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-key')
    out = tmp_path/'out'
    result = run_labels(run, out, prompt_path=prompt, evidence_policy='full_scope', groups_file=selection,
                        rpm=100000, send=lambda *args: reply())
    assert result['completed'] == result['total'] == 1
    assert not (out/'snapshots/2022/hierarchy.json').exists()
    assert len(json.loads((out/'snapshots/2022/labels.json').read_text())) == 1
    assert json.loads((run/'snapshots/2022/hierarchy.json').read_text()) == original
