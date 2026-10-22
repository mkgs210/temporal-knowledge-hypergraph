"""Offline topic naming: actual member examples, no withheld claim leakage."""
import json
from copy import deepcopy

from test_gemini_labels import fixture, reply


def member(node_id, text, kind='method'):
    return {'id': node_id, 'type': kind, 'surface_form': text, 'source_ids': [0]}


def test_topic_keywords_and_four_examples_come_from_members_not_adjacent_claims():
    from tkh.label_scope import topic_representation
    evidence = {'members': [
        member('a', 'Atomic force prediction'),
        member('b', 'Atomic energy prediction'),
        member('c', 'Atomic stress prediction'),
        member('d', 'Atomic structure prediction'),
        member('e', 'Silicon nitride materials'),
        member('f', 'Atomic force prediction')],
        'generation_claims': [],
        'claim_context': [{'node': member('adjacent', 'Unrelated secret medical trial', 'claim')}]}
    result = topic_representation(evidence)
    assert result['keywords']
    assert len(result['documents']) == 4
    assert all(d in evidence['members'] for d in result['documents'])
    assert len({d['surface_form'] for d in result['documents']}) == 4
    assert not any('medical' in word or 'secret' in word for word in result['keywords'])
    assert result['members_used'] == 6
    # Selecting central examples must not remove the minority from the full scope.
    assert evidence['members'][4]['surface_form'] == 'Silicon nitride materials'
    reversed_evidence = deepcopy(evidence)
    reversed_evidence['members'].reverse()
    assert topic_representation(reversed_evidence) == result


def test_claim_only_group_has_examples_but_no_invented_keywords_for_empty_vocabulary():
    from tkh.label_scope import topic_representation
    evidence = {'members': [], 'generation_claims': [member('c', 'The and or.', 'claim')]}
    result = topic_representation(evidence)
    assert result['keywords'] == []
    assert result['documents'] == evidence['generation_claims']


def test_four_examples_do_not_all_repeat_one_subtopic():
    from tkh.label_scope import topic_representation
    evidence = {'members': [
        member('a', 'Cross validation'),
        member('b', 'Five fold cross validation'),
        member('c', 'Repeated cross validation'),
        member('d', 'Nested cross validation'),
        member('e', 'Cross validation evaluation'),
        member('f', 'Gradient optimization'),
        member('g', 'Training data sampling')], 'generation_claims': []}
    examples = topic_representation(evidence)['documents']
    assert len(examples) == 4
    assert any('validation' not in d['surface_form'] for d in examples)


def test_topic_prompt_preserves_all_names_and_withholds_support_claims(tmp_path):
    from tkh.gemini_labels import prepare_jobs
    run, prompt, _ = fixture(tmp_path)
    jobs = prepare_jobs(run, prompt, evidence_policy='full_scope')
    for job in jobs:
        evidence = job['evidence']
        assert evidence['topic_representation']['keywords']
        assert {n['id'] for n in evidence['members']} == {'n0', 'n2', 'n4', 'n6'}
        for node in job['input']['support']['nodes']:
            if node['type'] == 'claim':
                assert node['id'] not in {d['id'] for d in evidence['topic_representation']['documents']}


def test_longer_label_and_explanatory_sentence_survive_generation_and_resume(tmp_path, monkeypatch):
    from tkh.gemini_labels import run_labels
    run, prompt, _ = fixture(tmp_path)
    monkeypatch.setenv('GEMINI_API_KEY', 'test-key')
    label = {'label': 'Graph neural network methods for predicting properties of atomic materials',
             'gloss': 'These methods represent atomic environments as graphs to predict energies, '
                      'forces and other material properties using learned relationships between atoms.',
             'status': 'coherent'}
    out = tmp_path/'out'
    result = run_labels(run, out, prompt_path=prompt, evidence_policy='full_scope',
                        length_policy='strict', rpm=100000, sleep=lambda _: None,
                        send=lambda *a: reply(json.dumps(label)))
    assert result['completed'] == 2
    assert result['rejected_this_invocation'] == 0
    rows = json.loads((out/'snapshots/2022/labels.json').read_text())
    assert all(r['gloss'] == label['gloss'] and r['label'] == label['label'] for r in rows)
    assert all(not r['length_warnings'] for r in rows)
    protocol = json.loads((out/'protocol.json').read_text())
    assert protocol['requested_style']['gloss_words'][1] >= len(label['gloss'].split())
    assert protocol['generation']['maxOutputTokens'] == 512
    def forbidden(*args):
        raise AssertionError('A completed label was generated again')
    assert run_labels(run, out, prompt_path=prompt, evidence_policy='full_scope',
                      length_policy='strict', send=forbidden)['completed'] == 2
