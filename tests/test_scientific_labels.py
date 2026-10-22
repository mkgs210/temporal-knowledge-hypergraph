"""Contracts for evidence-disjoint scientific label generation and checking."""
import json
import pytest
from tkh.scientific_labels import (
    build_label_input, parse_label_output, calibrate_checker, assess_support, summarize_support, support_documents,
)


def graph():
    return {'cutoff': 2022, 'nodes': [
        {'id': f'n{i}', 'type': 'claim' if i % 2 else 'method', 'surface_form': f'Graph model {i}',
         'observed_at': 2020, 'provenance': {'articles': [i // 2]}}
        for i in range(8)
    ], 'hyperedges': [
        {'id': f'e{i}', 'members': [f'n{2*i}', f'n{2*i+1}'], 'relation_type': 'claims',
         'year': 2020, 'provenance': {'article_id': i, 'article_year': 2020}}
        for i in range(4)
    ]}


def test_split_holds_out_entire_sources_and_records_exact_prompt():
    x = build_label_input(graph(), [f'n{i}' for i in range(8)], 2022, seed=13)
    a, b = x['generation'], x['support']
    assert set(a['source_ids']).isdisjoint(b['source_ids'])
    assert set(a['member_ids']).isdisjoint(b['member_ids'])
    assert len(a['source_ids']) == len(b['source_ids']) == 2
    assert set(a['member_ids'] + b['member_ids']) == {f'n{i}' for i in range(8)}
    for n in b['nodes']:
        assert n['id'] not in x['prompt']
    assert all(e['provenance']['article_id'] in a['source_ids'] for e in a['hyperedges'])
    assert x == build_label_input(graph(), list(reversed([f'n{i}' for i in range(8)])), 2022, seed=13)


def test_shared_source_member_cannot_leak_across_partitions():
    g = graph()
    g['nodes'][0]['provenance']['articles'] = [0, 1, 2, 3]
    x = build_label_input(g, [f'n{i}' for i in range(8)], 2022)
    assert 'n0' in x['excluded_shared_member_ids']
    assert 'n0' not in x['generation']['member_ids'] + x['support']['member_ids']


def test_rejects_future_or_unknown_member_instead_of_filling_from_raw_graph():
    g = graph()
    g['nodes'][0]['observed_at'] = 2023
    with pytest.raises(ValueError, match='future'):
        build_label_input(g, ['n0'], 2022)
    with pytest.raises(ValueError, match='unknown'):
        build_label_input(graph(), ['missing'], 2022)


def test_single_source_has_no_independent_support_evidence():
    x = build_label_input(graph(), ['n0', 'n1'], 2022)
    assert x['support']['member_ids'] == []
    assert x['support_available'] is False


def test_parser_accepts_one_sentence_and_preserves_uncertainty():
    raw = json.dumps({'label': 'Atomistic graph learning', 'gloss': 'Graph models represent atomic environments for property prediction.', 'status': 'mixed'})
    x = parse_label_output(raw)
    assert x['status'] == 'mixed'
    assert x['gloss'].startswith('Graph models')


@pytest.mark.parametrize('raw', [
    'not json', '{"label":"X"}',
    '{"label":"Methods","gloss":"Representative member names: A; B.","status":"coherent"}',
    '{"label":"Theme","gloss":"Various scientific methods and related topics.","status":"coherent"}',
    '{"label":"Theme","gloss":"First sentence. Second sentence.","status":"coherent"}',
])
def test_parser_does_not_turn_malformed_or_vacuous_output_into_pass(raw):
    with pytest.raises(ValueError):
        parse_label_output(raw)


def test_checker_calibration_fails_when_it_accepts_incompatible_controls():
    x = calibrate_checker([.95, .9, .94], [.96, .91, .93])
    assert x['usable'] is False
    assert assess_support('A scientific explanatory sentence.', [.98], x)['status'] == 'insufficient_evidence'


def test_supported_unsupported_and_unknown_are_not_conflated():
    c = calibrate_checker([.94, .96, .98], [.01, .03, .06])
    assert c['usable'] is True
    a = assess_support('Graph models predict atomic properties.', [.95, .94], c)
    b = assess_support('Graph models predict atomic properties.', [.01, .02], c)
    u = assess_support('Graph models predict atomic properties.', [], c)
    assert [a['status'], b['status'], u['status']] == ['supported', 'unsupported', 'insufficient_evidence']
    s = summarize_support([a, b, u])
    assert s['unsupported_fraction_all'] == pytest.approx(1/3)
    assert s['unsupported_fraction_assessable'] == .5
    assert s['insufficient_fraction'] == pytest.approx(1/3)
    assert s['is_human_verified_overclaim_rate'] is False


def test_ambiguous_support_is_insufficient_not_zero_overclaim():
    c = calibrate_checker([.94, .96], [.01, .03])
    x = assess_support('Graph models predict atomic properties.', [.5], c)
    assert x['status'] == 'insufficient_evidence'
    s = summarize_support([x])
    assert s['unsupported_fraction_assessable'] is None


def test_prompt_caps_members_and_reports_unseen_generation_members():
    x = build_label_input(graph(), [f'n{i}' for i in range(8)], 2022, max_generation_members=2)
    assert len(x['prompt_member_ids']) == 2
    assert set(x['prompt_member_ids']) <= set(x['generation']['member_ids'])
    assert x['generation_members_omitted_from_prompt'] == 2
    prompt_payload = json.loads(x['prompt'].split('EVIDENCE:\n')[1])
    assert len(prompt_payload['nodes']) == 2


def test_heldout_documents_use_claim_sentences_not_surface_name_list():
    from tkh.scientific_labels import support_documents
    x = build_label_input(graph(), [f'n{i}' for i in range(8)], 2022)
    docs = support_documents(x)
    assert len(docs) == 2
    assert all(d['claim_node_ids'] and d['text'].startswith('Graph model') for d in docs)
    assert not set(sum([d['claim_node_ids'] for d in docs], [])) & set(x['generation']['member_ids'])
    g = graph()
    g['hyperedges'] = []
    y = build_label_input(g, ['n0', 'n2', 'n4', 'n6'], 2022)
    assert support_documents(y) == []


def test_corrupt_label_control_cannot_reuse_same_source_evidence():
    from tkh.scientific_labels import cross_group_controls
    records = [
        {'id': 'a', 'level': 0, 'label': {'gloss': 'Atoms form crystal structures in these materials.'},
         'input': {'generation': {'source_ids': [1]}, 'support': {'source_ids': [2]}}},
        {'id': 'b', 'level': 0, 'label': {'gloss': 'A graph network predicts energetic molecular properties.'},
         'input': {'generation': {'source_ids': [2]}, 'support': {'source_ids': [3]}}},
        {'id': 'c', 'level': 0, 'label': {'gloss': 'A language model predicts tokens in sentences.'},
         'input': {'generation': {'source_ids': [4]}, 'support': {'source_ids': [5]}}},
    ]
    pairs = cross_group_controls(records)
    assert pairs[0]['recipient_id'] == 'a'
    assert pairs[0]['donor_id'] == 'c'
    assert pairs[0]['is_known_false'] is False
    assert cross_group_controls(records[:2]) == []


def test_model_fenced_object_with_explanation_is_explicit_format_violation():
    from tkh.scientific_labels import extract_model_label
    obj = json.dumps({'label': 'Atomistic graph learning', 'gloss': 'Graph models represent atomic environments for property prediction.', 'status': 'coherent'})
    x = extract_model_label('```json\n' + obj + '\n```\nExplanation: evidence is limited.')
    assert x['label']['label'] == 'Atomistic graph learning'
    assert x['format_compliant'] is False
    assert x['ignored_trailing_text'] == 'Explanation: evidence is limited.'
    assert extract_model_label(obj)['format_compliant'] is True
    with pytest.raises(ValueError):
        extract_model_label('```json\n' + obj + '\n```\n' + obj)


def test_incident_claims_supply_support_for_nonclaim_members_without_source_leak():
    x = build_label_input(graph(), ['n0', 'n2', 'n4', 'n6'], 2022)
    assert x['support']['claim_context']
    docs = support_documents(x)
    assert len(docs) == 2
    assert all(set(d['source_ids']) <= set(x['support']['source_ids']) for d in docs)
    assert all(d['supporting_member_ids'] for d in docs)
    assert set(sum([d['claim_node_ids'] for d in docs], [])) <= {'n1', 'n3', 'n5', 'n7'}
    assert all(d['edge_ids'] for d in docs)


def test_incident_claim_with_future_date_is_never_used_as_support():
    g = graph()
    g['nodes'][1]['observed_at'] = 2023
    with pytest.raises(ValueError, match='future'):
        build_label_input(g, ['n0', 'n2', 'n4', 'n6'], 2022)


def test_four_synthetic_controls_do_not_set_empirical_decision_thresholds():
    a = calibrate_checker([.81, .99], [.01, .19])
    b = calibrate_checker([.97, .98], [.02, .03])
    assert a['support_threshold'] == b['support_threshold'] == .8
    assert a['unsupported_threshold'] == b['unsupported_threshold'] == .2


def test_imported_generator_must_match_exact_group_and_prompt_before_support_is_attached():
    from tkh.scientific_labels import attach_generated_labels
    inp=build_label_input(graph(),[f'n{i}' for i in range(8)],2022)
    expected=[{'id':'g','level':0,'member_count':8,'input':inp}]
    raw=json.dumps({'label':'Atomic property learning','gloss':'Atomic graph models predict properties from local structural environments.','status':'coherent'})
    output=[{'id':'g','level':0,'prompt_sha256':inp['prompt_sha256'],'raw_output':raw}]
    records=attach_generated_labels(expected,output)
    assert records[0]['input']==inp
    assert records[0]['label']['label']=='Atomic property learning'
    assert 'label' not in expected[0]
    output[0]['prompt_sha256']='different'
    with pytest.raises(ValueError,match='prompt'):
        attach_generated_labels(expected,output)


def test_imported_generation_cannot_drop_groups_or_silently_duplicate_a_group():
    from tkh.scientific_labels import attach_generated_labels
    with pytest.raises(ValueError,match='group'):
        attach_generated_labels([{'id':'g','level':0}],[])
    with pytest.raises(ValueError,match='group'):
        attach_generated_labels([], [{'id':'g','level':0},{'id':'g','level':0}])


def test_scientific_evidence_prompt_excludes_author_noise_and_adds_only_generation_claims():
    g=graph()
    for i in range(40):
        g['nodes'].append({'id':f'author{i}','type':'author','surface_form':f'Writer {i}',
            'observed_at':2020,'provenance':{'articles':[i%4]}})
    members=[n['id'] for n in g['nodes'] if n['type']!='claim']
    inp=build_label_input(g,members,2022,max_generation_members=24,selection_policy='science_evidence')
    payload=json.loads(inp['prompt'].split('EVIDENCE:\n')[1])
    assert payload['nodes'] and all(n['type']!='author' for n in payload['nodes'])
    assert payload['claim_context']
    for claim in payload['claim_context']:
        assert set(claim['node']['provenance']['articles'])<=set(inp['generation']['source_ids'])
        assert set(claim['supporting_member_ids'])<=set(inp['prompt_member_ids'])
    for node in inp['support']['nodes']:
        assert node['id'] not in inp['prompt']
