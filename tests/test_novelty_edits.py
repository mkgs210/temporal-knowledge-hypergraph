"""Atomic novelty transactions: hand-checked budget and criterion examples."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tkh.novelty_edits import calibrate_novelty, propose_topic_edits, topic_edit_criterion


def fixture(budget=3):
    nodes = [{'id': f'{g}{i}', 'type': 'method', 'surface_form': f'{g} method {i}',
              'observed_at': 2022, 'provenance': {'articles': [s]}}
             for s, g in enumerate('abc') for i in range(2)]
    x = np.array([[1, 0, 0], [1, 0, 0], [np.cos(.02), np.sin(.02), 0],
                  [np.cos(.02), np.sin(.02), 0], [0, 1, 0], [0, 1, 0]], float)
    view = {'cutoff': 2022, 'nodes': nodes, 'hyperedges': [], 'policy': 'test'}
    previous = SimpleNamespace(view=view, vectors=x, assignment={n['id']: n['id'][0] for n in nodes},
                               parents={'a': 'p', 'b': 'p', 'c': 'p'},
                               config={'budgets': [1, budget], 'gamma': .03,
                                       'normalizations': {'native_papers': 1}})
    return previous


def later(previous, count=3):
    view = deepcopy(previous.view)
    view['cutoff'] = 2023
    view['nodes'] += [{'id': f'z{i}', 'type': 'method', 'surface_form': f'new {i}',
                       'observed_at': 2023, 'provenance': {'articles': [3]}} for i in range(count)]
    return view, np.vstack([previous.vectors, np.tile([0., 0., 1.], (count, 1))])


def calibration(previous):
    return calibrate_novelty(previous.view, previous.vectors, alpha=.2, min_cohort_size=3)


def test_calibration_holds_out_every_node_sharing_the_source():
    previous = fixture()
    previous.view['nodes'][0]['provenance']['articles'] = [0, 1]
    c = calibration(previous)
    rows = [r for r in c['scores'] if r['held_out_source'] == 0]
    assert {r['node_id'] for r in rows} == {'a0', 'a1'}
    fold = next(f for f in c['folds'] if f['held_out_source'] == 0)
    assert 'a0' not in fold['reference_ids'] and 'a1' not in fold['reference_ids']
    assert c['cutoff'] == 2022 and c['guaranteed_false_alarm_control'] is False
    assert 0 < c['novelty_threshold'] < 1


def test_saturated_budget_frees_slot_with_same_parent_merge_and_preserves_input():
    previous = fixture()
    view, x = later(previous)
    old_assignment = dict(previous.assignment)
    result = propose_topic_edits(previous, view, x, max_old_edits=2, calibration=calibration(previous))
    assert result['accepted']
    assert len(result['parents']) == 3
    born = set(result['parents']) - set(previous.parents)
    retired = set(previous.parents) - set(result['parents'])
    assert len(born) == len(retired) == 1
    assert {result['assignment'][f'z{i}'] for i in range(3)} == born
    assert len(result['moved_old_ids']) == 2
    assert previous.assignment == old_assignment and len(previous.parents) == 3
    assert set(result['assignment']) == {n['id'] for n in view['nodes']}
    assert all(result['parents'][result['assignment'][n]] == previous.parents[g]
               for n, g in previous.assignment.items())
    assert {e['kind'] for e in result['events']} == {'merge', 'birth'}
    assert result['trace']['criterion_after']['total'] < result['trace']['criterion_before']['total']


def test_conservative_calibration_requires_cohesive_multiple_source_support():
    previous=fixture()
    c=calibrate_novelty(previous.view,previous.vectors,cohesion_quantile=.75,min_cohort_sources=2)
    view,x=later(previous)
    one=propose_topic_edits(previous,view,x,max_old_edits=2,calibration=c)
    assert not one['accepted']
    assert any(r['reason']=='insufficient_distinct_sources' for r in one['trace']['deferred'])
    view['nodes'][-1]['provenance']['articles']=[4]
    multi=propose_topic_edits(previous,view,x,max_old_edits=2,calibration=c)
    assert multi['accepted']
    assert len(multi['parents'])==3
    assert c['cohesion_quantile']==.75 and c['min_cohort_sources']==2


def test_insufficient_edit_budget_defers_without_partial_merge_or_new_id():
    previous = fixture()
    view, x = later(previous)
    result = propose_topic_edits(previous, view, x, max_old_edits=1, calibration=calibration(previous))
    assert not result['accepted'] and not result['events']
    assert result['parents'] == previous.parents
    assert result['moved_old_ids'] == []
    assert result['trace']['deferred'][0]['reason'] == 'old_edit_budget'


def test_free_slot_birth_needs_no_old_node_edit():
    previous = fixture(budget=4)
    view, x = later(previous)
    result = propose_topic_edits(previous, view, x, max_old_edits=0, calibration=calibration(previous))
    assert result['accepted'] and len(result['parents']) == 4
    assert result['moved_old_ids'] == []
    assert [e['kind'] for e in result['events']] == ['birth']


def test_single_novel_name_cannot_create_a_topic():
    previous = fixture(budget=4)
    view, x = later(previous, count=1)
    result = propose_topic_edits(previous, view, x, max_old_edits=3, calibration=calibration(previous))
    assert not result['accepted'] and result['parents'] == previous.parents
    assert result['trace']['deferred'][0]['reason'] == 'cohort_too_small'


def test_high_native_cost_can_reject_semantically_attractive_birth():
    previous = fixture(budget=4)
    previous.config['gamma'] = 100.
    view, x = later(previous)
    view['hyperedges'] = [{'id': 'native', 'relation_type': 'compares',
                          'members': ['a0', 'z0', 'z1', 'z2'], 'year': 2023,
                          'provenance': {'article_id': 3}}]
    result = propose_topic_edits(previous, view, x, max_old_edits=6, calibration=calibration(previous))
    assert not result['accepted'] and not result['events']
    assert result['trace']['deferred'][0]['reason'] == 'no_beneficial_transaction'


def test_coarse_path_change_counts_as_old_movement_in_criterion():
    previous = fixture()
    changed_parents = dict(previous.parents, a='q')
    values = topic_edit_criterion(previous, previous.view, previous.vectors,
                                 previous.assignment, changed_parents, temporal_weight=.6)
    assert values['old_path_moves'] == 2
    assert values['temporal'] == pytest.approx(.2)


def test_future_calibration_and_rewritten_old_vectors_are_rejected():
    previous = fixture()
    view, x = later(previous)
    c = calibration(previous)
    c['cutoff'] = 2024
    with pytest.raises(ValueError, match='calibration'):
        propose_topic_edits(previous, view, x, max_old_edits=2, calibration=c)
    x[0] = [0, 1, 0]
    with pytest.raises(ValueError, match='embedding'):
        propose_topic_edits(previous, view, x, max_old_edits=2, calibration=calibration(previous))


def test_same_source_only_calibration_is_explicitly_unavailable():
    previous = fixture()
    for node in previous.view['nodes']:
        node['provenance']['articles'] = [0]
    c = calibration(previous)
    assert c['status'] == 'unavailable' and c['novelty_threshold'] is None
    view, x = later(previous)
    result = propose_topic_edits(previous, view, x, max_old_edits=6, calibration=c)
    assert not result['accepted'] and result['trace']['status'] == 'calibration_unavailable'


@pytest.mark.parametrize('mutation', ['future_edge', 'future_source', 'unknown_member', 'duplicate_endpoint'])
def test_calibrator_rejects_invalid_hyperedge_evidence(mutation):
    previous = fixture()
    edge = {'id': 'e', 'members': ['a0', 'a1'], 'relation_type': 'compares', 'year': 2022,
            'provenance': {'article_id': 0, 'article_year': 2022}}
    if mutation == 'future_edge':
        edge['year'] = 2023
    elif mutation == 'future_source':
        edge['provenance']['article_year'] = 2023
    elif mutation == 'unknown_member':
        edge['members'].append('not-a-node')
    else:
        edge['members'].append('a0')
    previous.view['hyperedges'] = [edge]
    with pytest.raises(ValueError):
        calibration(previous)


def test_calibration_uses_no_same_source_reference_when_node_is_shared():
    previous = fixture()
    previous.view['nodes'][0]['provenance']['articles'] = [0, 1]
    c = calibration(previous)
    fold = next(f for f in c['folds'] if f['held_out_source'] == 1)
    assert set(fold['reference_ids']) == {'a1', 'c0', 'c1'}


def test_a_merge_cannot_use_a_group_from_a_different_parent_to_free_capacity():
    previous = fixture()
    previous.parents = {'a': 'p', 'b': 'q', 'c': 'r'}
    previous.config['budgets'][0] = 3
    view, x = later(previous)
    result = propose_topic_edits(previous, view, x, max_old_edits=6, calibration=calibration(previous))
    assert not result['accepted'] and result['parents'] == previous.parents
    assert result['trace']['deferred'][0]['reason'] == 'no_same_parent_merge'


def test_no_novelty_alarm_is_a_legal_append_without_old_movement():
    previous = fixture()
    view, x = later(previous)
    x[6:] = [1., 0., 0.]
    result = propose_topic_edits(previous, view, x, max_old_edits=0, calibration=calibration(previous))
    assert not result['accepted'] and len(result['assignment']) == 9
    assert result['moved_old_ids'] == [] and result['parents'] == previous.parents


def test_criterion_counts_native_three_way_occupancy_and_spherical_dispersion():
    previous = fixture()
    previous.vectors = np.repeat(np.eye(3), 2, axis=0)
    previous.view['hyperedges'] = [{'id': 'threeway', 'relation_type': 'compares',
                                    'members': ['a0', 'b0', 'c0'], 'year': 2022,
                                    'provenance': {'article_id': 0}}]
    values = topic_edit_criterion(previous, previous.view, previous.vectors,
                                 previous.assignment, previous.parents)
    assert values['semantic_fine'] == pytest.approx(0.)
    assert values['semantic_coarse'] == pytest.approx(.42264973081037427)
    assert values['native_fine'] == pytest.approx(.06)
    assert values['native_coarse'] == pytest.approx(0.)
    assert values['total'] == pytest.approx(.48264973081037427)


def test_starting_assignment_preserves_earlier_refinement_and_charges_it():
    previous = fixture()
    view, x = later(previous)
    starting = dict(previous.assignment, c1='a', z0='a', z1='a', z2='a')
    result = propose_topic_edits(previous, view, x, max_old_edits=1, calibration=calibration(previous),
                                 starting_assignment=starting, starting_parents=previous.parents)
    assert not result['accepted']
    assert result['assignment']['c1'] == 'a'
    assert result['moved_old_ids'] == ['c1']
    assert result['trace']['deferred'][0]['reason'] == 'old_edit_budget'


def test_retiring_fine_group_remaps_and_charges_its_old_metadata_members():
    previous = fixture()
    for g in 'abc':
        nid = f'metadata-{g}'
        previous.view['nodes'].append({'id': nid, 'type': 'author', 'surface_form': nid,
                                       'observed_at': 2022, 'provenance': {'articles': [0]}})
        previous.assignment[nid] = g
    previous.vectors = np.vstack([previous.vectors, np.tile([1., 0., 0.], (3, 1))])
    view, x = later(previous)
    small = propose_topic_edits(previous, view, x, max_old_edits=2, calibration=calibration(previous))
    assert not small['accepted']
    result = propose_topic_edits(previous, view, x, max_old_edits=3, calibration=calibration(previous))
    assert result['accepted']
    assert len(result['moved_old_ids']) == 3 and len(result['moved_old_scientific_ids']) == 2
    retired = (set(previous.parents) - set(result['parents'])).pop()
    assert result['metadata_assignment'][f'metadata-{retired}'] in result['parents']
    assert result['metadata_assignment'][f'metadata-{retired}'] != retired
    assert all(g in result['parents'] for g in result['metadata_assignment'].values())
