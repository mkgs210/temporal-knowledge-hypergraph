"""Behavioral checks for bounded refinement, beyond unchanged trace flags."""
from copy import deepcopy
import importlib

import numpy as np
import pytest

from tkh.divisive_hierarchy import build_divisive, DivisiveState
from tkh.divisive_solver import objective


def api():
    return importlib.import_module('tkh.local_update')


def toy():
    nodes = [{'id': f'n{i:02d}', 'type': 'method', 'surface_form': f'method {i}',
              'observed_at': 2022, 'provenance': {'articles': [i // 10]}} for i in range(40)]
    x = np.array([np.eye(4)[i // 10] for i in range(40)])
    edges = [{'id': f'old-{i}', 'relation_type': 'claims', 'members': [n['id'] for n in nodes[i:i+10]],
              'year': 2022, 'provenance': {'article_id': i // 10}} for i in range(0, 40, 10)]
    view = {'cutoff': 2022, 'policy': 'test', 'limitations': [], 'nodes': nodes, 'hyperedges': edges}
    return view, x


def test_local_update_module_exists():
    assert importlib.util.find_spec('tkh.local_update') is not None


def test_weak_new_edges_to_every_topic_do_not_expand_the_frontier_over_whole_groups(tmp_path):
    view, x = toy(); previous = build_divisive(view, x, budgets=(2, 4))
    previous.save(tmp_path/'old'); previous = DivisiveState.load(tmp_path/'old')
    later = deepcopy(view); later['cutoff'] = 2023
    later['nodes'].append({'id': 'new', 'type': 'method', 'surface_form': 'new method',
                           'observed_at': 2023, 'provenance': {'articles': [5]}})
    later['hyperedges'] += [{'id': f'fresh-{i}', 'relation_type': 'claims',
        'members': ['new', n['id']], 'year': 2023, 'provenance': {'article_id': 5}}
        for i, n in enumerate(view['nodes'])]
    current = api().update_local(previous, later, np.vstack([x, np.eye(4)[0]]), max_old_fraction=.1)
    frontier = set(current.trace['allowed_old_nodes'])
    assert len(frontier) <= 4
    assert current.trace['affected_old_nodes_count'] == 40
    assert len(current.trace['deferred_affected_old_ids']) >= 36
    for n in set(previous.assignment) - frontier:
        assert current.assignment[n] == previous.assignment[n]
        assert current.parents[current.assignment[n]] == previous.parents[previous.assignment[n]]
    assert current.trace['initial_builder_calls'] == 0
    assert current.trace['resumed_from_saved_state']
    assert current.config['budgets'] == [2, 4]
    assert (current.affinity[:40, :40] != previous.affinity).nnz == 0
    np.testing.assert_array_equal(current.vectors[:40], previous.vectors)


def test_temporal_anchor_cost_matches_full_native_cost_for_every_allowed_move():
    edges = [{'id': 'e', 'members': ['a', 'b', 'c', 'd'], 'weight': 2.},
             {'id': 's', 'members': ['a', 'd'], 'weight': .7}]
    labels = {'a': 0, 'b': 0, 'c': 1, 'd': 2}
    q, fixed = api().regularized_boundary(edges, {'a'}, labels, labels, .4)
    start = {'a': 0, **fixed}
    for target in range(3):
        end = {'a': target, **fixed}
        full = {**labels, 'a': target}
        expected = objective(edges, full) - objective(edges, labels) + .4 * (target != 0)
        assert objective(q, end) - objective(q, start) == pytest.approx(expected)


def test_noop_does_not_create_events_or_modify_previous_state(tmp_path):
    view, x = toy(); previous = build_divisive(view, x, budgets=(2, 4))
    before = previous.to_dict()
    current = api().update_local(previous, deepcopy(view), x.copy())
    assert current.assignment == previous.assignment
    assert current.parents == previous.parents
    assert current.events == []
    assert current.trace['allowed_old_nodes'] == []
    assert previous.to_dict() == before


def test_new_only_component_cannot_change_remote_old_paths():
    view, x = toy(); previous = build_divisive(view, x, budgets=(2, 4))
    later = deepcopy(view); later['cutoff'] = 2023
    later['nodes'] += [{'id': f'z{i}', 'type': 'method', 'surface_form': f'new direction {i}',
                       'observed_at': 2023, 'provenance': {'articles': [9]}} for i in range(3)]
    later['hyperedges'].append({'id': 'new-only', 'relation_type': 'claims',
        'members': ['z0', 'z1', 'z2'], 'year': 2023, 'provenance': {'article_id': 9}})
    current = api().update_local(previous, later, np.vstack([x, np.tile([.5,.5,.5,.5], (3,1))]), max_old_fraction=0.)
    assert {n:current.assignment[n] for n in previous.assignment} == previous.assignment
    for n in previous.assignment:
        assert current.parents[current.assignment[n]] == previous.parents[previous.assignment[n]]
    assert len(current.assignment) == 43
    assert len(current.parents) <= 4
    assert len(set(current.parents.values())) <= 2


def test_changed_relations_can_move_an_old_record_and_temporal_cost_can_prevent_that_move():
    view, x = toy(); previous = build_divisive(view, x, budgets=(1, 4), gamma=1.)
    later = deepcopy(view); later['cutoff'] = 2023
    # Independent new source relations consistently associate n00 with topic 1.
    for source in range(5, 10):
        later['hyperedges'].append({'id': f'evidence-{source}', 'relation_type': 'claims',
            'members': ['n00'] + [f'n{i:02d}' for i in range(10, 20)], 'year': 2023,
            'provenance': {'article_id': source}})
    responsive = api().update_local(previous,later,x,max_old_fraction=.1,movement_strength=.05)
    assert responsive.assignment['n00'] == previous.assignment['n10']
    assert responsive.assignment['n00'] != previous.assignment['n00']
    assert len(responsive.trace['moved_ids']) <= 4
    resistant = api().update_local(previous,later,x,max_old_fraction=.1,movement_strength=80.)
    assert resistant.assignment['n00'] == previous.assignment['n00']


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan')])
def test_invalid_old_edit_fraction_rejected(value):
    view, x = toy(); previous = build_divisive(view, x, budgets=(2, 4))
    with pytest.raises(ValueError, match='fraction'):
        api().update_local(previous, view, x, max_old_fraction=value)


def test_hundred_additional_articles_keep_frozen_display_budget(tmp_path):
    from test_growth_budgets import check_hundred_article_growth
    check_hundred_article_growth(tmp_path, api().update_local)


def test_hundred_articles_with_novelty_enabled_cannot_expand_the_12_50_caps(tmp_path):
    from test_growth_budgets import check_hundred_article_growth
    from tkh.novelty_edits import calibrate_novelty
    frozen_calibration=None
    def updater(previous,view,x):
        nonlocal frozen_calibration
        if frozen_calibration is None:
            frozen_calibration=calibrate_novelty(previous.view,previous.vectors)
        return api().update_local(previous,view,x,novelty_calibration=frozen_calibration)
    check_hundred_article_growth(tmp_path,updater)


def test_novelty_transaction_is_used_without_raising_fine_budget_or_restarting():
    from test_novelty_edits import fixture, later, calibration
    original = fixture()
    # Use a real persisted-state-compatible starting structure with its affinity.
    previous = build_divisive(original.view,original.vectors,budgets=(1,3),gamma=.03)
    # The hand-checked split/merge example needs distinct a/b subtopics, which a
    # generic cut may tie differently; these are legal native input partitions.
    previous.assignment = original.assignment.copy(); previous.parents = original.parents.copy()
    view,x = later(previous)
    current = api().update_local(previous,view,x,max_old_fraction=.5,
                                 mode='bounded',novelty_calibration=calibration(original))
    assert current.trace['novelty']['accepted']
    assert len(current.parents) == 3
    assert current.config['budgets'] == [1,3]
    assert len({current.assignment[f'z{i}'] for i in range(3)}) == 1
    born = set(current.parents)-set(previous.parents)
    assert current.assignment['z0'] in born
    assert {e['kind'] for e in current.events} >= {'birth','merge','death'}
    assert len(current.trace['moved_ids']) <= 3


def test_new_author_of_a_merged_old_article_gets_a_surviving_topic_path():
    from test_novelty_edits import fixture, later, calibration
    original = fixture()
    for source, group in enumerate('abc'):
        original.view['nodes'].append({'id':'paper-'+group,'type':'article','surface_form':'old paper '+group,
            'observed_at':2022,'provenance':{'articles':[source]}})
        original.vectors = np.vstack([original.vectors, original.vectors[source*2]])
        original.assignment['paper-'+group] = group
    previous = build_divisive(original.view,original.vectors,budgets=(1,3))
    previous.assignment = original.assignment.copy(); previous.parents = original.parents.copy()
    view,x = later(previous)
    for source,group in enumerate('abc'):
        view['nodes'].append({'id':'new-author-'+group,'type':'author','surface_form':'new author '+group,
            'observed_at':2023,'provenance':{'articles':[source]}})
        x = np.vstack([x, previous.vectors[source*2]])
    current = api().update_local(previous,view,x,max_old_fraction=1.,
                                 novelty_calibration=calibration(original))
    assert current.trace['novelty']['accepted']
    for group in 'abc':
        assert current.assignment['new-author-'+group] == current.assignment['paper-'+group]
        assert current.assignment['new-author-'+group] in current.parents


def test_genuine_old_split_is_integrated_atomically_and_keeps_fixed_group_budget():
    from test_topic_split import fixture
    original, view, x, calibration, _, _ = fixture()
    previous = build_divisive(original.view,original.vectors,budgets=(1,3),gamma=.03)
    previous.assignment=original.assignment.copy();previous.parents=original.parents.copy()
    current=api().update_local(previous,view,x,max_old_fraction=.5,
        novelty_calibration=calibration,allow_old_topic_split=True)
    assert current.trace['novelty']['accepted']
    split=next(e for e in current.events if e['kind']=='split')
    assert split['retained_group_id']=='s'
    assert split['new_group_id'] in current.parents
    assert len(current.parents)==3 and current.config['budgets']==[1,3]
    assert len({current.assignment[n] for n in ['s0','s1','s2','s3']})==2
    assert len(current.trace['moved_ids'])<=4
    assert {e['kind'] for e in current.events}>={'merge','split','death'}
    for n in set(previous.assignment)-set(current.trace['allowed_old_nodes']):
        assert current.assignment[n]==previous.assignment[n]


def test_old_split_requires_first_view_calibration():
    view,x=toy();previous=build_divisive(view,x,budgets=(2,4))
    with pytest.raises(ValueError,match='calibration'):
        api().update_local(previous,view,x,allow_old_topic_split=True)
