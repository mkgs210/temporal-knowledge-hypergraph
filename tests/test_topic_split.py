"""True old-member splits, atomic donor merges, and immutable global budgets."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tkh.novelty_edits import calibrate_novelty
from tkh.topic_split import propose_old_topic_split


def fixture(budget=3, metadata=False):
    names = ['s0', 's1', 's2', 's3', 'a0', 'a1', 'b0', 'b1']
    x = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0],
                  [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], float)
    nodes = [{'id': n, 'type': 'method', 'surface_form': n, 'observed_at': 2022,
              'provenance': {'articles': [i // 2]}} for i, n in enumerate(names)]
    assignment = {n: n[0] for n in names}
    if metadata:
        for g in 'sab':
            n = 'meta-' + g
            nodes.append({'id': n, 'type': 'article', 'surface_form': n,
                          'observed_at': 2022, 'provenance': {'articles': [0]}})
            assignment[n] = g
        x = np.vstack([x, np.tile([1., 0., 0.], (3, 1))])
    view = {'cutoff': 2022, 'nodes': nodes, 'hyperedges': []}
    previous = SimpleNamespace(view=view, vectors=x, assignment=assignment,
                               parents={'s': 'p', 'a': 'p', 'b': 'p'},
                               config={'budgets': [1, budget], 'gamma': .03,
                                       'normalizations': {'native_papers': 1}})
    c = calibrate_novelty(view, x)
    current = deepcopy(view)
    current['cutoff'] = 2023
    current['nodes'].append({'id': 'new', 'type': 'method', 'surface_form': 'new',
                             'observed_at': 2023, 'provenance': {'articles': [4]}})
    current['hyperedges'].append({'id': 'changed', 'relation_type': 'compares',
                                  'members': ['s2', 's3', 'new'], 'year': 2023,
                                  'provenance': {'article_id': 4}})
    science = {n: g for n, g in assignment.items() if not n.startswith('meta-')}
    science['new'] = 's'
    meta = {n: g for n, g in assignment.items() if n.startswith('meta-')}
    return previous, current, np.vstack([x, [0., 1., 0.]]), c, science, meta


def run(f, limit=4):
    p, v, x, c, a, m = f
    return propose_old_topic_split(p, v, x, starting_assignment=a,
                                   starting_parents=p.parents, metadata_assignment=m,
                                   max_old_edits=limit, calibration=c)


def test_saturated_transaction_splits_two_old_members_and_merges_donors():
    f = fixture()
    p, v, x, c, starting, meta = f
    baseline = deepcopy(starting)
    r = run(f)
    assert r['accepted'] and len(r['parents']) == 3
    split = next(e for e in r['events'] if e['kind'] == 'split')
    born = split['new_group_id']
    assert {r['assignment'][n] for n in ['s0', 's1', 's2', 's3']} == {'s', born}
    assert len([n for n in ['s0', 's1', 's2', 's3'] if r['assignment'][n] == born]) == 2
    assert len(r['moved_old_ids']) == 4
    assert {e['kind'] for e in r['events']} == {'split', 'merge'}
    assert all(parent == 'p' for parent in r['parents'].values())
    assert r['trace']['criterion_after']['total'] < r['trace']['criterion_before']['total']
    assert starting == baseline and p.parents == {'s': 'p', 'a': 'p', 'b': 'p'}


def test_too_small_budget_rolls_back_entire_merge_split():
    f = fixture()
    r = run(f, limit=3)
    assert not r['accepted'] and r['assignment'] == f[4] and r['parents'] == f[0].parents
    assert not r['events'] and not r['moved_old_ids']
    assert r['trace']['deferred'][0]['reason'] == 'old_edit_budget'


def test_free_slot_needs_only_genuine_split_old_members():
    r = run(fixture(budget=4), limit=2)
    assert r['accepted'] and len(r['parents']) == 4
    assert len(r['moved_old_ids']) == 2
    assert [e['kind'] for e in r['events']] == ['split']


def test_old_metadata_donor_path_is_counted_and_does_not_dangle():
    f = fixture(metadata=True)
    assert not run(f, limit=4)['accepted']
    r = run(f, limit=5)
    assert r['accepted'] and len(r['moved_old_ids']) == 5
    assert len(r['moved_old_scientific_ids']) == 4
    assert r['metadata_assignment']['meta-s'] == 's'
    assert set(r['metadata_assignment'].values()) <= r['parents'].keys()


def test_unchanged_view_cannot_trigger_opportunistic_old_split():
    f = list(fixture())
    p = f[0]
    f[1], f[2] = deepcopy(p.view), p.vectors.copy()
    f[4] = {n: g for n, g in p.assignment.items()}
    r = run(f)
    assert not r['accepted'] and r['trace']['candidate_count'] == 0


def test_unsupported_touch_with_no_cohesive_old_children_is_deferred():
    f = list(fixture())
    # Each proposed old pair has cosine .7, below the .8 support threshold.
    p = f[0]
    p.vectors[:4] = [[1., 0., 0.], [.7, np.sqrt(.51), 0.],
                     [-1., 0., 0.], [-.7, -np.sqrt(.51), 0.]]
    f[2] = np.vstack([p.vectors, [-1., 0., 0.]])
    f[3] = calibrate_novelty(p.view, p.vectors)
    f[3]['cohesion_threshold'] = .8
    r = run(f)
    assert not r['accepted'] and r['trace']['deferred'][0]['reason'] == 'incoherent_children'


def test_high_native_cost_can_reject_semantic_split():
    f = fixture(budget=4)
    f[0].config['gamma'] = 100.
    f[1]['hyperedges'].append({'id': 'tie', 'relation_type': 'compares',
                              'members': ['s0', 's2'], 'year': 2023,
                              'provenance': {'article_id': 5}})
    r = run(f, limit=8)
    assert not r['accepted']
    assert r['trace']['deferred'][0]['reason'] == 'no_beneficial_transaction'


def test_future_calibration_rejects_before_any_mutation():
    f = fixture()
    f[3]['cutoff'] = 2024
    with pytest.raises(ValueError, match='calibration'):
        run(f)


def test_no_donor_pair_in_same_parent_defers_without_growing_budget():
    f = fixture()
    f[0].parents = {'s': 'p', 'a': 'q', 'b': 'q'}
    f[0].config['budgets'][0] = 2
    r = run(f, limit=8)
    assert not r['accepted'] and len(r['parents']) == 3
    assert r['trace']['deferred'][0]['reason'] == 'no_same_parent_donor_pair'


def test_prior_scientific_and_metadata_moves_consume_same_global_budget():
    f = fixture(metadata=True)
    # Existing refinement has spent one edit unrelated to the split.
    f[4]['a0'] = 'b'
    r = run(f, limit=4)
    assert not r['accepted']
    assert r['assignment']['a0'] == 'b' and r['moved_old_ids'] == ['a0']


def test_unavailable_calibration_is_a_truthful_noop():
    f = fixture()
    f[3]['status'] = 'unavailable'
    r = run(f)
    assert not r['accepted'] and r['trace']['status'] == 'calibration_unavailable'
    assert r['assignment'] == f[4]


def test_complete_budget_validation_rejects_starting_overflow():
    f = fixture()
    f[0].config['budgets'][1] = 2
    with pytest.raises(ValueError, match='budgets'):
        run(f)


def test_a_cohesive_group_does_not_split_just_because_it_was_touched():
    f = fixture(budget=4)
    f[0].vectors[:4] = [1., 0., 0.]
    f = list(f)
    f[2] = np.vstack([f[0].vectors, [1., 0., 0.]])
    f[3] = calibrate_novelty(f[0].view, f[0].vectors)
    r = run(f)
    assert not r['accepted']
    assert r['trace']['deferred'][0]['reason'] in {'insufficient_semantic_gain', 'too_few_old_members_in_child'}


def test_split_requires_old_members_from_the_original_topic_in_both_children():
    f = fixture(budget=4)
    # Current refinement borrowed another old topic's nodes; that is not proof
    # that the original s topic has two surviving scientific subtopics.
    f[4].update(s2='a', s3='a', a0='s', a1='s')
    r = run(f, limit=8)
    assert not r['accepted']
    assert r['trace']['deferred'][0]['reason'] == 'too_few_original_old_members'
