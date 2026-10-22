"""Independent exhaustive tiny-graph checks of mathematical and state contracts."""
from collections import Counter
import copy
import itertools
import json
import math

import numpy as np
import pytest


def objective(ids, vectors, edges, assignment, weight, masses=None, old=None, penalty=0.):
    """Definition recomputed from member lists; no production math helpers."""
    masses = np.ones(len(ids)) if masses is None else np.asarray(masses)
    total = float(masses.sum())
    loss = 0.
    for group in set(assignment.values()):
        indices = [i for i, n in enumerate(ids) if assignment.get(n) == group]
        mean = np.average(vectors[indices], axis=0, weights=masses[indices])
        loss += sum(float(masses[i] * np.square(vectors[i] - mean).sum()) for i in indices) / total
    structural = 0.
    for edge in edges:
        occurrences = edge.get('counts', {n: 1 for n in edge['members']})
        counts = Counter()
        for node, count in occurrences.items():
            if node in assignment:
                counts[assignment[node]] += count
        arity = sum(occurrences.values())
        if arity > 1:
            structural -= sum((count / arity) * math.log(count / arity) for count in counts.values() if count) / math.log(arity)
    loss += weight * structural / max(len(edges), 1)
    if old:
        loss += penalty * sum(masses[i] for i, n in enumerate(ids)
                              if n in old and assignment.get(n) != old[n]) / total
    return loss


def brute_greedy(ids, x, edges, weight):
    assignment = {n: n for n in ids}
    result = {len(ids): {frozenset([n]) for n in ids}}
    while len(set(assignment.values())) > 1:
        choices = []
        for a, b in itertools.combinations(sorted(set(assignment.values())), 2):
            proposed = {n: (a if g == b else g) for n, g in assignment.items()}
            choices.append((objective(ids, x, edges, proposed, weight), a, b, proposed))
        best = min(choices, key=lambda row: row[:3])
        # Random continuous inputs keep the winning pair unique. This checks the
        # algorithm's objective rather than prescribing arbitrary tie ordering.
        values = sorted(c[0] for c in choices)
        if len(values) > 1:
            assert values[1] - values[0] > 1e-10
        assignment = best[3]
        result[len(set(assignment.values()))] = {
            frozenset(n for n, g in assignment.items() if g == group)
            for group in set(assignment.values())}
    return result


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('weight', [0., .2, 4.])
def test_every_greedy_cut_minimizes_joint_objective_over_all_next_pairs(seed, weight):
    from tkh.hierarchy import agglomerate
    rng = np.random.default_rng(seed)
    ids = list('abcdefg')
    x = rng.normal(size=(len(ids), 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    edges = [{'id': 'e0', 'members': list('abce')},
             {'id': 'e1', 'members': list('bfg')},
             {'id': 'e2', 'members': list('acg')},
             {'id': 'e3', 'members': ['d']}]
    expected = brute_greedy(ids, x, edges, weight)
    actual = agglomerate(ids, x, edges, list(range(1, len(ids) + 1)), weight)
    for count, groups in expected.items():
        assert {frozenset(g) for g in actual[count]} == groups


@pytest.mark.parametrize('seed', range(5))
@pytest.mark.parametrize('collapsed', [False, True])
def test_local_move_deltas_equal_full_native_objective_recomputation(seed, collapsed):
    from tkh.hierarchy import _LocalPartition
    rng = np.random.default_rng(seed + 100)
    ids = list('abcdef')
    x = rng.normal(size=(6, 3))
    masses = np.array([3., 1., 4., 2., 1., 5.]) if collapsed else np.ones(6)
    edges = [{'id': 'e1', 'members': list('abcde')}, {'id': 'e2', 'members': list('bdf')},
             {'id': 'e3', 'members': ['f']}]
    if collapsed:
        edges[0]['counts'] = dict(zip('abcde', [3, 1, 4, 2, 1]))
        edges[1]['counts'] = dict(zip('bdf', [1, 2, 5]))
        edges[2]['counts'] = {'f': 5}
    old = dict(zip(ids, ['A', 'A', 'A', 'B', 'B', 'B']))
    assignment = dict(zip(ids, ['A', 'A', 'B', 'B', 'C', 'C']))
    partition = _LocalPartition(ids, x, edges, assignment, .37, old, .09, masses=masses)
    # Includes empty target, disappearance of a source group, and changes after
    # multiple accepted moves (sufficient statistics must remain up to date).
    for node, target in [('a', 'B'), ('e', 'D'), ('f', 'A'), ('b', 'D'), ('c', 'C'), ('d', 'D')]:
        before = objective(ids, x, edges, assignment, .37, masses, old, .09)
        after_assignment = dict(assignment, **{node: target})
        expected = objective(ids, x, edges, after_assignment, .37, masses, old, .09) - before
        assert partition.delta(node, target) == pytest.approx(expected, abs=1e-12)
        partition.move(node, target)
        assignment = after_assignment
        assert partition.assignment == assignment


def sample_state():
    from tkh.hierarchy import build
    ids = list('abcd')
    view = {'cutoff': 2022, 'nodes': [{'id': n, 'type': 'method', 'surface_form': n} for n in ids],
            'hyperedges': [{'id': 'edge', 'members': ids[:3]}]}
    x = np.array([[1., 0.], [.99, .1], [0., 1.], [.1, .99]])
    return build(view, x, budgets=(1, 2)), view, x


@pytest.mark.parametrize('damage', ['missing_assignment', 'unknown_member', 'missing_parent'])
def test_load_rejects_corrupt_structural_checkpoint_metadata(tmp_path, damage):
    from tkh.hierarchy import State
    state, _, _ = sample_state()
    path = tmp_path / 'state'
    state.save(path)
    payload = json.loads((path / 'state.json').read_text())
    if damage == 'missing_assignment':
        del payload['assignment']['a']
    elif damage == 'unknown_member':
        payload['assignment']['unknown'] = next(iter(payload['parents']))
    else:
        del payload['parents'][next(iter(payload['parents']))]
    (path / 'state.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        State.load(path)


def test_load_rejects_changed_vectors_even_when_metadata_unmodified(tmp_path):
    from tkh.hierarchy import State
    state, _, _ = sample_state()
    path = tmp_path / 'state'
    state.save(path)
    altered = state.vectors.copy()
    altered[0, 0] += .2
    np.savez_compressed(path / 'vectors.npz', vectors=altered)
    with pytest.raises(ValueError, match='Corrupt'):
        State.load(path)


def test_update_keeps_assignments_outside_predeclared_allowed_groups():
    from tkh.hierarchy import build, update
    ids = list('abcdefghijkl')
    view = {'cutoff': 2022, 'nodes': [{'id': n, 'type': 'method', 'surface_form': n} for n in ids],
            'hyperedges': [{'id': 'old', 'members': ['a', 'b']}]}
    cuts = {3: [list('abcd'), list('efgh'), list('ijkl')],
            6: [list(pair) for pair in ['ab', 'cd', 'ef', 'gh', 'ij', 'kl']]}
    # Deliberately poor unrelated groups could improve under global refinement;
    # the small new edge must not silently permit touching all of them.
    x = np.array([[1., 0.], [.99, .1], [.95, .2], [.9, .3],
                  [1., 0.], [-1., 0.], [-.99, .1], [.99, -.1],
                  [0., 1.], [0., -1.], [.1, -.99], [-.1, .99]])
    state = build(view, x, budgets=(3, 6), cuts=cuts)
    later = copy.deepcopy(view)
    later['cutoff'] = 2023
    later['hyperedges'].append({'id': 'old-only', 'members': ['a', 'b']})
    before = state.to_dict()
    after = update(state, later, x)
    allowed = set(after.trace['allowed_old_nodes'])
    assert allowed == set('ab')
    for n in set(ids) - allowed:
        assert after.assignment[n] == state.assignment[n]
        assert after.parents[after.assignment[n]] == state.parents[state.assignment[n]]
    assert state.to_dict() == before
