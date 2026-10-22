"""Hand-computable native-edge and saved-state contracts, independent of Jina."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest


def toy():
    nodes = [{"id": str(i), "type": "method", "surface_form": s, "observed_at": 2022,
              "provenance": {"articles": [1]}} for i, s in enumerate(
        ["graph network", "graph convolution", "molecular force", "atomic force", "crystal lattice", "crystal energy"])]
    edges = [{"id": "e1", "members": ["0", "1", "2"], "relation_type": "uses", "year": 2022,
              "provenance": {"article_id": 1, "article_year": 2022}}]
    x = np.array([[1, 0, 0], [.99, .1, 0], [0, 1, 0], [0, .99, .1], [0, 0, 1], [.1, 0, .99]], dtype=float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return {"cutoff": 2022, "nodes": nodes, "hyperedges": edges}, x


def test_collapse_keeps_internal_edges_counts_and_composes():
    from tkh.hierarchy import collapse
    edges = [{"id": "e", "members": ["a", "b", "c", "d", "e"], "relation_type": "r", "provenance": {"source": 7}}]
    fine = collapse(edges, dict(zip("abcde", ["A", "A", "A", "B", "C"])))
    assert fine[0]["counts"] == {"A": 3, "B": 1, "C": 1}
    assert fine[0]["provenance"] == {"source": 7}
    assert collapse(fine, {"A": "X", "B": "X", "C": "X"}) == collapse(edges, dict.fromkeys("abcde", "X"))
    assert collapse(fine, {"A": "X", "B": "X", "C": "X"})[0]["counts"] == {"X": 5}


def test_structural_criterion_really_changes_merge_choice():
    from tkh.hierarchy import agglomerate
    x = np.array([[1., 0], [.99, .1], [0., 1]])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    edges = [{"id": "edge", "members": ["a", "c"], "relation_type": "r"}]
    semantic = agglomerate(["a", "b", "c"], x, edges, [2], structural_weight=0)[2]
    native = agglomerate(["a", "b", "c"], x, edges, [2], structural_weight=10)[2]
    assert {frozenset(g) for g in semantic} == {frozenset("ab"), frozenset("c")}
    assert {frozenset(g) for g in native} == {frozenset("ac"), frozenset("b")}


def test_shared_merge_history_is_laminar_for_all_budget_candidates():
    from tkh.hierarchy import agglomerate
    view, x = toy()
    cuts = agglomerate([n["id"] for n in view["nodes"]], x, view["hyperedges"], [1, 2, 3, 4])
    for finer, coarser in [(4, 3), (3, 2), (2, 1)]:
        assert all(sum(set(child) <= set(parent) for parent in cuts[coarser]) == 1 for child in cuts[finer])


def test_update_resumes_without_build_and_empty_batch_is_idempotent(tmp_path, monkeypatch):
    import tkh.hierarchy as h
    view, x = toy()
    state = h.build(view, x, budgets=(2, 3), structural_weight=.2)
    before = copy.deepcopy(state.to_dict())
    state.save(tmp_path / "checkpoint")
    resumed = h.State.load(tmp_path / "checkpoint")
    monkeypatch.setattr(h, "agglomerate", lambda *a, **kw: pytest.fail("update rebuilt the hierarchy"))
    same = h.update(resumed, view, x)
    assert same.to_dict() == before
    later = copy.deepcopy(view)
    later["cutoff"] = 2023
    later["nodes"].append({"id": "6", "type": "method", "surface_form": "graph attention", "observed_at": 2023,
                           "provenance": {"articles": [2]}})
    later["hyperedges"].append({"id": "e2", "members": ["0", "6"], "relation_type": "uses", "year": 2023,
                               "provenance": {"article_id": 2, "article_year": 2023}})
    xx = np.vstack([x, [1, 0, 0]])
    a = h.update(state, later, xx)
    b = h.update(resumed, later, xx)
    assert a.to_dict() == b.to_dict()
    assert state.to_dict() == before
    assert h.update(a, later, xx).to_dict() == a.to_dict()
    assert set(a.assignment) == {str(i) for i in range(7)}
    assert len(set(a.assignment.values())) <= 3
    assert a.trace["initial_builder_calls"] == 0


def test_invalid_update_does_not_damage_checkpoint(tmp_path):
    from tkh.hierarchy import State, build, update
    view, x = toy()
    state = build(view, x, budgets=(2, 3))
    state.save(tmp_path / "good")
    before = (tmp_path / "good" / "state.json").read_bytes()
    broken = copy.deepcopy(view)
    broken["hyperedges"][0]["members"] = ["unknown"]
    with pytest.raises(ValueError):
        update(state, broken, x)
    assert (tmp_path / "good" / "state.json").read_bytes() == before
    assert State.load(tmp_path / "good").to_dict() == state.to_dict()


def test_new_edge_between_old_nodes_is_processed():
    from tkh.hierarchy import build, update
    view, x = toy()
    state = build(view, x, budgets=(2, 3), structural_weight=10)
    later = copy.deepcopy(view)
    later["cutoff"] = 2023
    later["hyperedges"].append({"id": "old-only", "members": ["1", "4"], "relation_type": "r", "year": 2023})
    changed = update(state, later, x)
    assert changed.trace["new_edges"] == ["old-only"]
    assert {"1", "4"} <= set(changed.trace["allowed_old_nodes"])
    assert changed.trace["initial_builder_calls"] == 0


def test_label_hash_changes_with_text_and_parent_union_is_exact():
    from tkh.hierarchy import build, label_hash
    view, x = toy()
    state = build(view, x, budgets=(2, 3))
    hierarchy, inputs = state.export()
    for level in hierarchy["levels"]:
        for group in level["groups"]:
            assert label_hash(inputs[group["id"]]) == group["label_input_hash"]
            assert group["label"] and group["gloss"] and set(group["evidence"]) <= set(group["members"])
    for level in hierarchy["levels"][1:]:
        parents = {g["id"]: set(g["members"]) for g in hierarchy["levels"][level["index"] - 1]["groups"]}
        for pid, members in parents.items():
            assert set().union(*(set(g["members"]) for g in level["groups"] if g["parent_id"] == pid)) == members
    altered = copy.deepcopy(inputs[next(iter(inputs))])
    altered[0]["surface_form"] += " DIFFERENT"
    assert label_hash(altered) != label_hash(inputs[next(iter(inputs))])


def test_novel_topic_gets_new_id_without_exceeding_full_budget():
    from tkh.hierarchy import build, update
    view, _ = toy()
    x = np.array([[1., 0, 0], [.99, .01, 0], [.98, .02, 0], [.95, .05, 0], [.94, .06, 0], [.93, .07, 0]])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    state = build(view, x, budgets=(1, 3))
    later = copy.deepcopy(view)
    later['cutoff'] = 2023
    later['nodes'].append({'id': '6', 'type': 'method', 'surface_form': 'entirely new topic', 'observed_at': 2023})
    after = update(state, later, np.vstack([x, [0, 0, 1]]))
    assert after.assignment['6'] not in state.parents
    assert len(after.parents) <= 3
    assert after.next_id > state.next_id
    assert any(e['kind'] == 'birth' and e['level'] == 1 for e in after.events)
    assert any(e['kind'] == 'merge' and e['level'] == 1 for e in after.events)
    assert after.trace['novel_topic_births']


def test_repeated_novel_repairs_only_reference_groups_in_public_snapshots():
    from tkh.hierarchy import build, update
    rng = np.random.default_rng(15)
    x = rng.normal(size=(22, 32))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    oldids = [f'n{i:02d}' for i in range(10)]
    newids = [f'z{i:02d}' for i in range(12)]
    def view(ids, year):
        return {'cutoff': year, 'nodes': [{'id': n, 'type': 'method', 'surface_form': n} for n in ids], 'hyperedges': []}
    before = build(view(oldids, 2022), x[:10], budgets=(1, 5))
    after = update(before, view(oldids + newids, 2023), x)
    for event in after.events:
        if event['level'] == 1 and event['kind'] != 'death':
            assert event['group_id'] in after.parents
        if event['level'] == 1:
            assert all(g in before.parents for g in event['previous_group_ids'])


@pytest.mark.parametrize('change', ['policy', 'observed_at', 'sources'])
def test_update_rejects_silent_time_policy_and_old_evidence_rewrites(change):
    from tkh.hierarchy import build, update
    view, x = toy()
    view['policy'] = 'frozen-policy'
    before = build(view, x, budgets=(2, 3))
    later = copy.deepcopy(view)
    later['cutoff'] = 2023
    if change == 'policy':
        later['policy'] = 'different'
    elif change == 'observed_at':
        later['nodes'][0]['observed_at'] = 1990
    else:
        later['nodes'][0]['provenance']['articles'] = []
    with pytest.raises(ValueError):
        update(before, later, x)


def test_public_merge_event_is_supported_by_final_member_union():
    from tkh.hierarchy import build, update
    rng = np.random.default_rng(1)
    x = rng.normal(size=(22, 32)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = [f'n{i:02d}' for i in range(10)] + [f'z{i:02d}' for i in range(12)]
    def view(ns, year):
        return {'cutoff': year, 'nodes': [{'id': n, 'type': 'method', 'surface_form': n} for n in ns], 'hyperedges': []}
    before = build(view(ids[:10], 2022), x[:10], budgets=(1, 5))
    after = update(before, view(ids, 2023), x)
    for event in after.events:
        if event['kind'] == 'merge':
            members = {n for n,g in after.assignment.items() if g == event['group_id']}
            assert len(event['previous_group_ids']) >= 2
            for old in event['previous_group_ids']:
                assert {n for n,g in before.assignment.items() if g == old} <= members


def test_coarse_move_cannot_hide_untouched_members_absorbed_by_budget_repair():
    from tkh.hierarchy import build, update
    rng = np.random.default_rng(37)
    x = rng.normal(size=(18, 16)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    oldids = [f'n{i:02d}' for i in range(12)]
    newids = [f'z{i:02d}' for i in range(6)]
    def view(ns, year, edges):
        return {'cutoff': year, 'nodes': [{'id': n, 'type': 'method', 'surface_form': n} for n in ns], 'hyperedges': edges}
    cuts = {3: [oldids[i:i+4] for i in range(0, 12, 4)], 6: [oldids[i:i+2] for i in range(0, 12, 2)]}
    before = build(view(oldids, 2022, []), x[:12], budgets=(3, 6), cuts=cuts)
    after = update(before, view(oldids+newids, 2023, [{'id': 'e', 'members': oldids[:2]}]), x)
    for nid in oldids[2:]:
        assert after.parents[after.assignment[nid]] == before.parents[before.assignment[nid]]
