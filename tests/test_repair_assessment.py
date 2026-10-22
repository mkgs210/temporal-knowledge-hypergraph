"""Scientific assessment must check artifacts, not trust diagnostic flags."""
from copy import deepcopy
import json

import numpy as np
import pytest


def frontier_fixture():
    previous = {"nodes": [{"id": n, "type": "method"} for n in "abcd"]}
    current = {"nodes": previous["nodes"] + [{"id": "e", "type": "method"}]}
    def hierarchy(left, right):
        return {"levels": [
            {"index": 0, "groups": [{"id": "C", "members": left + right}]},
            {"index": 1, "groups": [{"id": "F1", "parent_id": "C", "members": left},
                                      {"id": "F2", "parent_id": "C", "members": right}]},
        ]}
    trace = {"allowed_old_nodes": ["b"], "editable_ids": ["b", "e"],
             "moved_ids": ["b"], "old_edit_limit": 1,
             "initial_builder_calls": 0, "realized_old_path_edits": 1,
             "resumed_from_saved_state": True, "solver_calls": []}
    return previous, hierarchy(list("ab"), list("cd")), current, hierarchy(["a"], list("bcde")), trace


def test_frontier_audit_recomputes_movement_and_remote_paths():
    from tkh.repair_assessment import audit_frontier
    report = audit_frontier(*frontier_fixture(), mode="bounded", max_old_fraction=.25)
    assert report["actual_moved_ids"] == ["b"]
    assert report["outside_frontier_old_nodes"] == 3
    assert report["outside_frontier_unchanged"] is True
    assert report["allowed_old_fraction"] == .25


@pytest.mark.parametrize("corruption", ["hidden_move", "budget", "new_as_old", "missing_new", "rebuild"])
def test_frontier_audit_rejects_violations_even_if_trace_claims_success(corruption):
    from tkh.repair_assessment import audit_frontier
    previous, old, current, new, trace = frontier_fixture()
    trace["boundary_checks"] = {"outside_region_unchanged": True}
    if corruption == "hidden_move":
        trace["moved_ids"] = []
        trace["allowed_old_nodes"] = ["a"]
    elif corruption == "budget":
        trace["old_edit_limit"] = 2
        trace["allowed_old_nodes"] = ["a", "b"]
    elif corruption == "new_as_old":
        trace["allowed_old_nodes"] = ["e"]
    elif corruption == "missing_new":
        trace["editable_ids"] = ["b"]
    else:
        trace["initial_builder_calls"] = 1
    with pytest.raises(ValueError):
        audit_frontier(previous, old, current, new, trace, mode="bounded", max_old_fraction=.25)


def test_coarse_parent_change_counts_as_an_old_path_edit():
    from tkh.repair_assessment import audit_frontier
    previous, old, current, new, trace = frontier_fixture()
    new["levels"][1]["groups"][0]["parent_id"] = "NEW"
    with pytest.raises(ValueError):
        audit_frontier(previous, old, current, new, trace, mode="bounded", max_old_fraction=.25)


def test_budgeted_metadata_donor_paths_are_audited_as_real_old_edits():
    from tkh.repair_assessment import audit_frontier
    previous, old, current, new, trace = frontier_fixture()
    previous['nodes'][1]['type'] = 'article'
    current['nodes'][1]['type'] = 'article'
    trace['novelty'] = {'accepted':True, 'moved_old_ids':['b']}
    report = audit_frontier(previous,old,current,new,trace,mode='bounded',max_old_fraction=.34)
    assert report['actual_moved_ids'] == ['b']
    assert report['budgeted_metadata_edits'] == 1


def test_perturbation_interval_is_conditional_on_five_seeds_not_a_temporal_ci():
    from tkh.repair_assessment import conditional_seed_interval
    report = conditional_seed_interval([.5, .6, .7, .8, .9])
    assert report["mean"] == .7
    assert report["low"] == pytest.approx(.5036756838522439)
    assert report["high"] == pytest.approx(.896324316147756)
    assert report["scope"] == "conditional mean over the fixed initial view and edge-deletion randomization; not chronological-process uncertainty"
    assert conditional_seed_interval([.8])["low"] is None


def test_prediction_freeze_is_verified_before_reference_is_opened(tmp_path):
    from tkh.repair_assessment import freeze_predictions, score_locked_predictions
    predictions = {"new_method": {"arm": {"q": ["a"]}}}
    freeze_predictions(tmp_path, predictions, {"policy": "new predictions"})
    (tmp_path / "predictions.json").write_text(json.dumps({"new_method": {"arm": {"q": ["b"]}}}))
    with pytest.raises(ValueError, match="prediction"):
        score_locked_predictions(tmp_path, {"nodes": []}, [], tmp_path / "absent-gold.json")


def test_new_predictions_are_scored_without_legacy_equality_or_historical_evidence(tmp_path):
    from tkh.repair_assessment import freeze_predictions, score_locked_predictions
    predictions = {"new_method": {"arm": {"q": ["a"]}}}
    freeze_predictions(tmp_path, predictions, {})
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"q": {"type": "A", "expected_methods": ["Alpha"]}}))
    graph = {"nodes": [{"id": "a", "surface_form": "Alpha", "type": "method"}]}
    scores = score_locked_predictions(tmp_path, graph, [{"question_id": "q", "question": "Alpha?", "type": "A"}], gold)
    assert scores["new_method"]["arm"]["tracks"]["A"]["metrics"]["30"]["hits"] == 1


def test_cached_channels_match_existing_formulas_and_do_not_mutate_inputs():
    from tkh.repair_assessment import prepare_channel_cache
    class Index:
        ids = ["a", "b", "author"]
        nodes = {"a": {"type": "method"}, "b": {"type": "dataset"}, "author": {"type": "author"}}
        def bm25_scores(self, text, context):
            return np.array([2., 1., 4.]) if context else np.array([1., 3., 4.])
        def tag_scores(self, text):
            return np.array([.4, .6, .9])
    passages = np.array([[1., 0.], [0., 1.], [1., 0.]])
    queries = np.array([[1., 0.]])
    old = passages.copy()
    cache = prepare_channel_cache(Index(), [{"question_id": "q", "question": "X", "type": "A"}], passages, queries)
    assert cache["q"]["bm25_own"].tolist() == [1., 3., 0.]
    assert cache["q"]["lexical"].tolist() == [2., 1., 0.]
    assert cache["q"]["dense"].tolist() == pytest.approx([1., np.exp(-20), 0.])
    assert np.array_equal(passages, old)


def test_exact_quotient_audit_detects_multiplicity_loss():
    from tkh.repair_assessment import audit_quotient
    view = {"hyperedges": [{"id": "e", "members": list("abc"), "year": 2022}]}
    hierarchy = {"levels": [{"index": 0, "groups": [{"id": "C", "members": list("abc")}]}]}
    artifact = {"levels": [{"index": 0, "hyperedges": [{"id": "e", "year": 2022, "members": ["C"], "counts": {"C": 3}}]}]}
    assert audit_quotient(view, hierarchy, artifact)[0]["endpoint_total"] == 3
    artifact["levels"][0]["hyperedges"][0]["counts"]["C"] = 1
    with pytest.raises(ValueError):
        audit_quotient(view, hierarchy, artifact)


def test_balanced_perturbation_rebuilds_the_same_builder_not_the_old_algorithm(tmp_path):
    from test_balanced_hierarchy import instance
    from tkh.balanced_hierarchy import build_balanced
    from tkh.repair_assessment import _fresh_perturbations
    from tkh.submission_assessment import _partition_digest
    view,x=instance()
    hierarchy=build_balanced(view,x,budgets=(6,18)).export()[0]
    report=_fresh_perturbations(view,hierarchy,x,{'budgets':[6,18],'gamma':.03,
        'seed':20261019,'initial_strategy':'balanced'},tmp_path)
    changed=deepcopy(view)
    removed=set(report['rows'][0]['removed_edge_ids'])
    changed['hyperedges']=[e for e in changed['hyperedges'] if e['id'] not in removed]
    expected=build_balanced(changed,x,budgets=(6,18)).export()[0]
    assert report['rows'][0]['partition_digest']==_partition_digest(expected)
    assert report['initial_strategy']=='balanced'
