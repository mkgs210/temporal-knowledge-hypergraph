"""Small independent calculations for scientific reporting and routing costs."""
from copy import deepcopy
import math

import numpy as np
import pytest


def state_fixture():
    from tkh.hierarchy import State
    nodes = [{"id": nid, "type": "method", "surface_form": name, "observed_at": 2022,
              "provenance": {"articles": sources}}
             for nid, name, sources in [("a", "Graph model", [1]), ("b", "Graph network", [1]),
                                        ("c", "Atomic force", [1, 2]), ("d", "Atomic energy", [2])]]
    edges = [{"id": "e1", "members": ["a", "b"], "relation_type": "uses", "year": 2022,
              "provenance": {"article_id": 1, "article_year": 2022}},
             {"id": "e2", "members": ["b", "c", "d"], "relation_type": "evaluated_on", "year": 2022,
              "provenance": {"article_id": 2, "article_year": 2022}}]
    return State({"cutoff": 2022, "nodes": nodes, "hyperedges": edges},
                 np.array([[1., 0, 0], [.9, .1, 0], [0, 1, 0], [0, 0, 1]]),
                 {"a": "g1", "b": "g1", "c": "g2", "d": "g3"},
                 {"g1": "C1", "g2": "C2", "g3": "C2"},
                 {"budgets": [2, 3]}, [], {"initial_builder_calls": 1})


def test_measure_reports_native_edge_entropy_source_capture_and_explicit_label_limits():
    from tkh.hierarchy_evaluation import measure
    report = measure(state_fixture())
    assert report["snapshot"]["nodes"] == 4
    assert report["snapshot"]["hyperedges"] == 2
    assert report["snapshot"]["node_types"] == {"method": 4}
    assert report["snapshot"]["relation_types"] == {"uses": 1, "evaluated_on": 1}
    assert report["snapshot"]["arity"]["mean"] == 2.5
    assert report["snapshot"]["isolated_nodes"] == 0
    coarse = report["levels"][0]
    h = -(math.log(1/3)/3 + 2*math.log(2/3)/3) / math.log(3)
    assert coarse["mean_native_edge_entropy"] == pytest.approx(h / 2)
    assert coarse["internal_edge_fraction"] == .5
    assert coarse["source_purity_node_weighted"] == .875
    assert coarse["cross_source_groups"] == 1
    assert coarse["group_sizes"]["values"] == [2, 2]
    assert coarse["child_counts"]["values"] == [1, 2]
    assert report["labels"]["semantic_faithfulness"]["status"] == "not_evaluated"
    assert report["labels"]["extractive_evidence_coverage"] == 1
    assert report["coherence"]["status"] == "not_evaluated"


def test_missing_edges_produce_null_measurements_not_fake_success():
    from tkh.hierarchy_evaluation import measure
    state = state_fixture()
    state.view["hyperedges"] = []
    report = measure(state)
    assert report["snapshot"]["isolated_nodes"] == 4
    assert report["levels"][0]["mean_native_edge_entropy"] is None
    assert report["levels"][0]["internal_edge_fraction"] is None
    assert report["snapshot"]["arity"]["mean"] is None


def test_temporal_ari_distinguishes_persistent_ids_from_membership_similarity():
    from tkh.hierarchy_evaluation import compare_temporal
    before = state_fixture()
    after = deepcopy(before)
    after.view["cutoff"] = 2023
    after.assignment = {"a": "g4", "b": "g4", "c": "g5", "d": "g6"}
    after.parents = {"g4": "C2", "g5": "C1", "g6": "C1"}
    report = compare_temporal(before, after)
    assert report["shared_old_nodes"] == 4
    assert report["levels"][0]["ari_shared_old_nodes"] == 1
    assert report["levels"][0]["persistent_id_movement"] == 4
    assert report["levels"][1]["ari_shared_old_nodes"] == 1
    assert report["levels"][1]["persistent_id_movement"] == 4
    assert report["new_nodes"] == 0
    assert report["uncertainty"]["status"] == "not_estimated"


def test_temporal_ari_uses_only_shared_nodes_and_returns_na_for_too_few():
    from tkh.hierarchy_evaluation import compare_temporal
    before = state_fixture()
    after = deepcopy(before)
    after.view["nodes"] = after.view["nodes"][:1]
    after.assignment = {"a": "g1"}
    report = compare_temporal(before, after)
    assert report["shared_old_nodes"] == 1
    assert report["removed_nodes"] == 3
    assert report["levels"][0]["ari_shared_old_nodes"] is None


def test_independent_pair_coherence_and_random_partition_null_are_reproducible():
    from tkh.hierarchy_evaluation import compare_coherence
    state = state_fixture()
    x = np.array([[1., 0], [1, 0], [0, 1], [0, 1]])
    report = compare_coherence(state, x, seed=19, null_samples=40, family="independent test family")
    assert report == compare_coherence(state, x, seed=19, null_samples=40, family="independent test family")
    assert report["levels"][0]["weighted_pairwise_cosine"] == 1
    assert report["levels"][0]["within_group_pairs"] == 2
    assert report["levels"][1]["within_group_pairs"] == 1
    null = report["levels"][0]["null_values"]
    assert len(null) == 40
    assert set(null) <= {0., 1.}
    assert report["null_policy"] == "preserve_group_sizes_and_type_composition"
    assert "not a confidence interval" in report["interval_interpretation"]


def test_coherence_does_not_recycle_construction_vectors_and_singletons_are_na():
    from tkh.hierarchy_evaluation import compare_coherence
    state = state_fixture()
    with pytest.raises(ValueError, match="construction"):
        compare_coherence(state, state.vectors)
    state.assignment = dict(zip("abcd", ["g1", "g2", "g3", "g4"]))
    state.parents = dict(zip(["g1", "g2", "g3", "g4"], ["C1", "C1", "C2", "C2"]))
    report = compare_coherence(state, np.eye(4), null_samples=2)
    assert report["levels"][1]["weighted_pairwise_cosine"] is None
    assert report["levels"][1]["monte_carlo_p_greater"] is None


def routing_fixture():
    # Group A's average loses the best individual match a; B's average wins.
    hierarchy = {"levels": [
        {"index": 0, "groups": [{"id": "A", "parent_id": None, "members": ["a", "b"]},
                                    {"id": "B", "parent_id": None, "members": ["c", "d"]}]},
        {"index": 1, "groups": [{"id": "aa", "parent_id": "A", "members": ["a", "b"]},
                                    {"id": "bb", "parent_id": "B", "members": ["c", "d"]}]},
    ]}
    x = np.array([[1., 0], [-.8, .6], [.8, .6], [.8, -.6]])
    return hierarchy, x


def test_routing_exposes_candidate_loss_and_counts_centroid_overhead():
    from tkh.hierarchy_evaluation import routing
    hierarchy, x = routing_fixture()
    report = routing(hierarchy, list("abcd"), x, [[1., 0]], beams=(1, 1), top_k=1, query_ids=["q"])[0]
    assert report["query_id"] == "q"
    assert report["flat"]["ranked"] == [{"id": "a", "score": 1.}]
    assert report["hierarchy"]["ranked"] == [{"id": "c", "score": .8}]
    assert report["flat_topk_recall"] == 0
    assert report["hierarchy"]["work"] == {"node_dot_products": 2, "centroid_dot_products": 3, "total_dot_products": 5}
    assert report["flat"]["work"]["total_dot_products"] == 4
    assert report["hierarchy"]["candidate_count"] == 2


def test_full_beam_matches_flat_scores_ranking_and_ties():
    from tkh.hierarchy_evaluation import routing
    hierarchy, x = routing_fixture()
    report = routing(hierarchy, list("abcd"), x, [[1., 0]], beams=(2, 2), top_k=4)[0]
    assert report["flat"]["ranked"] == report["hierarchy"]["ranked"]
    assert [n["id"] for n in report["flat"]["ranked"]] == list("acdb")
    assert report["flat_topk_recall"] == 1


def test_routing_rejects_vector_misalignment_and_zero_query():
    from tkh.hierarchy_evaluation import routing
    hierarchy, x = routing_fixture()
    for ids, xx, q in [(list("abc"), x, [[1., 0]]), (list("abcd"), x, [[0., 0]]),
                       (list("abce"), x, [[1., 0]])]:
        with pytest.raises(ValueError):
            routing(hierarchy, ids, xx, q)
