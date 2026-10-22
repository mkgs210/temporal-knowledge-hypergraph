"""Search-index contracts derived from graph-only, hand-computable fixtures."""
import copy
import json

import numpy as np
import pytest
from scipy import sparse


def fixture_graph():
    return {
        "nodes": [
            {"id": "a", "type": "article", "surface_form": "Quantum molecular forces", "provenance": {"articles": [17]}},
            {"id": "b", "type": "article", "surface_form": "Crystal lattice energy", "provenance": {"articles": [23]}},
            {"id": "m", "type": "method", "surface_form": "Learning atom interactions", "aliases": ["Atomic regressor"], "provenance": {"articles": [17, 23]}},
            {"id": "n", "type": "component", "surface_form": "Smooth potential", "provenance": {"articles": [17]}},
            {"id": "u", "type": "author", "surface_form": "Unique Author", "provenance": {"articles": [17]}},
            {"id": "z", "type": "dataset", "surface_form": "Isolated database"},
        ],
        "hyperedges": [{"id": "e", "members": ["m", "n", "u"], "relation_type": "uses", "weight": 2}],
    }


def test_native_incidence_step_obeys_weighted_hand_computation():
    from tkh.tag_graph_index import IncidenceLayer
    layer = IncidenceLayer(["a", "b", "c"], [["a", "b"], ["b", "c"]], [1, 3], name="native")
    assert sparse.isspmatrix_csr(layer.matrix)
    assert layer.matrix.shape == (3, 2)
    assert layer.matrix.nnz == 4
    # At b, select the two edges with probabilities 1/4 and 3/4,
    # then distribute uniformly to each edge's two members.
    np.testing.assert_allclose(layer.step([0, 1, 0], [1, 0, 0]), [.125, .5, .375])
    json.dumps(layer.describe(), allow_nan=False)


def test_isolated_node_mass_returns_to_restart_even_without_any_edges():
    from tkh.tag_graph_index import IncidenceLayer
    layer = IncidenceLayer(["a", "b", "z"], [["a", "b"]])
    np.testing.assert_allclose(layer.step([.2, .3, .5], [.8, .2, 0]), [.65, .35, 0])
    empty = IncidenceLayer(["a", "b"], [])
    np.testing.assert_allclose(empty.step([0, 2], [2, 1]), [4 / 3, 2 / 3])


@pytest.mark.parametrize("ids,edges,weights", [
    (["a", "a"], [["a"]], None),
    (["a"], [["missing"]], None),
    (["a"], [["a", "a"]], None),
    (["a"], [[]], None),
    (["a"], [["a"]], [-1]),
    (["a"], [["a"]], [float("nan")]),
    (["a"], [["a"]], []),
])
def test_layer_rejects_invalid_memberships_and_weights(ids, edges, weights):
    from tkh.tag_graph_index import IncidenceLayer
    with pytest.raises(ValueError):
        IncidenceLayer(ids, edges, weights)


def test_bm25_stems_explicit_aliases_and_has_no_unmatched_positive_scores():
    from tkh.tag_graph_index import TagGraphIndex
    index = TagGraphIndex(fixture_graph())
    scores = index.bm25_scores("regressors", context=False)
    assert scores[index.ids.index("m")] > 0
    assert np.count_nonzero(scores) == 1
    np.testing.assert_array_equal(index.bm25_scores("qzxvunknown"), np.zeros(6))
    np.testing.assert_array_equal(index.bm25_scores("the and"), np.zeros(6))


def test_context_tracks_multiple_sources_and_native_support_without_mutation():
    from tkh.tag_graph_index import TagGraphIndex
    graph = fixture_graph()
    before = copy.deepcopy(graph)
    index = TagGraphIndex(graph)
    docs = {doc["id"]: doc for doc in index.documents}
    assert index.ids == ["a", "b", "m", "n", "u", "z"]
    assert "Quantum molecular forces" in docs["m"]["text"]
    assert "Crystal lattice energy" in docs["m"]["text"]
    assert "Smooth potential" in docs["m"]["text"]
    assert set(docs["m"]["source_article_ids"]) == {17, 23}
    assert set(docs["m"]["context_node_ids"]) == {"a", "b", "n"}
    assert docs["m"]["context_edge_ids"] == ["e"]
    assert "Unique Author" not in docs["m"]["text"]
    assert docs["u"]["text"] == docs["u"]["own_text"]
    assert docs["u"]["tag_ids"] == []
    assert index.bm25_scores("Quantum", context=False)[index.ids.index("n")] == 0
    assert index.bm25_scores("Quantum", context=True)[index.ids.index("n")] > 0
    assert index.layers["native"].matrix.nnz == 3
    source = index.layers["sources"]
    assert source.matrix.shape == (6, 2)
    assert source.matrix.nnz == 6
    assert graph == before
    json.dumps(index.describe(), allow_nan=False)
    json.dumps(index.tag_records, allow_nan=False)


def test_tags_are_extractive_queryable_and_do_not_fabricate_ties():
    from tkh.tag_graph_index import TagGraphIndex
    graph = {"nodes": [
        {"id": "a", "type": "article", "surface_form": "Running molecular models"},
        {"id": "b", "type": "method", "surface_form": "Molecular runners"},
        {"id": "c", "type": "dataset", "surface_form": "Crystal energy"},
    ], "hyperedges": []}
    index = TagGraphIndex(graph)
    assert index.tag_records
    assert all(record["idf"] > 0 and record["members"] for record in index.tag_records)
    assert all(record["support"] for record in index.tag_records)
    scores = index.tag_scores("runs")
    assert scores[index.ids.index("a")] > 0
    assert np.isfinite(scores).all() and (scores >= 0).all()
    assert index.query_tags("runs")
    np.testing.assert_array_equal(index.tag_scores("qzxvunknown"), np.zeros(3))
    assert index.query_tags("qzxvunknown") == []
    assert all(len(doc["tag_ids"]) <= 12 for doc in index.documents)


def test_empty_vocabulary_is_a_zero_channel():
    from tkh.tag_graph_index import TagGraphIndex
    index = TagGraphIndex({"nodes": [{"id": "u", "type": "author", "surface_form": "the and"}], "hyperedges": []})
    assert index.tag_records == []
    np.testing.assert_array_equal(index.tag_scores("anything"), [0])
    np.testing.assert_array_equal(index.bm25_scores("anything"), [0])


def test_graph_rejects_duplicate_members_and_invalid_native_weight():
    from tkh.tag_graph_index import TagGraphIndex
    graph = fixture_graph()
    graph["hyperedges"][0]["members"].append("m")
    with pytest.raises(ValueError):
        TagGraphIndex(graph)
    graph = fixture_graph()
    graph["hyperedges"][0]["weight"] = float("inf")
    with pytest.raises(ValueError):
        TagGraphIndex(graph)


def test_tag_records_keep_original_surface_phrases_and_stemmed_keys():
    from tkh.tag_graph_index import TagGraphIndex
    index = TagGraphIndex({"nodes": [
        {"id": "a", "type": "article", "surface_form": "Running quantum models"},
        {"id": "b", "type": "method", "surface_form": "Crystal energy"},
    ], "hyperedges": []})
    running = next(record for record in index.tag_records if record["key"] == "run")
    assert running["text"] == "Running"
    assert running["support"][0]["phrase"] == "Running"


def test_source_mapping_rejects_ambiguous_ownership_and_marks_missing_titles():
    from tkh.tag_graph_index import TagGraphIndex
    graph = fixture_graph()
    graph["nodes"][1]["provenance"]["articles"] = [17]
    with pytest.raises(ValueError, match="(?i)ambiguous"):
        TagGraphIndex(graph)
    graph = fixture_graph()
    graph["nodes"][2]["provenance"]["articles"].append(999)
    index = TagGraphIndex(graph)
    doc = next(doc for doc in index.documents if doc["id"] == "m")
    assert doc["unresolved_source_article_ids"] == [999]
    assert "999" not in doc["text"]


def test_context_is_bounded_deterministically_without_truncating_own_text():
    from tkh.tag_graph_index import TagGraphIndex
    graph = {"nodes": [{"id": "a", "type": "method", "surface_form": "Long " * 1000}] + [
        {"id": f"n{i:02}", "type": "component", "surface_form": f"Potential {i}"} for i in range(10)],
        "hyperedges": [{"id": "e", "relation_type": "uses", "members": ["a"] + [f"n{i:02}" for i in range(10)]}]}
    index = TagGraphIndex(graph)
    doc = index.documents[0]
    assert doc["context_node_ids"] == [f"n{i:02}" for i in range(8)]
    assert doc["text"].startswith(doc["own_text"])
    assert doc["own_text"].count("Long") == 1000


@pytest.mark.parametrize("mass,restart", [([float("nan"), 0], [1, 0]), ([-1, 2], [1, 0]), ([1], [1, 0]), ([1, 0], [0, 0])])
def test_incidence_step_rejects_invalid_masses_or_missing_dangling_restart(mass, restart):
    from tkh.tag_graph_index import IncidenceLayer
    with pytest.raises(ValueError):
        IncidenceLayer(["a", "b"], []).step(mass, restart)
