"""Cutoff-safe records, not membership-only views of the final export."""
import copy
import json
from pathlib import Path

import pytest


def graph_fixture():
    def node(nid, kind, text, first, articles):
        return {"id": nid, "type": kind, "surface_form": text,
                "first_seen_year": first, "last_seen_year": 2025, "year": first,
                "aliases": ["Unversioned future synonym"],
                "provenance": {"articles": articles, "source": "example",
                               "extracted_at": "2026-08-01", "status": "DONE"}}
    return {"nodes": [node("p1", "article", "Old paper", 2022, [1]),
                      node("p2", "article", "New paper", 2023, [2]),
                      node("a", "method", "Shared method", 1990, [1, 2]),
                      node("b", "task", "New task", 2023, [2])],
            "hyperedges": [
                {"id": "e1", "relation_type": "presents", "members": ["p1", "a"],
                 "year": 2022, "provenance": {"article_id": 1, "article_year": 2022},
                 "attributes": {"evidence": "text"}},
                {"id": "e2", "relation_type": "addresses", "members": ["a", "b"],
                 "year": 1990, "provenance": {"article_id": 2, "article_year": 2023}},
                {"id": "e3", "relation_type": "addresses", "members": ["p1", "b"],
                 "year": 2022, "provenance": {"article_id": 1, "article_year": 2022}},
                {"id": "e4", "relation_type": "cites", "members": ["p1", "a"],
                 "year": 2023, "provenance": {"article_id": 1, "article_year": 2022}},
            ]}


def test_cutoff_removes_future_attributes_and_keeps_original_graph_unchanged():
    from tkh.temporal_input import temporal_view
    graph = graph_fixture()
    original = copy.deepcopy(graph)
    view = temporal_view(graph, 2022)
    assert view["nodes"] == [
        {"id": "a", "type": "method", "surface_form": "Shared method", "observed_at": 2022,
         "provenance": {"articles": [1]}},
        {"id": "p1", "type": "article", "surface_form": "Old paper", "observed_at": 2022,
         "provenance": {"articles": [1]}},
    ]
    assert view["hyperedges"] == [original["hyperedges"][0]]
    assert view["cutoff"] == 2022
    assert view["limitations"]  # Historical wording cannot be reconstructed.
    assert graph == original
    view["hyperedges"][0]["attributes"]["evidence"] = "mutated"
    assert graph == original


def test_no_hyperedge_is_shortened_when_a_member_is_not_yet_visible():
    from tkh.temporal_input import temporal_view
    graph = graph_fixture()
    view = temporal_view(graph, 2022)
    assert [edge["id"] for edge in view["hyperedges"]] == ["e1"]
    later = temporal_view(graph, 2023)
    assert [edge["id"] for edge in later["hyperedges"]] == ["e1", "e2", "e3", "e4"]


@pytest.mark.parametrize("problem", ["unknown_node_source", "empty_sources", "missing_first_seen",
                                     "unknown_edge_source", "missing_edge_year", "source_year_conflict"])
def test_missing_or_conflicting_source_metadata_fails_closed(problem):
    from tkh.temporal_input import temporal_view
    graph = graph_fixture()
    if problem == "unknown_node_source":
        graph["nodes"][2]["provenance"]["articles"] = [999]
    elif problem == "empty_sources":
        graph["nodes"][2]["provenance"]["articles"] = []
    elif problem == "missing_first_seen":
        del graph["nodes"][2]["first_seen_year"]
    elif problem == "unknown_edge_source":
        graph["hyperedges"][0]["provenance"]["article_id"] = 999
    elif problem == "missing_edge_year":
        del graph["hyperedges"][0]["provenance"]["article_year"]
    else:
        graph["hyperedges"][0]["provenance"]["article_year"] = 2021
    with pytest.raises(ValueError):
        temporal_view(graph, 2022)


def test_duplicate_ids_and_ambiguous_paper_sources_are_rejected():
    from tkh.temporal_input import temporal_view
    graph = graph_fixture()
    graph["nodes"].append(copy.deepcopy(graph["nodes"][0]))
    with pytest.raises(ValueError):
        temporal_view(graph, 2022)
    graph = graph_fixture()
    graph["nodes"][1]["provenance"]["articles"] = [1]
    with pytest.raises(ValueError):
        temporal_view(graph, 2022)


def test_delta_carries_new_nodes_and_new_facts_about_existing_nodes():
    from tkh.temporal_input import temporal_view, delta
    graph = graph_fixture()
    before, after = temporal_view(graph, 2022), temporal_view(graph, 2023)
    changes = delta(before, after)
    assert [n["id"] for n in changes["nodes_added"]] == ["b", "p2"]
    assert [n["id"] for n in changes["nodes_changed"]] == ["a"]
    assert changes["nodes_changed"][0]["provenance"] == {"articles": [1, 2]}
    assert [e["id"] for e in changes["hyperedges_added"]] == ["e2", "e3", "e4"]
    assert delta(after, after) == {"nodes_added": [], "nodes_changed": [], "hyperedges_added": []}


@pytest.mark.parametrize("problem", ["node_removed", "edge_removed", "edge_changed", "duplicate_node", "duplicate_edge"])
def test_delta_rejects_non_append_only_or_ambiguous_inputs(problem):
    from tkh.temporal_input import temporal_view, delta
    before = temporal_view(graph_fixture(), 2023)
    after = copy.deepcopy(before)
    if problem == "node_removed":
        after["nodes"].pop()
    elif problem == "edge_removed":
        after["hyperedges"].pop()
    elif problem == "edge_changed":
        after["hyperedges"][0]["attributes"] = {"changed": True}
    elif problem == "duplicate_node":
        after["nodes"].append(copy.deepcopy(after["nodes"][0]))
    else:
        before["hyperedges"].append(copy.deepcopy(before["hyperedges"][0]))
    with pytest.raises(ValueError):
        delta(before, after)


def test_real_export_regression_counts_and_deterministic_views():
    from tkh.temporal_input import temporal_view, delta
    path = Path(__file__).resolve().parents[1] / "data" / "tkh_collection10.json"
    graph = json.loads(path.read_text())
    previous = None
    for year, nn, ne in [(2022, 2137, 526), (2023, 3376, 820), (2024, 4094, 983),
                         (2025, 4987, 1239), (2026, 5798, 1429)]:
        view = temporal_view(graph, year)
        assert (len(view["nodes"]), len(view["hyperedges"])) == (nn, ne)
        assert temporal_view(graph, year) == view
        if previous:
            delta(previous, view)  # Real yearly sequence must satisfy the append-only contract.
        previous = view
