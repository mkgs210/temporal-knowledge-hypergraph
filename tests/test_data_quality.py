"""Known counterexamples to provenance and temporal consistency, without gold."""
import copy
import json
import subprocess
import sys

import pytest


def consistent_graph():
    return {"nodes": [
        {"id": "p", "type": "article", "surface_form": "A paper", "year": 2020,
         "origin_year": 2020, "first_seen_year": 2020, "last_seen_year": 2020,
         "provenance": {"articles": [1]}},
        {"id": "m", "type": "method", "surface_form": "An older idea", "year": 1990,
         "origin_year": 1990, "first_seen_year": 2020, "last_seen_year": 2020,
         "provenance": {"articles": [1]}},
    ], "hyperedges": [
        {"id": "e", "relation_type": "presents", "members": ["p", "m"], "year": 2020,
         "provenance": {"article_id": 1, "article_year": 2020}},
    ]}


def test_old_origin_is_valid_and_inputs_are_not_repaired():
    from tkh.data_quality import check_consistency
    graph = consistent_graph()
    before = copy.deepcopy(graph)
    assert check_consistency(graph) == []
    assert graph == before


@pytest.mark.parametrize("field,value,code,expected", [
    ("first_seen_year", 1997, "first_seen_vs_sources", 2020),
    ("last_seen_year", 2023, "last_seen_vs_sources", 2020),
    ("year", 2020, "node_year_rule", 1990),
])
def test_dates_that_contradict_declared_semantics_are_errors(field, value, code, expected):
    from tkh.data_quality import check_consistency
    graph = consistent_graph()
    graph["nodes"][1][field] = value
    flags = check_consistency(graph)
    row = next(f for f in flags if f["code"] == code and f["node_id"] == "m")
    assert row["severity"] == "error"
    assert row["observed"] == value
    assert row["expected"] == expected


def test_edge_source_must_be_known_and_its_publication_year_must_match():
    from tkh.data_quality import check_consistency
    graph = consistent_graph()
    graph["hyperedges"][0]["provenance"]["article_year"] = 2021
    assert any(f["code"] == "edge_source_year" and f["severity"] == "error" for f in check_consistency(graph))
    graph["hyperedges"][0]["provenance"]["article_id"] = 99
    assert any(f["code"] == "unknown_edge_source" and f["article_id"] == 99 for f in check_consistency(graph))


def test_each_incident_edge_source_must_be_declared_on_member():
    from tkh.data_quality import check_consistency
    graph = consistent_graph()
    graph["nodes"][1]["provenance"]["articles"] = []
    row = next(f for f in check_consistency(graph) if f["code"] == "missing_member_source")
    assert row["node_id"] == "m"
    assert row["edge_id"] == "e"
    assert row["article_id"] == 1
    assert row["severity"] == "error"


def test_different_fact_year_is_warning_not_automatic_rejection():
    from tkh.data_quality import check_consistency
    graph = consistent_graph()
    graph["hyperedges"][0]["year"] = 1997
    flags = check_consistency(graph)
    assert {f["code"] for f in flags} == {"fact_year_vs_source", "edge_before_member_observation"}
    assert all(f["severity"] == "warning" for f in flags)


def test_cli_rejects_inconsistent_export_and_preserves_it(tmp_path):
    graph = consistent_graph()
    graph["nodes"][1]["first_seen_year"] = 1997
    path = tmp_path / "graph.json"
    original = json.dumps(graph).encode()
    path.write_bytes(original)
    out = tmp_path / "check"
    result = subprocess.run([sys.executable, "-m", "tkh.data_quality", "--graph", str(path), "--out", str(out)], capture_output=True)
    assert result.returncode == 1
    assert path.read_bytes() == original
    report = json.loads((out / "issues.json").read_text())
    assert report["status"] == "inconsistent"
    assert report["counts"]["first_seen_vs_sources"] == 1
    assert (out / "issues.csv").is_file()


def test_cli_accepts_clean_export(tmp_path):
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(consistent_graph()))
    result = subprocess.run([sys.executable, "-m", "tkh.data_quality", "--graph", str(path), "--out", str(tmp_path / "check")], capture_output=True)
    assert result.returncode == 0, result.stderr


def test_source_article_id_cannot_name_two_different_papers():
    from tkh.data_quality import check_consistency
    graph = consistent_graph()
    second = copy.deepcopy(graph["nodes"][0])
    second.update(id="other_paper", surface_form="Another paper")
    graph["nodes"].append(second)
    assert any(f["code"] == "ambiguous_article_id" and f["severity"] == "error"
               for f in check_consistency(graph))


@pytest.mark.parametrize("sources,edge_source", [("1", "1"), ([True], True), ([[1]], [1])])
def test_malformed_source_ids_and_containers_cannot_pass(sources, edge_source):
    from tkh.data_quality import check_consistency
    graph = consistent_graph()
    for node in graph["nodes"]:
        node["provenance"]["articles"] = sources
    graph["hyperedges"][0]["provenance"]["article_id"] = edge_source
    assert any(f["code"] == "invalid_node_sources" and f["severity"] == "error"
               for f in check_consistency(graph))
