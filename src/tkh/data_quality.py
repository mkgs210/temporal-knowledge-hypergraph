"""Strict checks against the export's declared date/provenance semantics.

Errors describe internal record inconsistencies, not incorrect scientific claims.
Warnings are review flags, not automatic reasons to delete data. No repairs.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from .corpus import validate_graph


def check_consistency(graph):
    validate_graph(graph)
    nodes = {n["id"]: n for n in graph["nodes"]}
    issues = []

    def flag(code, severity, observed, expected, node_id=None, edge_id=None, article_id=None):
        issues.append({"code": code, "severity": severity, "node_id": node_id, "edge_id": edge_id,
                       "article_id": article_id, "observed": observed, "expected": expected})

    # Validate before lookup: strings are iterable, and True compares equal to 1.
    source_lists, articles = {}, {}
    for node in graph["nodes"]:
        provenance = node.get("provenance")
        sources = provenance.get("articles") if isinstance(provenance, dict) else None
        if not isinstance(sources, list) or not all(type(aid) is int for aid in sources):
            flag("invalid_node_sources", "error", sources, "list of integer article IDs", node_id=node["id"])
            source_lists[node["id"]] = []
            continue
        source_lists[node["id"]] = sources
        if node["type"] == "article":
            for aid in sources:
                if aid in articles:
                    flag("ambiguous_article_id", "error", [articles[aid]["id"], node["id"]],
                         "one article node per source ID", node_id=node["id"], article_id=aid)
                else:
                    articles[aid] = node

    for n in graph["nodes"]:
        for key in ("year", "first_seen_year", "last_seen_year"):
            if type(n.get(key)) is not int:
                flag("invalid_node_year", "error", {key: n.get(key)}, "integer year", node_id=n["id"])
        first, last, origin = n.get("first_seen_year"), n.get("last_seen_year"), n.get("origin_year")
        if origin is not None and type(origin) is not int:
            flag("invalid_origin_year", "error", origin, "integer or null", node_id=n["id"])
        if type(first) is int and type(last) is int and first > last:
            flag("first_after_last", "error", first, {"not_later_than": last}, node_id=n["id"])
        expected_year = origin if origin is not None else first
        if type(expected_year) is int and n.get("year") != expected_year:
            flag("node_year_rule", "error", n.get("year"), expected_year, node_id=n["id"])
        if type(origin) is int and type(first) is int and origin > first:
            flag("origin_after_observation", "warning", origin, {"not_later_than": first}, node_id=n["id"])
        sources = source_lists[n["id"]]
        if not sources:
            flag("empty_node_sources", "error", sources, "nonempty list of source article IDs", node_id=n["id"])
        for aid in sources:
            if aid not in articles:
                flag("unknown_node_source", "error", aid, "existing article ID", node_id=n["id"], article_id=aid)
        source_years = [articles[aid].get("year") for aid in sources if aid in articles]
        if source_years and len(source_years) == len(sources) and all(type(y) is int for y in source_years):
            if type(first) is int and first != min(source_years):
                flag("first_seen_vs_sources", "error", first, min(source_years), node_id=n["id"])
            if type(last) is int and last != max(source_years):
                flag("last_seen_vs_sources", "error", last, max(source_years), node_id=n["id"])

    for edge in graph["hyperedges"]:
        provenance = edge.get("provenance")
        if not isinstance(provenance, dict):
            flag("invalid_edge_provenance", "error", provenance, "object with integer article_id/article_year", edge_id=edge["id"])
            provenance = {}
        aid, source_year, fact_year = provenance.get("article_id"), provenance.get("article_year"), edge.get("year")
        known_source = type(aid) is int and aid in articles
        if not known_source:
            flag("unknown_edge_source", "error", aid, "existing article ID", edge_id=edge["id"], article_id=aid)
        elif source_year != articles[aid].get("year"):
            flag("edge_source_year", "error", source_year, articles[aid].get("year"), edge_id=edge["id"], article_id=aid)
        for key, value in (("year", fact_year), ("article_year", source_year)):
            if type(value) is not int:
                flag("invalid_edge_year", "error", {key: value}, "integer year", edge_id=edge["id"])
        if type(fact_year) is int and type(source_year) is int and fact_year != source_year:
            flag("fact_year_vs_source", "warning", fact_year, {"source_year": source_year, "equality_required": False},
                 edge_id=edge["id"], article_id=aid)
        later_members = []
        for nid in sorted(set(edge["members"])):
            member = nodes[nid]
            sources = source_lists[nid]
            if known_source and aid not in sources:
                flag("missing_member_source", "error", sources, {"must_include": aid},
                     node_id=nid, edge_id=edge["id"], article_id=aid)
            if type(fact_year) is int and type(member.get("first_seen_year")) is int and member["first_seen_year"] > fact_year:
                later_members.append(nid)
        if later_members:
            flag("edge_before_member_observation", "warning", {"edge_year": fact_year, "members": later_members},
                 "review snapshot eligibility of all endpoints", edge_id=edge["id"], article_id=aid)
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/tkh_collection10.json"))
    parser.add_argument("--out", type=Path, required=True, help="New directory for issues.json, issues.csv and manifest")
    args = parser.parse_args()
    raw = args.graph.read_bytes()
    graph = json.loads(raw)
    issues = check_consistency(graph)
    errors = [i for i in issues if i["severity"] == "error"]
    report = {"status": "inconsistent" if errors else "consistent_under_checked_rules",
              "errors": len(errors), "warnings": len(issues) - len(errors),
              "counts": dict(sorted(Counter(i["code"] for i in issues).items())),
              "distinct_error_nodes": len({i["node_id"] for i in errors if i["node_id"] is not None}),
              "distinct_error_edges": len({i["edge_id"] for i in errors if i["edge_id"] is not None}),
              "scope": "record date/source consistency, no scientific truth validation or repairs", "issues": issues}
    if args.graph.read_bytes() != raw:
        raise ValueError("Input changed during validation")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "issues.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["code", "severity", "node_id", "edge_id", "article_id", "observed", "expected"]
    with (args.out / "issues.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in issues:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    (args.out / "manifest.json").write_text(json.dumps({"graph_sha256": hashlib.sha256(raw).hexdigest(),
                                                        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                        "schema_validator_sha256": hashlib.sha256(Path(__file__).with_name("corpus.py").read_bytes()).hexdigest(),
                                                        "repairs": False}, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['errors']} errors, {report['warnings']} warnings; details: {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
