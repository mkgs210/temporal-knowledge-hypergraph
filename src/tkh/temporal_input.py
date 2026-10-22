"""Conservative, derived temporal inputs; source export is never repaired.

Publication years provide retrospective availability, not a historical database
log. Undated aliases and final metadata are deliberately excluded. The wording
of surface forms and edge attributes has no historical versions in this export;
this remaining limitation is carried in every returned view.
"""
from copy import deepcopy


POLICY = "conservative-publication-v1"
LIMITATIONS = [
    "Retrospective publication-year view; this is not the database ingestion history.",
    "Surface-form wording and edge attributes lack historical versions; their exact historical wording cannot be established.",
    "Year-level dates cannot establish availability by a particular month, including February 2026.",
    "Undated aliases, final last_seen metadata and ingestion/extraction timestamps are excluded.",
    "Known raw provenance/date contradictions are not repaired by this derived view.",
]


def _index(records, kind):
    result = {}
    for record in records:
        key = record.get("id")
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Invalid or duplicate {kind} ID: {key!r}")
        result[key] = record
    return result


def _year(value, context):
    if type(value) is not int:
        raise ValueError(f"Missing or invalid year: {context}")
    return value


def _sources(node):
    sources = node.get("provenance", {}).get("articles")
    if not isinstance(sources, list) or not sources or any(type(a) is not int for a in sources):
        raise ValueError(f"Missing or invalid source articles: {node['id']}")
    return sorted(set(sources))


def temporal_view(graph, cutoff):
    """Return deterministic, cutoff-filtered records fit for downstream inputs.

    Node availability is max(first_seen_year, earliest declared source year).
    Edges require both their fact year and source publication year at or before
    cutoff and ALL original endpoints visible. No edge is shortened. Unknown or
    ambiguous source metadata rejects the input rather than assuming visibility.
    """
    _year(cutoff, "cutoff")
    nodes = _index(graph["nodes"], "node")
    edges = _index(graph["hyperedges"], "hyperedge")
    publication = {}
    owners = {}
    for node in nodes.values():
        if node["type"] != "article":
            continue
        year = _year(node.get("first_seen_year"), node["id"])
        for source in _sources(node):
            if source in owners:
                raise ValueError(f"Ambiguous source article ID {source}: {owners[source]}, {node['id']}")
            owners[source] = node["id"]
            publication[source] = year

    visible = {}
    for nid, node in sorted(nodes.items()):
        sources = _sources(node)
        unknown = set(sources) - publication.keys()
        if unknown:
            raise ValueError(f"Unknown source publication metadata for {nid}: {sorted(unknown)}")
        observed = max(_year(node.get("first_seen_year"), nid), min(publication[a] for a in sources))
        if observed <= cutoff:
            visible[nid] = {"id": nid, "type": node["type"], "surface_form": node["surface_form"],
                            "observed_at": observed,
                            "provenance": {"articles": [a for a in sources if publication[a] <= cutoff]}}

    accepted = []
    for eid, edge in sorted(edges.items()):
        fact_year = _year(edge.get("year"), eid)
        provenance = edge.get("provenance", {})
        source = provenance.get("article_id")
        if type(source) is not int or source not in publication:
            raise ValueError(f"Unknown source publication metadata for {eid}: {source!r}")
        source_year = _year(provenance.get("article_year"), eid)
        if source_year != publication[source]:
            raise ValueError(f"Conflicting source publication year for {eid}: {source_year} != {publication[source]}")
        members = edge.get("members")
        if not isinstance(members, list) or not members or any(not isinstance(m, str) for m in members):
            raise ValueError(f"Missing or invalid hyperedge members: {eid}")
        if set(members) - nodes.keys():
            raise ValueError(f"Unknown members in {eid}: {sorted(set(members) - nodes.keys())}")
        if fact_year <= cutoff and source_year <= cutoff and set(members) <= visible.keys():
            record = {key: deepcopy(edge[key]) for key in
                      ("id", "relation_type", "members", "year", "provenance")}
            if "attributes" in edge:
                record["attributes"] = deepcopy(edge["attributes"])
            accepted.append(record)
    return {"cutoff": cutoff, "nodes": list(visible.values()), "hyperedges": accepted,
            "policy": POLICY, "limitations": list(LIMITATIONS)}


def delta(previous, current):
    """Return new nodes/edges and changed node metadata for an append-only view.

    Existing edge records are immutable: changing one under its old ID requires
    an explicit versioning policy outside this contract. Node records can gain
    provenance as newly visible articles arrive, and are returned in full.
    """
    if current["cutoff"] < previous["cutoff"]:
        raise ValueError("Temporal cutoff cannot move backwards")
    if current.get("policy") != previous.get("policy"):
        raise ValueError("Temporal policy cannot change within one sequence")
    old_nodes = _index(previous["nodes"], "node")
    new_nodes = _index(current["nodes"], "node")
    old_edges = _index(previous["hyperedges"], "hyperedge")
    new_edges = _index(current["hyperedges"], "hyperedge")
    if old_nodes.keys() - new_nodes.keys():
        raise ValueError("Append-only update removed nodes")
    if old_edges.keys() - new_edges.keys():
        raise ValueError("Append-only update removed hyperedges")
    for eid, edge in old_edges.items():
        if edge != new_edges[eid]:
            raise ValueError(f"Existing hyperedge payload changed: {eid}")
    return {
        "nodes_added": [deepcopy(new_nodes[nid]) for nid in sorted(new_nodes.keys() - old_nodes.keys())],
        "nodes_changed": [deepcopy(new_nodes[nid]) for nid in sorted(old_nodes) if new_nodes[nid] != old_nodes[nid]],
        "hyperedges_added": [deepcopy(new_edges[eid]) for eid in sorted(new_edges.keys() - old_edges.keys())],
    }
