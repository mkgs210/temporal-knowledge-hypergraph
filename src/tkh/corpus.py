"""Gold-free graph documents and question loading."""
import csv
import json
from pathlib import Path


def load_graph(path):
    graph = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_graph(graph)
    return graph


def validate_graph(graph):
    ids = set()
    for node in graph["nodes"]:
        if not isinstance(node.get("id"), str) or not node["id"] or node["id"] in ids:
            raise ValueError("Node IDs must be nonempty and unique")
        ids.add(node["id"])
        if not all(isinstance(node.get(key), str) and node[key].strip() for key in ("type", "surface_form")):
            raise ValueError(f"Missing type or text: {node['id']}")
        aliases = node.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError(f"Invalid aliases: {node['id']}")
    if not ids:
        raise ValueError("Graph has no nodes")
    edge_ids = set()
    for edge in graph["hyperedges"]:
        if not isinstance(edge.get("id"), str) or not edge["id"] or edge["id"] in edge_ids:
            raise ValueError("Hyperedge IDs must be nonempty and unique")
        edge_ids.add(edge["id"])
        if not isinstance(edge.get("members"), list) or not edge["members"]:
            raise ValueError(f"Hyperedge has no member list: {edge['id']}")
        if set(edge["members"]) - ids:
            raise ValueError(f"Dangling members: {edge['id']}")
        if not isinstance(edge.get("relation_type"), str):
            raise ValueError(f"Missing relation type: {edge['id']}")


def load_questions(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    ids = set()
    for row in rows:
        if not row.get("question_id") or row["question_id"] in ids:
            raise ValueError("Question IDs must be nonempty and unique")
        ids.add(row["question_id"])
        if not row.get("question", "").strip() or row.get("type") not in ("A", "B"):
            raise ValueError(f"Invalid question: {row['question_id']}")
    if not rows:
        raise ValueError("No questions")
    return rows


def documents(graph, profile="node"):
    validate_graph(graph)
    if profile not in ("node", "context"):
        raise ValueError("Unknown document profile")
    nodes = {node["id"]: node for node in graph["nodes"]}
    neighbors = {nid: set() for nid in nodes}
    relations = {nid: set() for nid in nodes}
    edge_ids = {nid: [] for nid in nodes}
    if profile == "context":
        for edge in graph["hyperedges"]:
            members = set(edge["members"])
            for nid in members:
                neighbors[nid].update(members - {nid})
                relations[nid].add(edge["relation_type"].replace("_", " "))
                edge_ids[nid].append(edge["id"])

    def name(node):
        return "; ".join([node["type"] + ": " + node["surface_form"], *node.get("aliases", [])])

    result = []
    for nid, node in sorted(nodes.items()):
        texts = [name(node)]
        if profile == "context":
            texts += sorted(relations[nid])
            texts += [name(nodes[other]) for other in sorted(neighbors[nid])]
        result.append({"id": nid, "type": node["type"], "surface_form": node["surface_form"],
                       "text": "\n".join(texts), "neighbor_ids": sorted(neighbors[nid]),
                       "edge_ids": sorted(edge_ids[nid])})
    return result
