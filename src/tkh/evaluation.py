"""Reference matching and retrieval metrics; never imported by the predictor."""
from collections import defaultdict
from functools import lru_cache
import re
import statistics
import unicodedata


@lru_cache(maxsize=32768)
def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def entity_targets(names, graph):
    index = defaultdict(set)
    for node in graph["nodes"]:
        for text in [node["surface_form"], *node.get("aliases", [])]:
            index[normalize(text)].add(node["id"])
    return [{"label": name, "node_ids": sorted(index[normalize(name)]),
             "match_rule": "exact_surface_or_explicit_graph_alias"} for name in names]


def _doi(text):
    return normalize(text or "").removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:").strip()


def _arxiv(text):
    return set(re.findall(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", text or ""))


def source_targets(sources, graph):
    articles = [node for node in graph["nodes"] if node["type"] == "article"]
    result = []
    for label, source in sources.items():
        wanted_doi = _doi(source.get("doi"))
        wanted_arxiv = _arxiv(" ".join(str(source.get(key) or "") for key in ("arxiv_id", "arxiv", "venue", "url")))
        title = normalize(source.get("title", ""))
        matches, warnings, rules = [], set(), set()
        for node in articles:
            provenance = node.get("provenance", {})
            same_doi = bool(wanted_doi and wanted_doi == _doi(provenance.get("doi")))
            same_arxiv = bool(wanted_arxiv & _arxiv(str(provenance.get("arxiv_id") or "")))
            same_title = bool(title and title == normalize(node["surface_form"]))
            if same_doi or same_arxiv or same_title:
                matches.append(node["id"])
                rules.update(rule for rule, matched in (("doi", same_doi), ("arxiv", same_arxiv), ("title", same_title)) if matched)
                if (same_doi or same_arxiv) and title and not same_title:
                    warnings.add("title_disagreement")
                year = node.get("first_seen_year", node.get("year"))
                if year is not None and source.get("date") and int(source["date"][:4]) != int(year):
                    warnings.add("publication_or_version_year_disagreement")
        result.append({"label": label, "title": source.get("title"), "node_ids": sorted(matches),
                       "match_rules": sorted(rules), "warnings": sorted(warnings)})
    return result


def score_targets(ranking, targets, ks):
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate node IDs")
    if not targets:
        raise ValueError("Reference targets must be nonempty")
    positions = {nid: i + 1 for i, nid in enumerate(ranking)}
    ranks = [min((positions[nid] for nid in row["node_ids"] if nid in positions), default=None) for row in targets]
    reachable = sum(bool(row["node_ids"]) for row in targets)
    result = {}
    for k in ks:
        hits = sum(rank is not None and rank <= k for rank in ranks)
        first = min((rank for rank in ranks if rank is not None and rank <= k), default=None)
        result[str(k)] = {"hits": hits, "targets_all": len(targets), "targets_reachable": reachable,
                          "recall_all": hits / len(targets), "recall_reachable": hits / reachable if reachable else None,
                          "any_hit": int(hits > 0), "reciprocal_rank": 1 / first if first else 0.,
                          "first_hit_rank": first, "returned": min(k, len(ranking))}
    return result


def evaluate(graph, questions, predictions, gold, ks):
    qids = {row["question_id"] for row in questions}
    if qids != set(gold) or qids != set(predictions):
        raise ValueError("Questions, predictions and reference answers must have identical IDs")
    valid_ids = {node["id"] for node in graph["nodes"]}
    records = []
    for question in questions:
        qid = question["question_id"]
        answer = gold[qid]
        if answer["type"] != question["type"]:
            raise ValueError(f"Question/reference type mismatch: {qid}")
        ranking = predictions[qid]
        if set(ranking) - valid_ids:
            raise ValueError(f"Unknown predicted node: {qid}")
        if question["type"] == "A":
            targets = entity_targets(answer["expected_methods"], graph)
            track = "A"
        else:
            required = answer.get("required_citations", list(answer["sources"]))
            targets = source_targets({name: answer["sources"][name] for name in required}, graph)
            track = "B_sources"
        positions = {nid: i + 1 for i, nid in enumerate(ranking)}
        for target in targets:
            target["first_rank"] = min((positions[nid] for nid in target["node_ids"] if nid in positions), default=None)
        row = {"question_id": qid, "question": question["question"], "type": question["type"], "track": track,
               "targets": targets, "metrics": score_targets(ranking, targets, ks),
               "temporal_evaluation": "not_evaluated_full_export_has_no_monthly_cutoff"}
        if question["type"] == "B":
            row["claim_evaluation"] = {"status": "not_evaluated", "required_claims": len(answer["required_claims"])}
        records.append(row)
    tracks = {}
    for track in ("A", "B_sources"):
        rows = [row for row in records if row["track"] == track]
        summary = {}
        for k in ks:
            values = [row["metrics"][str(k)] for row in rows]
            if not values:
                continue
            reachable_values = [v["recall_reachable"] for v in values if v["recall_reachable"] is not None]
            total = sum(v["targets_all"] for v in values)
            reachable = sum(v["targets_reachable"] for v in values)
            hits = sum(v["hits"] for v in values)
            summary[str(k)] = {"hits": hits, "targets_all": total, "targets_reachable": reachable,
                               "recall_all_micro": hits / total, "recall_all_macro": statistics.mean(v["recall_all"] for v in values),
                               "recall_reachable_micro": hits / reachable if reachable else None,
                               "recall_reachable_macro": statistics.mean(reachable_values) if reachable_values else None,
                               "reachable_questions": len(reachable_values), "any_hit_questions": sum(v["any_hit"] for v in values),
                               "any_hit_rate": statistics.mean(v["any_hit"] for v in values),
                               "mrr": statistics.mean(v["reciprocal_rank"] for v in values)}
        tracks[track] = {"questions": len(rows), "metrics": summary}
    return {"tracks": tracks, "questions": records,
            "claim_quality_status": "not_evaluated", "precision_status": "not_estimated_incomplete_reference_lists"}
