"""Measured diagnostics, independent-family controls and honest routing costs.

None is used for an undefined measurement. Neither passing structural checks nor
extracting member names establishes semantic quality or scientific faithfulness.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score


def _distribution(values):
    values = list(values)
    x = np.asarray(values, dtype=float)
    return {"count": len(values), "values": values,
            "min": float(x.min()) if len(x) else None,
            "max": float(x.max()) if len(x) else None,
            "mean": float(x.mean()) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None}


def _maps(state):
    return [{nid: state.parents[gid] for nid, gid in state.assignment.items()}, dict(state.assignment)]


def _members(mapping):
    groups = defaultdict(list)
    for nid, gid in sorted(mapping.items()):
        groups[gid].append(nid)
    return dict(sorted(groups.items()))


def _normalized_entropy(counts):
    total = sum(counts)
    if total <= 1:
        return 0.
    return -sum(c / total * math.log(c / total) for c in counts if c) / math.log(total)


def _source_purity(members, by_id):
    # A node with two sources contributes half a vote to each. It is not counted
    # twice in the denominator, nor arbitrarily assigned to its first source.
    sources = Counter()
    known = 0
    for nid in members:
        declared = set(by_id[nid].get("provenance", {}).get("articles", []))
        if declared:
            known += 1
            for source in declared:
                sources[source] += 1 / len(declared)
    return (max(sources.values()) / known if known else None), len(sources), known


def measure(state, independent_vectors=None):
    """Snapshot T1/T3/T4 diagnostics and limited T5 extractive-label checks.

    Optional independent vectors are never silently taken from state.vectors.
    The caller must establish the provenance of the independent representation.
    """
    view = state.view
    by_id = {n["id"]: n for n in view["nodes"]}
    edges = view["hyperedges"]
    incident = {n for edge in edges for n in edge["members"]}
    reports = []
    for level, mapping in enumerate(_maps(state)):
        groups = _members(mapping)
        child_counts = Counter(state.parents.values()) if level == 0 else Counter(mapping.values())
        rows, purity_sum, purity_denominator = [], 0., 0
        for gid, members in groups.items():
            purity, source_count, known = _source_purity(members, by_id)
            if purity is not None:
                purity_sum += purity * known
                purity_denominator += known
            rows.append({"id": gid, "size": len(members), "source_count": source_count,
                         "source_purity": purity, "child_count": child_counts[gid]})
        entropies = []
        internal = 0
        for edge in edges:
            # Count all native endpoints. No pairwise graph is constructed.
            counts = Counter(mapping[n] for n in edge["members"])
            entropies.append(_normalized_entropy(counts.values()))
            internal += len(counts) == 1
        reports.append({"level": level, "groups": len(groups), "group_diagnostics": rows,
                        "group_sizes": _distribution([len(ns) for ns in groups.values()]),
                        "child_counts": _distribution([child_counts[gid] for gid in groups]),
                        "largest_group_fraction": max(map(len, groups.values())) / len(by_id) if by_id else None,
                        "mean_native_edge_entropy": float(np.mean(entropies)) if entropies else None,
                        "internal_edge_fraction": internal / len(edges) if edges else None,
                        "cross_source_groups": sum(r["source_count"] > 1 for r in rows),
                        "source_purity_node_weighted": purity_sum / purity_denominator if purity_denominator else None,
                        "source_purity_interpretation": "source-capture diagnostic, not a quality score"})
    hierarchy, _ = state.export()
    required_groups = [g for level in hierarchy["levels"][:2] for g in level["groups"]]
    denominator = sum(len(g["members"]) for g in required_groups)
    covered = sum(len(set(g.get("evidence", [])) & set(g["members"])) for g in required_groups)
    generic = {"topic", "group", "cluster", "miscellaneous", "other", "unknown"}
    vague = sum(len(g.get("label", "").split()) < 2 or g.get("label", "").strip().casefold() in generic
                for g in required_groups)
    label_report = {"method": "extractive_representative_names",
                    "required_semantic_groups": len(required_groups),
                    "missing_label_or_gloss": sum(not g.get("label", "").strip() or not g.get("gloss", "").strip()
                                                 for g in required_groups),
                    "vagueness_proxy_flagged_groups": vague,
                    "vagueness_interpretation": "short/generic-name heuristic only; no semantic adequacy conclusion",
                    "extractive_evidence_coverage": covered / denominator if denominator else None,
                    "semantic_faithfulness": {"status": "not_evaluated",
                                             "reason": "Extractive member names do not establish coverage or adequacy of a scientific topic description."}}
    return {"snapshot": {"cutoff": view["cutoff"], "nodes": len(by_id), "hyperedges": len(edges),
                         "node_types": dict(sorted(Counter(n["type"] for n in by_id.values()).items())),
                         "relation_types": dict(sorted(Counter(e["relation_type"] for e in edges).items())),
                         "arity": _distribution([len(e["members"]) for e in edges]),
                         "isolated_nodes": len(by_id.keys() - incident),
                         "node_observation_years": dict(sorted(Counter(str(n.get("observed_at", "unknown")) for n in by_id.values()).items())),
                         "edge_fact_years": dict(sorted(Counter(str(e.get("year", "unknown")) for e in edges).items())),
                         "edge_source_years": dict(sorted(Counter(str(e.get("provenance", {}).get("article_year", "unknown")) for e in edges).items())),
                         "time_policy": view.get("policy"), "limitations": deepcopy(view.get("limitations", []))},
            "levels": reports, "growth_and_cost_trace": deepcopy(state.trace), "labels": label_report,
            "coherence": compare_coherence(state, independent_vectors) if independent_vectors is not None else
                         {"status": "not_evaluated", "reason": "Independent representations were not supplied."}}


def compare_temporal(old, new):
    """ARI on shared old nodes, alongside movement of persistent group IDs.

    ARI is invariant to relabeling and therefore cannot on its own establish ID
    stability. Consecutive years are one dependent history, not independent units.
    """
    before, after = set(old.assignment), set(new.assignment)
    shared = sorted(before & after)
    rows = []
    for level, (a, b) in enumerate(zip(_maps(old), _maps(new))):
        moved = sum(a[n] != b[n] for n in shared)
        rows.append({"level": level,
                     "ari_shared_old_nodes": float(adjusted_rand_score([a[n] for n in shared], [b[n] for n in shared])) if len(shared) >= 2 else None,
                     "persistent_id_movement": moved,
                     "persistent_id_movement_fraction": moved / len(shared) if shared else None,
                     "new_group_ids": sorted(set(b.values()) - set(a.values())),
                     "retired_group_ids": sorted(set(a.values()) - set(b.values())),
                     "unchanged_old_membership": moved == 0 if shared else None})
    return {"from_cutoff": old.view["cutoff"], "to_cutoff": new.view["cutoff"],
            "shared_old_nodes": len(shared), "new_nodes": len(after - before), "removed_nodes": len(before - after),
            "levels": rows, "uncertainty": {"status": "not_estimated",
                "reason": "Shared nodes and adjacent transitions are dependent; no independent-node bootstrap confidence interval is claimed."},
            "interpretation": "High stability must be considered together with new-topic response and update quality."}


def _unit_vectors(vectors, rows):
    x = vectors.tocsr().astype(float) if sparse.issparse(vectors) else np.asarray(vectors, dtype=float)
    if x.ndim != 2 or x.shape[0] != rows or not x.shape[1]:
        raise ValueError("Vector rows/dimensions do not match node or query IDs")
    values = x.data if sparse.issparse(x) else x
    if not np.isfinite(values).all():
        raise ValueError("Vectors contain non-finite values")
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel()) if sparse.issparse(x) else np.linalg.norm(x, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Cosine is undefined for zero vectors")
    return x.multiply((1 / norms)[:, None]).tocsr() if sparse.issparse(x) else x / norms[:, None]


def _pair_coherence(vectors, labels):
    numerator, pairs = 0., 0
    for group in sorted(set(labels)):
        positions = np.flatnonzero(labels == group)
        count = len(positions)
        if count < 2:
            continue
        summed = np.asarray(vectors[positions].sum(axis=0)).ravel()
        numerator += (float(summed @ summed) - count) / 2
        pairs += count * (count - 1) // 2
    return (float(np.clip(numerator / pairs, -1., 1.)) if pairs else None), pairs


def compare_coherence(state, independent_vectors, seed=0, null_samples=20, family=None):
    """Pair-weighted cosine vs size/type-preserving random partitions.

    A central interval of the random-null values is a reference distribution,
    not an effect confidence interval. The Monte Carlo upper-tail p-value uses
    (1 + exceedances)/(1 + samples), and no multiplicity correction is implied.
    """
    if isinstance(null_samples, bool) or not isinstance(null_samples, int) or null_samples < 1:
        raise ValueError("null_samples must be a positive integer")
    n = len(state.view["nodes"])
    if not sparse.issparse(independent_vectors):
        supplied = np.asarray(independent_vectors)
        if supplied.shape == state.vectors.shape and np.array_equal(supplied, state.vectors):
            raise ValueError("Independent evaluation cannot reuse construction vectors")
    x = _unit_vectors(independent_vectors, n)
    types = np.array([node["type"] for node in state.view["nodes"]])
    ids = [node["id"] for node in state.view["nodes"]]
    rng = np.random.default_rng(seed)
    levels = []
    for level, mapping in enumerate(_maps(state)):
        labels = np.array([mapping[n] for n in ids])
        actual, pairs = _pair_coherence(x, labels)
        null = []
        for _ in range(null_samples):
            randomized = labels.copy()
            for kind in sorted(set(types)):
                positions = np.flatnonzero(types == kind)
                randomized[positions] = rng.permutation(labels[positions])
            value, _ = _pair_coherence(x, randomized)
            null.append(value)
        finite = [v for v in null if v is not None]
        mean = float(np.mean(finite)) if finite else None
        levels.append({"level": level, "weighted_pairwise_cosine": actual,
                       "within_group_pairs": pairs, "null_values": null, "null_mean": mean,
                       "difference_from_null_mean": actual - mean if actual is not None and mean is not None else None,
                       "null_central_95_interval": [float(v) for v in np.quantile(finite, [.025, .975])] if finite else None,
                       "monte_carlo_p_greater": (1 + sum(v >= actual for v in finite)) / (1 + len(finite)) if actual is not None and finite else None})
    return {"status": "measured", "representation_family": family,
            "independence": "caller_must_verify_representation_provenance", "seed": seed,
            "null_samples": null_samples, "null_policy": "preserve_group_sizes_and_type_composition",
            "interval_interpretation": "Central 95% range of the random-partition null; not a confidence interval for a population effect.",
            "levels": levels}


def routing(hierarchy, node_ids, passage_vectors, query_vectors, beams=(2, 8), top_k=100, query_ids=None):
    """Coarse-to-fine routing and exact flat cosine with identical final scorer.

    Centroids are normalized means of normalized passage vectors, prepared once
    offline. Work counts actual query-time vector dot products, including routing
    overhead. Both tracks rank the same source nodes; no reranker is introduced.
    """
    ids = list(node_ids)
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Node IDs must be unique and nonempty")
    if len(beams) != 2 or any(type(b) is not int or b < 1 for b in beams) or type(top_k) is not int or top_k < 1:
        raise ValueError("Positive two-level beams and top_k are required")
    x = _unit_vectors(passage_vectors, len(ids))
    if sparse.issparse(x):
        x = x.toarray()
    raw_queries = np.asarray(query_vectors, dtype=float)
    if raw_queries.ndim == 1:
        raw_queries = raw_queries[None, :]
    q = _unit_vectors(raw_queries, len(raw_queries))
    if q.shape[1] != x.shape[1]:
        raise ValueError("Passage and query vector dimensions differ")
    qids = list(query_ids) if query_ids is not None else [str(i) for i in range(len(q))]
    if len(qids) != len(q) or len(set(qids)) != len(qids):
        raise ValueError("Query IDs must be unique and align with query vectors")
    levels = hierarchy["levels"]
    if len(levels) < 2:
        raise ValueError("Routing requires two semantic levels")
    index = {nid: i for i, nid in enumerate(ids)}
    group_rows = []
    for level in levels[:2]:
        groups = sorted(level["groups"], key=lambda g: g["id"])
        seen = []
        for group in groups:
            members = group["members"]
            if not members or set(members) - index.keys():
                raise ValueError("Hierarchy member IDs do not align with vector IDs")
            seen.extend(members)
        if len(seen) != len(ids) or set(seen) != set(ids):
            raise ValueError("Hierarchy level is not a partition of vector IDs")
        group_rows.append(groups)
    tops, fines = group_rows
    top_by_id = {g["id"]: g for g in tops}
    if len(top_by_id) != len(tops) or len({g["id"] for g in fines}) != len(fines):
        raise ValueError("Duplicate hierarchy group IDs")
    for group in fines:
        if group["parent_id"] not in top_by_id or not set(group["members"]) <= set(top_by_id[group["parent_id"]]["members"]):
            raise ValueError("Hierarchy parent does not contain child")
    def centroid(group):
        vector = x[[index[n] for n in group["members"]]].mean(axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector
    top_vectors = np.array([centroid(g) for g in tops])
    fine_vectors = np.array([centroid(g) for g in fines])
    def ranked(positions, scores):
        order = sorted(range(len(positions)), key=lambda j: (-float(scores[j]), ids[positions[j]]))[:top_k]
        return [{"id": ids[positions[j]], "score": float(scores[j])} for j in order]
    results = []
    for qid, query in zip(qids, q):
        flat_scores = x @ query
        flat = ranked(list(range(len(ids))), flat_scores)
        top_scores = top_vectors @ query
        top_order = sorted(range(len(tops)), key=lambda j: (-float(top_scores[j]), tops[j]["id"]))[:beams[0]]
        selected_top = {tops[j]["id"] for j in top_order}
        fine_candidates = [j for j, g in enumerate(fines) if g["parent_id"] in selected_top]
        fine_scores = fine_vectors[fine_candidates] @ query
        selected_fine = sorted(range(len(fine_candidates)), key=lambda j: (-float(fine_scores[j]), fines[fine_candidates[j]]["id"]))[:beams[1]]
        candidates = sorted({index[n] for j in selected_fine for n in fines[fine_candidates[j]]["members"]})
        scores = x[candidates] @ query
        routed = ranked(candidates, scores)
        centroid_work = len(tops) + len(fine_candidates)
        flat_work = {"node_dot_products": len(ids), "centroid_dot_products": 0, "total_dot_products": len(ids)}
        work = {"node_dot_products": len(candidates), "centroid_dot_products": centroid_work,
                "total_dot_products": len(candidates) + centroid_work}
        overlap = len({n["id"] for n in flat} & {n["id"] for n in routed})
        results.append({"query_id": qid, "flat": {"ranked": flat, "work": flat_work},
                        "hierarchy": {"ranked": routed, "work": work, "candidate_count": len(candidates),
                                      "selected_top_groups": sorted(selected_top),
                                      "selected_fine_groups": [fines[fine_candidates[j]]["id"] for j in selected_fine]},
                        "flat_topk_recall": overlap / len(flat) if flat else None,
                        "preparation": "Group centroid construction is offline; query encoding cost is shared and excluded from both tracks."})
    return results
