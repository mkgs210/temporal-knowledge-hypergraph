"""Dependence-aware references for *fixed* chronological partitions.

The finite exported graph has an exact shared-old ARI, not a sampling error.
An optional pairs-cluster bootstrap asks about source-component composition,
conditional on both fitted hierarchies. It does not refit either hierarchy.
Overlapping papers are merged transitively, never treated as independent copies
of the same node. On this dataset that deliberately makes most CIs unavailable.

References (methodological scope, not a coverage theorem for this implementation):
Cameron & Miller, A Practitioner's Guide to Cluster-Robust Inference (2015),
https://faculty.econ.ucdavis.edu/faculty/cameron/research/Cameron_Miller_JHR_2014_July_09.pdf
Phipson & Smyth, Permutation P-values Should Never Be Zero (2010),
https://doi.org/10.2202/1544-6115.1585
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score


def _aligned(previous_labels, current_labels):
    if not isinstance(previous_labels, Mapping) or not isinstance(current_labels, Mapping):
        raise ValueError("Labels must be node-ID mappings")
    if any(not isinstance(n, str) or not n for n in (*previous_labels, *current_labels)):
        raise ValueError("Node IDs must be nonempty strings")
    ids = sorted(previous_labels.keys() & current_labels.keys())
    for labels in (previous_labels, current_labels):
        if any(type(labels[n]) not in (str, int) for n in ids):
            raise ValueError("Group labels must be strings or integers")
    # Encode independently; ARI is invariant to cluster-name permutations.
    def encode(labels):
        unique = {}
        return np.array([unique.setdefault(labels[n], len(unique)) for n in ids], dtype=np.int64)
    return ids, encode(previous_labels), encode(current_labels)


def _ari(a, b):
    return None if len(a) < 2 else float(adjusted_rand_score(a, b))


def _ari_counts(counts):
    """Ordinary integer contingency ARI; multiplicities are whole sampled blocks."""
    total = float(counts.sum())
    if total < 2:
        return None
    choose = lambda x: float(np.sum(x * (x - 1) / 2))
    cells = choose(counts)
    row = choose(counts.sum(axis=1))
    col = choose(counts.sum(axis=0))
    expected = row * col / (total * (total - 1) / 2)
    denominator = (row + col) / 2 - expected
    return 1. if denominator == 0 else (cells - expected) / denominator


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _source_key(source):
    return (0, source) if type(source) is int else (1, source)


def _blocks(ids, node_sources):
    parents = {}
    normalized = {}
    for nid in ids:
        sources = node_sources.get(nid)
        if (not isinstance(sources, Sequence) or isinstance(sources, (str, bytes))
                or not sources or any(type(s) not in (str, int) or s == "" for s in sources)):
            raise ValueError(f"Missing or invalid source-paper list for {nid}")
        normalized[nid] = sorted(set(sources), key=_source_key)
        for source in normalized[nid]:
            parents.setdefault(source, source)

    def find(source):
        while source != parents[source]:
            parents[source] = parents[parents[source]]
            source = parents[source]
        return source

    for sources in normalized.values():
        for other in sources[1:]:
            parents[find(other)] = find(sources[0])
    groups = defaultdict(list)
    papers = defaultdict(list)
    for source in sorted(parents, key=_source_key):
        papers[find(source)].append(source)
    for i, nid in enumerate(ids):
        groups[find(normalized[nid][0])].append(i)
    roots = sorted(groups, key=lambda r: tuple(_source_key(s) for s in papers[r]))
    blocks = [np.array(groups[r], dtype=int) for r in roots]
    size = np.array([len(b) for b in blocks], dtype=float)
    summary = {
        "count": len(blocks), "paper_count": len(parents),
        "overlapping_nodes": sum(len(s) > 1 for s in normalized.values()),
        "sources": [papers[r] for r in roots],
        "node_counts": size.astype(int).tolist(),
        "largest_node_fraction": float(size.max() / size.sum()) if len(size) else None,
        "effective_count_by_node_mass": float(size.sum() ** 2 / np.sum(size ** 2)) if len(size) else 0.,
        "policy": "transitive components of overlapping source-paper membership",
    }
    return blocks, normalized, summary


def source_block_ari(previous_labels, current_labels, node_sources, *,
                     n_resamples=2000, confidence=.95, seed=0, min_blocks=20):
    """Exact shared-ID ARI, conditional source-component bootstrap and diagnostics.

    ``min_blocks`` is an explicit conservative diagnostic guard, not a theorem
    establishing coverage. Both component count and effective count by node mass
    must reach it. No statistical interval is emitted otherwise. Leave-one-source
    deletion always remains a fixed-partition sensitivity analysis, not a CI.
    All provenance lists must come from the *previous* cutoff.
    """
    _positive_integer(n_resamples, "n_resamples")
    _positive_integer(min_blocks, "min_blocks", 2)
    if not isinstance(confidence, (float, int)) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if not isinstance(node_sources, Mapping):
        raise ValueError("node_sources must be a mapping")
    ids, a, b = _aligned(previous_labels, current_labels)
    blocks, sources, summary = _blocks(ids, node_sources)
    point = _ari(a, b)
    moved = sum(previous_labels[n] != current_labels[n] for n in ids)
    rows = []
    for source in sorted({s for ss in sources.values() for s in ss}, key=_source_key):
        keep = np.array([source not in sources[n] for n in ids], dtype=bool)
        rows.append({"source": source, "removed_nodes": int((~keep).sum()),
                     "retained_nodes": int(keep.sum()), "ari": _ari(a[keep], b[keep])})
    valid = [r["ari"] for r in rows if r["ari"] is not None]
    result = {
        "point_ari": point, "shared_old_nodes": len(ids),
        "new_nodes": len(current_labels.keys() - previous_labels.keys()),
        "removed_nodes": len(previous_labels.keys() - current_labels.keys()),
        "persistent_id_movement": moved,
        "movement_fraction": moved / len(ids) if ids else None,
        "blocks": summary, "seed": seed, "requested_resamples": n_resamples,
        "minimum_component_guard": min_blocks, "interval": None, "bootstrap_ari": [],
        "estimand": "shared-old node-weighted ARI of two fixed fitted partitions",
        "interval_interpretation": "conditional source-component composition, not uncertainty of the evolutionary fitting procedure",
        "limitations": [
            "Finite-graph ARI is exact; a confidence interval needs an additional sampling model.",
            "Source components are assumed exchangeable and independent for the conditional bootstrap; semantic and algorithmic dependencies may violate this.",
            "Neither hierarchy is refitted; optimization, new-paper arrivals and graph-extraction uncertainty are excluded.",
            "The percentile interval has no demonstrated nominal coverage on this corpus; the component guard does not prove coverage.",
            "Dependent overlapping papers are merged; replacing them by independently sampled nodes would invent information.",
        ],
        "source_delete_one": {
            "interpretation": "fixed-partition sensitivity, not a confidence interval",
            "rows": rows, "range": [min(valid), max(valid)] if valid else None,
            "maximum_absolute_change": max(abs(v - point) for v in valid) if valid and point is not None else None,
            "unavailable_deletions": len(rows) - len(valid),
        },
    }
    if point is None:
        result["uncertainty_status"] = "too_few_shared_nodes"
        return result
    if len(blocks) < min_blocks or summary["effective_count_by_node_mass"] < min_blocks:
        result["uncertainty_status"] = "insufficient_source_components"
        return result
    # Integer block contingencies avoid repeatedly copying all node labels.
    n_a, n_b = int(a.max()) + 1, int(b.max()) + 1
    tables = np.array([np.bincount(a[idx] * n_b + b[idx], minlength=n_a * n_b)
                       for idx in blocks], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        counts = tables[chosen].sum(axis=0).reshape(n_a, n_b)
        samples.append(float(_ari_counts(counts)))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    result.update(uncertainty_status="conditional_component_bootstrap",
                  bootstrap_ari=samples,
                  interval={"low": float(low), "high": float(high), "confidence": confidence,
                            "method": "pairs source-component percentile bootstrap",
                            "coverage_verified": False})
    return result


def alignment_significance(previous_labels, current_labels, node_types=None, *,
                           n_permutations=1999, seed=0):
    """Chance-alignment reference with exact fixed-partition randomization null.

    Uniformly permute current membership labels among shared IDs within each
    node type. This preserves current group sizes and type-by-group counts.
    The null is random ID correspondence, not iid observed graph nodes. It tests
    neither locality of changes nor semantic quality nor prospective stability.
    """
    _positive_integer(n_permutations, "n_permutations")
    ids, a, b = _aligned(previous_labels, current_labels)
    if node_types is None:
        node_types = dict.fromkeys(ids, "all")
    if any(n not in node_types or not isinstance(node_types[n], str) for n in ids):
        raise ValueError("Every shared node needs a string node type")
    point = _ari(a, b)
    result = {
        "point_ari": point, "shared_old_nodes": len(ids), "seed": seed,
        "n_permutations": n_permutations, "null_ari": [], "p_greater_equal": None,
        "null": "uniform current-label permutation within shared-old node types, conditional on both fitted partitions",
        "interpretation": "fixed-partition random-correspondence test, not temporal-process uncertainty",
        "limitations": ["Rejecting random correspondence does not establish useful adaptation or semantic quality.",
                        "Frozen partitions will also strongly reject this null.",
                        "This is a one-sided unadjusted per-level/per-transition reference; multiplicity must be reported when testing a family."],
    }
    if point is None:
        return result
    strata = [np.array([i for i, n in enumerate(ids) if node_types[n] == kind], dtype=int)
              for kind in sorted({node_types[n] for n in ids})]
    n_a, n_b = int(a.max()) + 1, int(b.max()) + 1
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        permuted = b.copy()
        for idx in strata:
            permuted[idx] = rng.permutation(b[idx])
        counts = np.bincount(a * n_b + permuted, minlength=n_a * n_b).reshape(n_a, n_b)
        null.append(float(_ari_counts(counts)))
    # A tiny tolerance only handles the equivalent floating implementations of ARI.
    exceed = sum(v >= point - 1e-12 for v in null)
    result.update(null_ari=null, exceedances=exceed,
                  p_greater_equal=(1 + exceed) / (1 + n_permutations),
                  attainable_p_resolution=1 / (1 + n_permutations),
                  null_mean=float(np.mean(null)),
                  null_range=[min(null), max(null)])
    return result


def _snapshot_labels(view, hierarchy, levels):
    if hierarchy.get("cutoff") != view.get("cutoff"):
        raise ValueError("Hierarchy and view cutoff differ")
    nodes = {n["id"]: n for n in view["nodes"]}
    if len(nodes) != len(view["nodes"]):
        raise ValueError("Duplicate node ID")
    result = {}
    level_rows = {level["index"]: level for level in hierarchy["levels"]}
    if len(level_rows) != len(hierarchy["levels"]):
        raise ValueError("Duplicate hierarchy level")
    for index in levels:
        if index not in level_rows:
            raise ValueError(f"Missing hierarchy level {index}")
        labels = {}
        group_ids = set()
        for group in level_rows[index]["groups"]:
            if group["id"] in group_ids:
                raise ValueError("Duplicate group ID")
            group_ids.add(group["id"])
            for member in group["members"]:
                if member in labels or member not in nodes:
                    raise ValueError("Overlapping or unknown hierarchy member")
                labels[member] = group["id"]
        if labels.keys() != nodes.keys():
            raise ValueError("Partition does not cover snapshot nodes")
        result[index] = labels
    return nodes, result


def snapshot_temporal_uncertainty(previous_view, previous_hierarchy, current_view,
                                  current_hierarchy, *, levels=(0, 1), n_resamples=2000,
                                  n_permutations=1999, confidence=.95, seed=0, min_blocks=20):
    """Validate/aligned snapshots, using previous-cutoff provenance for blocking."""
    if previous_view["cutoff"] >= current_view["cutoff"]:
        raise ValueError("Expected strictly increasing snapshot cutoffs")
    old_nodes, old = _snapshot_labels(previous_view, previous_hierarchy, levels)
    new_nodes, new = _snapshot_labels(current_view, current_hierarchy, levels)
    shared = sorted(old_nodes.keys() & new_nodes.keys())
    if any(old_nodes[n]["type"] != new_nodes[n]["type"] for n in shared):
        raise ValueError("A shared node changed type")
    sources = {n: old_nodes[n].get("provenance", {}).get("articles") for n in shared}
    blocks, _, _ = _blocks(shared, sources)
    component = {shared[i]: k for k, block in enumerate(blocks) for i in block}
    def crossing(view):
        return sum(len({component[n] for n in edge["members"] if n in component}) > 1
                   for edge in view.get("hyperedges", []))
    dependence = {
        "previous_shared_old_hyperedges_crossing_source_components": crossing(previous_view),
        "current_shared_old_hyperedges_crossing_source_components": crossing(current_view),
        "remaining_dependence": "Even without a crossing native edge, semantic similarity and shared fitted objectives can couple source components.",
    }
    reports = []
    for index in levels:
        report = source_block_ari(old[index], new[index], sources, n_resamples=n_resamples,
                                  confidence=confidence, seed=seed + index, min_blocks=min_blocks)
        report["level"] = index
        if report["interval"] is not None and (crossing(previous_view) or crossing(current_view)):
            report["interval"] = None
            report["uncertainty_status"] = "cross_component_hyperedge_dependence"
        report["alignment_significance"] = alignment_significance(
            old[index], new[index], {n: old_nodes[n]["type"] for n in shared},
            n_permutations=n_permutations, seed=seed + index)
        reports.append(report)
    return {
        "previous_cutoff": previous_view["cutoff"], "current_cutoff": current_view["cutoff"],
        "source_policy": "previous-cutoff provenance only", "levels": reports,
        "dependence": dependence, "satisfies_unconditional_ci_requirement": False,
        "significance_scope": "T6 significance reference supplied for a fixed-partition random-correspondence null; genuine process-level confidence intervals remain unavailable.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    snapshots = sorted(p for p in args.snapshots.iterdir() if (p / "hierarchy.json").is_file())
    records, hashes = [], {}
    for directory in snapshots:
        for name in ("input.json", "hierarchy.json"):
            path = directory / name
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append((json.loads((directory / "input.json").read_text()),
                        json.loads((directory / "hierarchy.json").read_text())))
    transitions = [snapshot_temporal_uncertainty(*previous, *current,
                   n_resamples=args.resamples, n_permutations=args.permutations,
                   seed=args.seed + i * 100) for i, (previous, current)
                   in enumerate(zip(records, records[1:]))]
    # One family consists of all requested level/transition alignment tests.
    family_size = sum(r["alignment_significance"]["p_greater_equal"] is not None
                      for transition in transitions for r in transition["levels"])
    for transition in transitions:
        for report in transition["levels"]:
            reference = report["alignment_significance"]
            p = reference["p_greater_equal"]
            reference["bonferroni_family_size"] = family_size
            reference["p_bonferroni"] = None if p is None else min(1., p * family_size)
    output = {"schema_version": "temporal-uncertainty-v1", "inputs_sha256": hashes,
              "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "transitions": transitions,
              "methodological_references": [
                  "https://faculty.econ.ucdavis.edu/faculty/cameron/research/Cameron_Miller_JHR_2014_July_09.pdf",
                  "https://doi.org/10.2202/1544-6115.1585"],
              "scientific_conclusion": "Finite-graph ARIs and conditional random-correspondence significance are measured. Source-overlap components are too few for a credible component bootstrap CI. Source-deletion sensitivity is not a CI and neither reference estimates future-evolution uncertainty."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
