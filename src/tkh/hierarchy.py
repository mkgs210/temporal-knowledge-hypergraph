"""Native entropy + Ward hierarchy, with a separate saved-state local updater.

No pairwise graph projection: each edge retains endpoint multiplicity throughout
coarsening. See docs/METHOD.md for the objective and its limits.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def label_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def collapse(edges, assignment):
    """Preserve every original edge, including entirely internal edges.

    Counts compose exactly; IDs, relation, provenance and attributes are retained.
    This loses endpoint identities in counts, but the original input is also saved.
    """
    result = []
    for edge in edges:
        counts = Counter()
        for member, count in edge.get("counts", dict.fromkeys(edge["members"], 1)).items():
            counts[assignment[member]] += count
        row = {k: deepcopy(v) for k, v in edge.items() if k not in ("members", "counts")}
        row.update(members=sorted(counts), counts=dict(sorted(counts.items())))
        result.append(row)
    return result


def entropy(counts):
    n = sum(counts)
    if n <= 1:
        return 0.
    return -sum((c / n) * math.log(c / n) for c in counts if c) / math.log(n)


def _entropy_merge(c, d, total):
    if not c or not d or total <= 1:
        return 0.
    return (c * math.log(c) + d * math.log(d) - (c + d) * math.log(c + d)) / (total * math.log(total))


def agglomerate(ids, vectors, edges, budgets, structural_weight=.2):
    """One exact greedy merge history, with all requested laminar cuts.

    A dense Ward matrix and lazy row-minimum heap avoid a Python heap of O(n²)
    entries. Unchanged-pair entropy deltas do not change after an unrelated merge.
    """
    ids = list(ids)
    x = np.asarray(vectors, dtype=np.float64)
    n = len(ids)
    if x.ndim != 2 or len(x) != n or not n or not np.isfinite(x).all():
        raise ValueError("Invalid vectors")
    targets = {min(n, int(b)) for b in budgets}
    if not targets or min(targets) < 1 or structural_weight < 0:
        raise ValueError("Invalid budget/weight")
    index = {nid: i for i, nid in enumerate(ids)}
    if len(index) != n:
        raise ValueError("Duplicate node ID")
    squared = np.einsum("ij,ij->i", x, x)
    ward = np.maximum((squared[:, None] + squared[None, :] - 2 * x @ x.T) / (2 * n), 0.)
    structural = np.zeros((n, n), dtype=float)
    incident = [set() for _ in ids]
    counts = []
    totals = []
    for ei, edge in enumerate(collapse(edges, {nid: nid for nid in ids})):
        counter = {index[k]: v for k, v in edge["counts"].items()}
        counts.append(counter)
        totals.append(sum(counter.values()))
        for i in counter:
            incident[i].add(ei)
        for i, c in counter.items():
            for j, d in counter.items():
                if j > i:
                    delta = structural_weight * _entropy_merge(c, d, totals[-1]) / max(len(edges), 1)
                    structural[i, j] += delta
                    structural[j, i] += delta
    cost = ward + structural
    np.fill_diagonal(cost, np.inf)
    active = np.ones(n, dtype=bool)
    size = np.ones(n, dtype=int)
    groups = [{nid} for nid in ids]
    best = np.full(n, -1, dtype=int)
    version = np.zeros(n, dtype=int)
    dirty = np.zeros(n, dtype=bool)
    heap = []

    def refresh(i):
        j = int(np.argmin(cost[i]))
        best[i] = j
        version[i] += 1
        dirty[i] = False
        heapq.heappush(heap, (float(cost[i, j]), i, j, int(version[i])))

    for i in range(n):
        refresh(i)
    cuts = {}

    def capture(k):
        cuts[k] = sorted((sorted(groups[i]) for i in range(n) if active[i]), key=lambda g: g[0])

    if n in targets:
        capture(n)
    for remaining in range(n, min(targets), -1):
        while True:
            _, a, b, rev = heapq.heappop(heap)
            if not active[a] or version[a] != rev:
                continue
            if dirty[a] or not active[b]:
                refresh(a)
                continue
            break
        # Reuse a's matrix slot; b is permanently retired.
        na, nb = size[a], size[b]
        old = ward[a].copy()
        ward[a] = ((na + size) * old + (nb + size) * ward[b] - size * ward[a, b]) / (na + nb + size)
        ward[:, a] = ward[a]
        size[a] += size[b]
        active[b] = False
        groups[a].update(groups[b])
        groups[b].clear()
        touched = incident[a] | incident[b]
        incident[a] = touched
        incident[b] = set()
        structural[a] = 0.
        for ei in touched:
            c = counts[ei].pop(a, 0) + counts[ei].pop(b, 0)
            counts[ei][a] = c
            for j, d in counts[ei].items():
                if j != a:
                    structural[a, j] += structural_weight * _entropy_merge(c, d, totals[ei]) / max(len(edges), 1)
        structural[:, a] = structural[a]
        cost[a] = ward[a] + structural[a]
        cost[:, a] = cost[a]
        cost[b, :] = cost[:, b] = np.inf
        cost[~active, a] = cost[a, ~active] = np.inf
        cost[a, a] = np.inf
        for i in np.flatnonzero(active):
            if i == a:
                continue
            if best[i] in (a, b):
                dirty[i] = True
            # A newly decreased value must enter the heap immediately; retired
            # minima remain lower bounds until lazily recomputed at the head.
            if best[i] == b or dirty[i]:
                # The stored heap priority may be stale in either direction.
                # Refresh affected minima; all other pairs are unchanged.
                refresh(int(i))
            elif cost[i, a] < cost[i, best[i]]:
                refresh(int(i))
        refresh(a)
        if remaining - 1 in targets:
            capture(remaining - 1)
    return {b: cuts[min(n, b)] for b in budgets}


def _validated(view, vectors):
    ids = [n["id"] for n in view["nodes"]]
    x = np.asarray(vectors, dtype=np.float64)
    if len(set(ids)) != len(ids) or x.ndim != 2 or len(x) != len(ids) or not np.isfinite(x).all():
        raise ValueError("Invalid node/vector input")
    if len({e["id"] for e in view["hyperedges"]}) != len(view["hyperedges"]):
        raise ValueError("Duplicate edge ID")
    if any(not set(e["members"]) <= set(ids) or not e["members"] for e in view["hyperedges"]):
        raise ValueError("Invalid hyperedge members")
    return ids, x


@dataclass
class State:
    view: dict
    vectors: np.ndarray
    assignment: dict
    parents: dict
    config: dict
    events: list
    trace: dict
    next_id: int = 0

    def to_dict(self):
        return deepcopy({"view": self.view, "assignment": self.assignment, "parents": self.parents,
                         "config": self.config, "events": self.events, "trace": self.trace, "next_id": self.next_id,
                         "vectors_sha256": hashlib.sha256(self.vectors.tobytes()).hexdigest()})

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
        try:
            np.savez_compressed(staging / "vectors.npz", vectors=self.vectors)
            (staging / "state.json").write_text(canonical(self.to_dict()) + "\n", encoding="utf-8")
            staging.rename(path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads((path / "state.json").read_text(encoding="utf-8"))
        x = np.load(path / "vectors.npz", allow_pickle=False)["vectors"]
        if hashlib.sha256(x.tobytes()).hexdigest() != data.pop("vectors_sha256"):
            raise ValueError("Corrupt checkpoint vectors")
        state = cls(vectors=x, **data)
        ids, _ = _validated(state.view, x)
        if set(state.assignment) != set(ids):
            raise ValueError("Checkpoint assignment does not cover exactly its nodes")
        if set(state.assignment.values()) != set(state.parents):
            raise ValueError("Checkpoint fine groups/parents disagree")
        if len(state.parents) > state.config["budgets"][1] or len(set(state.parents.values())) > state.config["budgets"][0]:
            raise ValueError("Checkpoint exceeds frozen budgets")
        return state

    def export(self):
        by_id = {n["id"]: n for n in self.view["nodes"]}
        ids = list(by_id)
        index = {nid: i for i, nid in enumerate(ids)}
        assignments = [{n: self.parents[self.assignment[n]] for n in ids}, self.assignment,
                       {n: "node:" + n for n in ids}]
        levels, inputs = [], {}
        for level, assignment in enumerate(assignments):
            members = defaultdict(list)
            for nid, gid in sorted(assignment.items()):
                members[gid].append(nid)
            rows = []
            for gid, ns in sorted(members.items()):
                payload = [{k: by_id[n][k] for k in ("id", "type", "surface_form")} for n in ns]
                inputs[gid] = payload
                xx = self.vectors[[index[n] for n in ns]]
                centrality = xx @ xx.mean(axis=0)
                order = sorted(range(len(ns)), key=lambda i: (-float(centrality[i]), ns[i]))
                # Extractive representatives; disclose limited coverage instead of
                # claiming the examples describe every member of a broad cluster.
                chosen = []
                for i in order:
                    if by_id[ns[i]]["surface_form"] not in [by_id[n]["surface_form"] for n in chosen]:
                        chosen.append(ns[i])
                    if len(chosen) == 3:
                        break
                title = " / ".join(by_id[n]["surface_form"] for n in chosen[:2])
                gloss = "Representative member names: " + "; ".join(by_id[n]["surface_form"] for n in chosen) + "."
                parent = None if level == 0 else (self.parents[gid] if level == 1 else self.assignment[ns[0]])
                rows.append({"id": gid, "parent_id": parent, "members": ns, "label": title,
                             "gloss": gloss, "evidence": chosen, "label_input_hash": label_hash(payload)})
            levels.append({"index": level, "groups": rows})
        return {"cutoff": self.view["cutoff"], "budgets": [*self.config["budgets"], len(ids)], "levels": levels}, inputs


def build(view, vectors, budgets=(12, 48), structural_weight=.2, cuts=None):
    ids, x = _validated(view, vectors)
    if len(budgets) != 2 or not 1 <= budgets[0] <= budgets[1]:
        raise ValueError("Expected two ascending semantic budgets")
    cuts = cuts or agglomerate(ids, x, view["hyperedges"], budgets, structural_weight)
    coarse = {n: f"L0-g{i:06d}" for i, group in enumerate(cuts[budgets[0]]) for n in group}
    assignment = {n: f"L1-g{i:06d}" for i, group in enumerate(cuts[budgets[1]]) for n in group}
    parents = {assignment[n]: coarse[n] for n in ids}
    if any(len({coarse[n] for n in group}) != 1 for group in cuts[budgets[1]]):
        raise ValueError("Cuts are not laminar")
    config = {"budgets": list(budgets), "structural_weight": structural_weight,
              "change_penalty": .02, "candidate_groups": 3, "refine_passes": 2, "novelty_cosine": .25,
              "refinement_scope": "new_edge_endpoints_v2"}
    events = [{"kind": "birth", "level": level, "group_id": gid, "node_ids": sorted(n for n in ids if mapping[n] == gid),
               "previous_group_ids": []} for level, mapping in [(0, coarse), (1, assignment), (2, {n: 'node:' + n for n in ids})]
              for gid in sorted(set(mapping.values()))]
    return State(deepcopy(view), x.copy(), assignment, parents, config, events,
                 {"initial_builder_calls": 1, "new_nodes": ids, "new_edges": [e["id"] for e in view["hyperedges"]]},
                 max(budgets) + 1)


class _LocalPartition:
    """Exact single-unit objective deltas with sufficient group statistics."""
    def __init__(self, ids, x, edges, assignment, weight, old=None, penalty=0., masses=None):
        self.ids, self.x = ids, x
        self.assignment = dict(assignment)
        self.weight, self.penalty = weight, penalty
        self.old = old or {}
        self.mass = np.ones(len(ids)) if masses is None else np.asarray(masses)
        self.total = float(self.mass.sum())
        self.index = {n: i for i, n in enumerate(ids)}
        self.sizes, self.sums = {}, {}
        self.incident = defaultdict(list)
        self.edge_units, self.counts, self.arity = [], [], []
        for ei, edge in enumerate(edges):
            units = edge.get("counts", dict.fromkeys(edge["members"], 1))
            self.edge_units.append(units)
            self.arity.append(sum(units.values()))
            c = Counter()
            for n, count in units.items():
                self.incident[n].append(ei)
                if n in self.assignment:
                    c[self.assignment[n]] += count
            self.counts.append(c)
        for n, group in self.assignment.items():
            i = self.index[n]
            self.sizes[group] = self.sizes.get(group, 0.) + self.mass[i]
            self.sums[group] = self.sums.get(group, np.zeros(x.shape[1])) + self.mass[i] * x[i]

    def delta(self, node, target):
        origin = self.assignment.get(node)
        if origin == target:
            return 0.
        i = self.index[node]
        m, v = self.mass[i], self.mass[i] * self.x[i]
        ds = 0.
        for gid, sign in [(origin, -1), (target, 1)]:
            if gid is None:
                continue
            size = self.sizes.get(gid, 0.)
            total = self.sums.get(gid, np.zeros_like(v))
            before = np.dot(total, total) / size if size else 0.
            after_size, after_sum = size + sign * m, total + sign * v
            after = np.dot(after_sum, after_sum) / after_size if after_size > 1e-9 else 0.
            ds += before - after
        # x² is constant for reassignment, and omitted for insertion comparisons.
        de = 0.
        for ei in self.incident[node]:
            c = self.counts[ei]
            count = self.edge_units[ei][node]
            after = c.copy()
            if origin is not None:
                after[origin] -= count
            after[target] += count
            n = self.arity[ei]
            def term(count):
                return -(count / n) * math.log(count / n) / math.log(n) if count and n > 1 else 0.
            de += sum(term(after[g]) - term(c[g]) for g in {origin, target} if g is not None)
        dt = self.penalty * m / self.total * ((target != self.old[node]) - (origin != self.old[node])) if node in self.old else 0.
        return ds / self.total + self.weight * de / max(len(self.counts), 1) + dt

    def move(self, node, target):
        origin = self.assignment.get(node)
        if origin == target:
            return
        i = self.index[node]
        mass, vector = self.mass[i], self.mass[i] * self.x[i]
        if origin is not None:
            self.sizes[origin] -= mass
            self.sums[origin] -= vector
        self.sizes[target] = self.sizes.get(target, 0.) + mass
        self.sums[target] = self.sums.get(target, np.zeros_like(vector)) + vector
        for ei in self.incident[node]:
            count = self.edge_units[ei][node]
            if origin is not None:
                self.counts[ei][origin] -= count
            self.counts[ei][target] += count
        self.assignment[node] = target


def update(previous, view, vectors):
    """Append evidence and refine a region declared before any old-node moves.

    This function never calls agglomerate/build. Year-sized batches may touch most
    groups; locality and actual work are reported, never assumed.
    """
    ids, x = _validated(view, vectors)
    old = previous.view
    if view.get('policy') != old.get('policy'):
        raise ValueError('Temporal input policy changed')
    if view["cutoff"] < old["cutoff"]:
        raise ValueError("Cannot move cutoff backwards")
    old_nodes = {n["id"]: n for n in old["nodes"]}
    nodes = {n["id"]: n for n in view["nodes"]}
    old_edges = {e["id"]: e for e in old["hyperedges"]}
    edges = {e["id"]: e for e in view["hyperedges"]}
    if not set(old_nodes) <= set(nodes) or not set(old_edges) <= set(edges):
        raise ValueError("Append-only update removed input")
    if any(edges[e] != old_edges[e] for e in old_edges):
        raise ValueError("Existing edge payload changed")
    if any(any(nodes[n][k] != old_nodes[n][k] for k in ("type", "surface_form")) for n in old_nodes):
        raise ValueError("Existing node identity/text changed; create an explicit version")
    for nid, record in old_nodes.items():
        if any(nodes[nid].get(k) != value for k, value in record.items() if k != 'provenance'):
            raise ValueError('Existing node metadata changed; create an explicit version')
        prior = record.get('provenance', {})
        later = nodes[nid].get('provenance', {})
        if (not set(prior.get('articles', [])) <= set(later.get('articles', [])) or
                any(later.get(k) != value for k, value in prior.items() if k != 'articles')):
            raise ValueError('Existing node evidence was removed or rewritten')
    old_x = {n["id"]: previous.vectors[i] for i, n in enumerate(old["nodes"])}
    if any(not np.array_equal(x[i], old_x[n]) for i, n in enumerate(ids) if n in old_x):
        raise ValueError("Existing embedding changed")
    if view == old:
        return State(deepcopy(old), x.copy(), dict(previous.assignment), dict(previous.parents),
                     deepcopy(previous.config), deepcopy(previous.events), deepcopy(previous.trace), previous.next_id)
    new_nodes = sorted(set(nodes) - set(old_nodes))
    new_edges = sorted(set(edges) - set(old_edges))
    cfg = deepcopy(previous.config)
    cfg['refinement_scope'] = 'new_edge_endpoints_v2'
    partition = _LocalPartition(ids, x, view["hyperedges"], previous.assignment,
                                cfg["structural_weight"], previous.assignment, cfg["change_penalty"])
    parents = dict(previous.parents)
    # Declare locality before assignment: edge-touched groups, semantic nearest
    # groups of new nodes, and nearest same-parent partners for budget repair.
    touched = {n for eid in new_edges for n in edges[eid]["members"] if n in old_nodes}
    allowed = {previous.assignment[n] for n in touched}
    initial_groups = sorted(parents)
    centers = np.array([partition.sums[g] / partition.sizes[g] for g in initial_groups])
    potential_novel = set()
    for n in new_nodes:
        i = partition.index[n]
        sim = centers @ x[i] / np.maximum(np.linalg.norm(centers, axis=1), 1e-12)
        nearest = sorted(range(len(initial_groups)), key=lambda j: (-sim[j], initial_groups[j]))[:cfg["candidate_groups"]]
        allowed.update(initial_groups[j] for j in nearest)
        if float(sim.max()) < cfg["novelty_cosine"]:
            potential_novel.add(n)
    for g in sorted(allowed.copy()):
        candidates = [h for h in initial_groups if h != g and parents[h] == parents[g]]
        if candidates:
            allowed.add(min(candidates, key=lambda h: (np.linalg.norm(partition.sums[g] / partition.sizes[g] - partition.sums[h] / partition.sizes[h]), h)))
    declared_groups = sorted(allowed)
    # Candidate groups are places to SEARCH for new evidence, not permission to
    # reshuffle every old member of those groups. Only directly touched old nodes
    # enter descent. Full-budget novelty repairs have a separately declared area.
    repair_old = {n for n, g in previous.assignment.items() if g in allowed} if potential_novel else set()
    allowed_old = sorted(touched | repair_old)
    refine_old = sorted(touched)
    next_id = previous.next_id
    merge_events, novel_births, merged_into = [], [], {}

    def merge_cost(a, b):
        na, nb = partition.sizes[a], partition.sizes[b]
        distance = partition.sums[a] / na - partition.sums[b] / nb
        cost = na * nb / (na + nb) * float(distance @ distance) / partition.total
        for ei, counts in enumerate(partition.counts):
            cost += cfg["structural_weight"] * _entropy_merge(counts[a], counts[b], partition.arity[ei]) / max(len(partition.counts), 1)
        # The surviving ID is a. Penalize old-node identity changes exactly.
        for nid, gid in partition.assignment.items():
            if gid == b and nid in previous.assignment:
                cost += cfg["change_penalty"] / partition.total * ((a != previous.assignment[nid]) - (b != previous.assignment[nid]))
        return cost

    # A geometrically novel arrival may require a new ID. At a full budget,
    # first merge an eligible same-parent pair inside the predeclared region.
    # This repair can increase the objective and is logged separately from descent.
    for n in new_nodes:
        available = sorted(g for g in allowed if g in parents and partition.sizes.get(g, 0) > 0)
        similarities = [float(partition.sums[g] @ x[partition.index[n]]) / max(np.linalg.norm(partition.sums[g]), 1e-12) for g in available]
        if n in potential_novel and max(similarities) < cfg["novelty_cosine"]:
            pair = [(a, b) for i, a in enumerate(available) for b in available[i + 1:] if parents[a] == parents[b]]
            free = len(parents) < cfg["budgets"][1]
            if free or pair:
                parent = parents[available[int(np.argmax(similarities))]]
                if not free:
                    a, b = min(pair, key=lambda p: (merge_cost(*p), p))
                    repair_cost = merge_cost(a, b)
                    for nid in sorted(list(partition.assignment)):
                        if partition.assignment[nid] == b:
                            partition.move(nid, a)
                    del parents[b]
                    merged_into[b] = a
                    allowed.remove(b)
                    merge_events.append({"kind": "merge", "level": 1, "group_id": a,
                                         "node_ids": [], "previous_group_ids": [g for g in (a, b) if g in previous.parents],
                                         "retired_group_id": b, "objective_delta": repair_cost})
                born = f"L1-new{next_id:06d}"
                next_id += 1
                parents[born] = parent
                allowed.add(born)
                partition.move(n, born)
                novel_births.append(born)
                continue
        target = min(sorted(allowed), key=lambda g: (partition.delta(n, g), g))
        partition.move(n, target)
    gains, moved = [], set()
    for _ in range(cfg["refine_passes"]):
        changes = 0
        for n in sorted(set(refine_old) | set(new_nodes)):
            current = partition.assignment[n]
            if partition.sizes[current] <= 1:
                continue
            target = min(sorted(allowed), key=lambda g: (partition.delta(n, g), g))
            gain = partition.delta(n, target)
            if gain < -1e-12:
                partition.move(n, target)
                gains.append(float(gain))
                moved.add(n)
                changes += 1
        if not changes:
            break
    assignment = partition.assignment
    # Coarse refinement operates on actual collapsed hyperedges, with group mass
    # and sufficient mean vectors. A whole fine group moves as one laminar unit.
    fine_ids = sorted(parents)
    masses = np.array([partition.sizes[g] for g in fine_ids])
    means = np.array([partition.sums[g] / partition.sizes[g] for g in fine_ids])
    coarsened = collapse(view["hyperedges"], assignment)
    coarse_old = {g: p for g, p in previous.parents.items() if g in parents}
    coarse = _LocalPartition(fine_ids, means, coarsened, parents, cfg["structural_weight"],
                             coarse_old, cfg["change_penalty"], masses=masses)
    allowed_top = {parents[g] for g in allowed}
    coarse_moves = []
    for g in sorted(allowed):
        old_members = {n for n, current_g in assignment.items() if current_g == g and n in old_nodes}
        if not old_members <= touched:
            continue
        current = coarse.assignment[g]
        if sum(coarse.assignment[h] == current for h in fine_ids) <= 1:
            continue
        target = min(sorted(allowed_top), key=lambda h: (coarse.delta(g, h), h))
        gain = coarse.delta(g, target)
        if gain < -1e-12:
            coarse.move(g, target)
            coarse_moves.append(g)
    parents = coarse.assignment
    previous_maps = [{n: previous.parents[g] for n, g in previous.assignment.items()}, previous.assignment]
    current_maps = [{n: parents[g] for n, g in assignment.items()}, assignment]
    events = []
    for level, (before, after) in enumerate(zip(previous_maps, current_maps)):
        for gid in sorted(set(before.values()) | set(after.values())):
            oldset = {n for n, g in before.items() if g == gid}
            newset = {n for n, g in after.items() if g == gid}
            if oldset == newset:
                continue
            kind = "birth" if not oldset else "death" if not newset else "growth" if oldset <= newset else "move"
            events.append({"kind": kind, "level": level, "group_id": gid, "node_ids": sorted(newset - oldset),
                           "removed_node_ids": sorted(oldset - newset), "previous_group_ids": [gid] if oldset else []})
    events.extend({"kind": "birth", "level": 2, "group_id": "node:" + n, "node_ids": [n], "previous_group_ids": []} for n in new_nodes)
    # Public events refer to the two persisted snapshots, not transient groups
    # born and retired inside a large batch. Keep detailed repair operations in
    # the trace, and map their ancestry to surviving IDs for the event log.
    merge_ancestry = defaultdict(set)
    for event in merge_events:
        survivor = event['group_id']
        while survivor in merged_into:
            survivor = merged_into[survivor]
        merge_ancestry[survivor].update(event['previous_group_ids'])
    for survivor, ancestors in sorted(merge_ancestry.items()):
        final_members = {n for n, g in assignment.items() if g == survivor}
        supported = [g for g in sorted(ancestors)
                     if {n for n, old_g in previous.assignment.items() if old_g == g} <= final_members]
        if len(supported) >= 2:
            events.append({'kind': 'merge', 'level': 1, 'group_id': survivor,
                           'node_ids': [], 'previous_group_ids': supported})
    for born in sorted(set(parents) - set(previous.parents)):
        predecessors = sorted({previous.assignment[n] for n, g in assignment.items()
                               if g == born and n in previous.assignment})
        for old_group in predecessors:
            destinations = {assignment[n] for n, g in previous.assignment.items() if g == old_group}
            if len(destinations) > 1:
                events.append({'kind': 'split', 'level': 1, 'group_id': born,
                               'node_ids': sorted(n for n, g in assignment.items() if g == born and previous.assignment.get(n) == old_group),
                               'previous_group_ids': [old_group]})
    trace = {"initial_builder_calls": 0, "new_nodes": new_nodes, "new_edges": new_edges,
             "allowed_groups": declared_groups, "allowed_old_nodes": allowed_old,
             "refinement_old_nodes": refine_old,
             "budget_repair_allowed_old_nodes": sorted(repair_old),
             "old_node_moves": sorted(n for n in old_nodes if assignment[n] != previous.assignment[n]),
             "coarse_group_moves": coarse_moves, "accepted_fine_objective_deltas": gains,
             "coarsened_edges_used": len(coarsened),
             "novel_topic_births": [g for g in novel_births if g in parents],
             "budget_repair_operations": merge_events}
    trace['global_input_scan'] = {'nodes': len(ids), 'hyperedges': len(view['hyperedges']),
                                 'reason': 'Sufficient statistics are reconstructed from the saved assignment; membership refinement is local, initialization and export still scan the snapshot.'}
    return State(deepcopy(view), x.copy(), assignment, parents, deepcopy(cfg), events, trace, next_id)
