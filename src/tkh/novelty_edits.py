"""Cutoff-safe novelty diagnostics and atomic, budgeted fine-topic transactions.

This is a two-stage proposal rule, not a new global optimizer. It uses ONLY the
construction vectors and native relations. Leave-source-out scores calibrate an
empirical alarm threshold, not a conformal guarantee under corpus drift.

Acceptance criterion: mean spherical dispersion at fine and coarse levels plus
native weighted KM1 at both levels plus a penalty for changed old hierarchy
paths. Spherical dispersion is n - ||sum(x)|| per group (Dhillon/Modha's cosine
prototype objective). A split can therefore improve semantic fit even when a
positive-weight cut objective alone cannot justify it. All terms are returned.
No retrieval-quality implication follows from acceptance.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import math

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .divisive_hierarchy import METADATA_TYPES, scientific_edges
from .divisive_solver import objective


def _inputs(view, vectors):
    nodes = list(view['nodes'])
    ids = [n['id'] for n in nodes]
    x = np.asarray(vectors, dtype=float)
    if (len(set(ids)) != len(ids) or not ids or x.ndim != 2 or len(x) != len(ids)
            or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-6)):
        raise ValueError('Expected aligned, finite unit embeddings and unique node IDs')
    if any(n.get('observed_at', view['cutoff']) > view['cutoff'] for n in nodes):
        raise ValueError('Future node evidence')
    edge_ids = set()
    pool = set(ids)
    for edge in view['hyperedges']:
        members = edge['members']
        if (edge['id'] in edge_ids or not members or len(members) != len(set(members))
                or set(members) - pool):
            raise ValueError('Invalid native hyperedge IDs or endpoints')
        edge_ids.add(edge['id'])
        for year in (edge.get('year'), edge.get('available_year'),
                     edge.get('provenance', {}).get('article_year')):
            if year is not None and (type(year) is not int or year > view['cutoff']):
                raise ValueError('Future or invalid hyperedge evidence')
    science = [n['id'] for n in nodes if n['type'] not in METADATA_TYPES]
    if not science:
        raise ValueError('No scientific nodes')
    return {n['id']: n for n in nodes}, {n: v for n, v in zip(ids, x, strict=True)}, science


def calibrate_novelty(view, vectors, *, alpha=.1, min_cohort_size=3,
                      temporal_weight=.05,cohesion_quantile=.1,min_cohort_sources=1):
    """Freeze first-view leave-source-out nearest-neighbor diagnostics.

    Every query/reference pair has disjoint membership of the held-out source;
    shared-source nodes are excluded from that source's reference pool. Nodes
    shared across papers may appear in multiple folds. The pooled quantile is
    descriptive and is NOT an independent-source false-positive guarantee.
    """
    if not 0 < alpha < 1 or type(min_cohort_size) is not int or min_cohort_size < 2:
        raise ValueError('Invalid calibration quantile or minimum cohort size')
    if not 0<cohesion_quantile<1 or type(min_cohort_sources)is not int or min_cohort_sources<1:
        raise ValueError('Invalid cohesion quantile or minimum cohort source count')
    if not math.isfinite(temporal_weight) or temporal_weight < 0:
        raise ValueError('Invalid temporal penalty')
    nodes, x, science = _inputs(view, vectors)
    sources = {n: set(nodes[n].get('provenance', {}).get('articles', [])) for n in science}
    rows, cohesion, folds = [], [], []
    for source in sorted(set().union(*sources.values()), key=str):
        held = sorted(n for n in science if source in sources[n])
        reference = sorted(n for n in science if source not in sources[n])
        if not reference:
            continue
        folds.append({'held_out_source': source, 'held_out_ids': held, 'reference_ids': reference})
        reference_x = np.stack([x[n] for n in reference])
        for n in held:
            rows.append({'held_out_source': source, 'node_id': n,
                         'max_cosine': float(np.clip(reference_x @ x[n], -1., 1.).max())})
        if len(held) > 1:
            hx = np.stack([x[n] for n in held])
            sims = np.clip(hx @ hx.T, -1., 1.)
            cohesion.extend(float(sims[i, j]) for i, j in itertools.combinations(range(len(held)), 2))
    digest = hashlib.sha256()
    digest.update(json.dumps(view, sort_keys=True, separators=(',', ':')).encode())
    digest.update(np.asarray(vectors, dtype='<f8').tobytes())
    available = bool(rows and cohesion)
    return {'schema_version': 'novelty-calibration-v1', 'cutoff': view['cutoff'],
            'input_sha256': digest.hexdigest(), 'status': 'available' if available else 'unavailable',
            'alpha': alpha, 'min_cohort_size': min_cohort_size, 'temporal_weight': temporal_weight,
            'cohesion_quantile':cohesion_quantile,'min_cohort_sources':min_cohort_sources,
            'novelty_threshold': float(np.quantile([r['max_cosine'] for r in rows], alpha)) if available else None,
            'cohesion_threshold': float(np.quantile(cohesion, cohesion_quantile)) if available else None,
            'scores': rows, 'folds': folds, 'cohesion_reference_values': cohesion,
            'source_count': len({r['held_out_source'] for r in rows}),
            'guaranteed_false_alarm_control': False,
            'limitations': ['Pooled dependent leave-source-out observations; no exchangeability guarantee.',
                            'Alarm thresholds do not establish scientific novelty or useful retrieval.',
                            'Cohesion is measured in construction space, never an independent quality score.']}


def _old_path_moves(previous, assignment, parents, metadata_assignment):
    complete = dict(metadata_assignment)
    complete.update(assignment)
    if not set(previous.assignment) <= complete.keys() or any(complete[n] not in parents for n in previous.assignment):
        raise ValueError('Old metadata/scientific path lacks a surviving fine-group parent')
    return sorted(n for n, old_group in previous.assignment.items()
                  if complete[n] != old_group or parents[complete[n]] != previous.parents[old_group])


def topic_edit_criterion(previous, view, vectors, assignment, parents, *, temporal_weight=.05,
                         metadata_assignment=None):
    """Evaluate the complete two-level proposal criterion, including old paths."""
    _, x, science = _inputs(view, vectors)
    if set(assignment) != set(science) or set(assignment.values()) != set(parents):
        raise ValueError('Scientific assignment must cover exactly the occupied fine groups')
    if not math.isfinite(temporal_weight) or temporal_weight < 0:
        raise ValueError('Invalid temporal penalty')
    coarse = {n: parents[assignment[n]] for n in science}
    def dispersion(mapping):
        sums = {}
        for n in science:
            sums[mapping[n]] = sums.get(mapping[n], np.zeros_like(x[n])) + x[n]
        return max(0., (len(science) - math.fsum(float(np.linalg.norm(v)) for v in sums.values())) / len(science))
    metadata = {n['id']: previous.assignment[n['id']] for n in previous.view['nodes']
                if n['type'] in METADATA_TYPES} if metadata_assignment is None else metadata_assignment
    moved = _old_path_moves(previous, assignment, parents, metadata)
    native, _ = scientific_edges(view, science, previous.config.get('gamma', .03),
                                previous.config.get('normalizations', {}).get('native_papers'))
    values = {'semantic_fine': dispersion(assignment), 'semantic_coarse': dispersion(coarse),
              'native_fine': objective(native, assignment), 'native_coarse': objective(native, coarse),
              'temporal': temporal_weight * len(moved) / max(1, len(previous.assignment)),
              'old_path_moves': len(moved)}
    values['total'] = math.fsum(values[k] for k in ('semantic_fine', 'semantic_coarse',
                                                  'native_fine', 'native_coarse', 'temporal'))
    return values


def propose_topic_edits(previous, view, vectors, *, max_old_edits, calibration,
                       starting_assignment=None, starting_parents=None):
    """Return scientific placement and optional atomic fine birth/merge edits.

    Existing coarse IDs are fixed. At fine capacity a new cohort may acquire a
    slot only by merging two existing same-parent groups, charging EVERY moved
    old path. No input/state object is mutated. Deferred proposals retain legal
    nearest-old placement; even an unavailable calibration never drops records.
    """
    if type(max_old_edits) is not int or max_old_edits < 0:
        raise ValueError('Old-node edit budget must be a nonnegative integer')
    nodes, x, science = _inputs(view, vectors)
    _, old_x, old_science = _inputs(previous.view, previous.vectors)
    if set(old_science) - set(science):
        raise ValueError('Append-only proposal cannot remove old scientific nodes')
    if any(not np.array_equal(x[n], old_x[n]) for n in old_science):
        raise ValueError('Existing embedding changed')
    if calibration.get('cutoff', float('inf')) > previous.view['cutoff']:
        raise ValueError('Future calibration evidence')
    coarse_budget, fine_budget = previous.config['budgets']
    parents = dict(previous.parents if starting_parents is None else starting_parents)
    if len(parents) > fine_budget or len(set(parents.values())) > coarse_budget:
        raise ValueError('Previous hierarchy exceeds its fixed budgets')
    assignment = ({n: previous.assignment[n] for n in old_science}
                  if starting_assignment is None else dict(starting_assignment))
    if starting_assignment is not None and set(assignment) != set(science):
        raise ValueError('Starting assignment must include every current scientific node')
    metadata = {n['id']: previous.assignment[n['id']] for n in previous.view['nodes']
                if n['type'] in METADATA_TYPES}
    reference = sorted(old_science)
    old_matrix = np.stack([x[n] for n in reference])
    new = sorted(set(science) - set(old_science))
    novelty_scores = {}
    for n in new:
        scores = np.clip(old_matrix @ x[n], -1., 1.)
        nearest = int(np.argmax(scores))
        if starting_assignment is None:
            assignment[n] = previous.assignment[reference[nearest]]
        novelty_scores[n] = float(scores[nearest])
    temporal_weight = calibration.get('temporal_weight', .05)
    before = topic_edit_criterion(previous, view, vectors, assignment, parents,
                                 temporal_weight=temporal_weight, metadata_assignment=metadata)
    if before['old_path_moves'] > max_old_edits:
        raise ValueError('Starting hierarchy paths already exceed old-record edit budget')
    trace = {'status': 'no_accepted_edit', 'calibration_sha256': calibration.get('input_sha256'),
             'calibration_cutoff': calibration.get('cutoff'), 'max_old_edits': max_old_edits,
             'novelty_scores': novelty_scores, 'criterion_before': before, 'deferred': [],
             'criterion': 'mean spherical dispersion(fine+coarse) + native KM1(fine+coarse) + temporal old-path fraction',
             'utility_claim': False}
    events = []
    def result():
        moved = _old_path_moves(previous, assignment, parents, metadata)
        if len(moved) > max_old_edits or len(parents) > fine_budget or len(set(parents.values())) > coarse_budget:
            raise AssertionError('Atomic topic transaction exceeded hard budget')
        trace['criterion_after'] = topic_edit_criterion(previous, view, vectors, assignment, parents,
                                                       temporal_weight=temporal_weight, metadata_assignment=metadata)
        trace['moved_old_ids'] = moved
        trace['moved_old_scientific_ids'] = [n for n in moved if n in set(old_science)]
        trace['status'] = 'accepted' if events else trace['status']
        return {'accepted': bool(events), 'assignment': assignment, 'parents': parents,
                'metadata_assignment': metadata, 'events': events, 'trace': trace,
                'moved_old_ids': moved, 'moved_old_scientific_ids': trace['moved_old_scientific_ids']}
    if calibration.get('status') != 'available':
        trace['status'] = 'calibration_unavailable'
        return result()
    threshold, cohesion = calibration['novelty_threshold'], calibration['cohesion_threshold']
    minimum = calibration['min_cohort_size']
    if (not all(isinstance(v, (float, int)) and math.isfinite(v) and -1 <= v <= 1 for v in (threshold, cohesion))
            or type(minimum) is not int or minimum < 2):
        raise ValueError('Invalid calibration thresholds')
    alarmed = [n for n in new if novelty_scores[n] < threshold]
    if not alarmed:
        return result()
    if len(alarmed) == 1:
        cohorts = [alarmed]
    else:
        xx = np.stack([x[n] for n in alarmed])
        distance = np.maximum(0., 1 - np.clip(xx @ xx.T, -1., 1.))
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0.)
        codes = fcluster(linkage(squareform(distance, checks=True), method='complete'),
                         t=max(0., 1 - cohesion) + 1e-12, criterion='distance')
        cohorts = [[n for n, g in zip(alarmed, codes, strict=True) if g == code] for code in sorted(set(codes))]
    for cohort in sorted(cohorts, key=lambda ns: (-len(ns), tuple(ns))):
        if len(cohort) < minimum:
            trace['deferred'].append({'members': cohort, 'reason': 'cohort_too_small'})
            continue
        cohort_sources=set().union(*(set(nodes[n].get('provenance',{}).get('articles',[])) for n in cohort))
        if len(cohort_sources)<calibration.get('min_cohort_sources',1):
            trace['deferred'].append({'members':cohort,'reason':'insufficient_distinct_sources',
                                      'declared_source_count':len(cohort_sources)})
            continue
        parent_votes = Counter(parents[assignment[n]] for n in cohort)
        parent = min(parent_votes, key=lambda p: (-parent_votes[p], str(p)))
        token = hashlib.sha256(json.dumps(cohort, separators=(',', ':')).encode()).hexdigest()[:16]
        born = f'L1-novel-{view["cutoff"]}-{token}'
        if born in parents:
            raise ValueError('Novel topic ID collision')
        baseline_cost = topic_edit_criterion(previous, view, vectors, assignment, parents,
                                              temporal_weight=temporal_weight, metadata_assignment=metadata)
        merges = [None]
        if len(parents) == fine_budget:
            # Only previously existing groups under the SAME coarse parent may
            # donate a slot; accepted novel groups cannot be silently erased.
            local = sorted(g for g in previous.parents if g in parents and parents[g] == parent)
            merges = list(itertools.permutations(local, 2))
        best = None
        feasible = 0
        for merge in merges:
            proposed, next_parents, next_metadata = dict(assignment), dict(parents), dict(metadata)
            lineage = []
            if merge is not None:
                survivor, retired = merge
                for n in science:
                    if proposed[n] == retired:
                        proposed[n] = survivor
                for n, group in metadata.items():
                    if group == retired:
                        next_metadata[n] = survivor
                del next_parents[retired]
                lineage.append({'kind': 'merge', 'level': 1, 'group_id': survivor,
                                'previous_group_ids': sorted(merge), 'retired_group_ids': [retired],
                                'parent_id': parent})
            origin = sorted({assignment[n] for n in cohort})
            next_parents[born] = parent
            for n in cohort:
                proposed[n] = born
            changed = _old_path_moves(previous, proposed, next_parents, next_metadata)
            if len(changed) > max_old_edits:
                continue
            feasible += 1
            cost = topic_edit_criterion(previous, view, vectors, proposed, next_parents,
                                        temporal_weight=temporal_weight, metadata_assignment=next_metadata)
            if cost['total'] < baseline_cost['total'] - 1e-12 and (best is None or cost['total'] < best[0]['total'] - 1e-12):
                lineage.append({'kind': 'birth', 'level': 1, 'group_id': born, 'parent_id': parent,
                                'previous_group_ids': origin, 'node_ids': list(cohort),
                                'criterion': 'cohesive new-node cohort; complete criterion improved'})
                best = cost, proposed, next_parents, next_metadata, lineage
        if best is None:
            reason = 'no_same_parent_merge' if not merges else 'old_edit_budget' if not feasible else 'no_beneficial_transaction'
            trace['deferred'].append({'members': cohort, 'reason': reason})
        else:
            cost, assignment, parents, metadata, lineage = best
            events.extend(lineage)
            trace.setdefault('transactions', []).append({'members': cohort, 'before': baseline_cost, 'after': cost,
                                                         'events': deepcopy(lineage)})
    return result()
