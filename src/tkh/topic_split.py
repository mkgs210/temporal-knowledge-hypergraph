"""One bounded, atomic old-topic split proposal in construction geometry.

Complete-link bisection proposes two internally cohesive children. Acceptance
uses the FULL spherical/native/temporal criterion, never an independent quality
encoder or reference answers. This local heuristic does not establish scientific
coherence or downstream utility. Coarse parents and their hard budget persist.
"""
from copy import deepcopy
import hashlib
import itertools
import json
import math

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform

from .novelty_edits import _inputs, _old_path_moves, topic_edit_criterion


def propose_old_topic_split(previous, view, vectors, *, starting_assignment,
                            starting_parents, metadata_assignment, max_old_edits,
                            calibration):
    """Try the most dispersed touched old fine group, with ≥2 OLD nodes/child.

    At fine capacity, a same-parent donor merge EXCLUDING the split target must
    free its slot. Both donor-ID orientations and both retained-child choices
    are checked, charging every changed old scientific/metadata hierarchy path.
    Metadata attached to the split target stays with its persistent ID. All
    proposals are atomic; deferred proposals leave the supplied paths intact.

    Computational scope is deliberately one touched candidate, at most four
    donor-orientation/retention trials per same-parent donor pair. The dense
    complete-link proposal costs O(m²) memory for that topic, not whole corpus.
    Full criterion evaluations scan current scientific records/native edges.
    """
    if type(max_old_edits) is not int or max_old_edits < 0:
        raise ValueError('Old edit budget must be a nonnegative integer')
    nodes, x, science = _inputs(view, vectors)
    old_nodes, old_x, old_science = _inputs(previous.view, previous.vectors)
    if set(old_nodes) - set(nodes):
        raise ValueError('Append-only split cannot remove old records')
    if view['cutoff'] < previous.view['cutoff']:
        raise ValueError('Temporal cutoff moved backwards')
    if any(not np.array_equal(x[n], old_x[n]) for n in old_nodes):
        raise ValueError('Existing embedding changed')
    if calibration.get('cutoff', math.inf) > previous.view['cutoff']:
        raise ValueError('Future calibration evidence')
    assignment, parents, metadata = map(dict, (starting_assignment, starting_parents, metadata_assignment))
    old_meta = set(old_nodes) - set(old_science)
    if not old_meta <= metadata.keys() or any(n in assignment for n in old_meta):
        raise ValueError('Old metadata paths must be supplied separately')
    coarse_budget, fine_budget = previous.config['budgets']
    if len(parents) > fine_budget or len(set(parents.values())) > coarse_budget:
        raise ValueError('Starting hierarchy exceeds fixed budgets')
    temporal_weight = calibration.get('temporal_weight', .05)
    before = topic_edit_criterion(previous, view, vectors, assignment, parents,
                                  temporal_weight=temporal_weight, metadata_assignment=metadata)
    if before['old_path_moves'] > max_old_edits:
        raise ValueError('Starting paths already exceed old edit budget')
    trace = {'status': 'no_accepted_edit', 'candidate_count': 0, 'candidate_limit': 1,
             'criterion_before': before, 'criterion_after': before, 'deferred': [],
             'max_old_edits': max_old_edits, 'calibration_cutoff': calibration.get('cutoff'),
             'calibration_sha256': calibration.get('input_sha256'), 'utility_claim': False,
             'proposal_policy': 'highest spherical dispersion touched old fine group; complete-link two-way',
             'alarm_interpretation': 'Construction-space cohesive split heuristic; not calibrated scientific novelty.'}
    events = []
    def result():
        moved = _old_path_moves(previous, assignment, parents, metadata)
        if len(moved) > max_old_edits or len(parents) > fine_budget or len(set(parents.values())) > coarse_budget:
            raise AssertionError('Atomic split exceeded hard budget')
        if set(assignment) != set(science) or set(assignment.values()) != set(parents):
            raise AssertionError('Atomic split lost scientific members or left an empty fine group')
        return {'accepted': bool(events), 'assignment': assignment, 'parents': parents,
                'metadata_assignment': metadata, 'events': events, 'trace': trace,
                'moved_old_ids': moved, 'moved_old_scientific_ids': [n for n in moved if n in set(old_science)]}
    if calibration.get('status') != 'available':
        trace['status'] = 'calibration_unavailable'
        return result()
    cohesion = calibration.get('cohesion_threshold')
    min_old = calibration.get('split_min_old_child', 2)
    min_gain = calibration.get('split_min_semantic_gain', 0.)
    if (type(min_old) is not int or min_old < 2 or not isinstance(cohesion, (int, float))
            or not math.isfinite(cohesion) or not -1 <= cohesion <= 1
            or not isinstance(min_gain, (float, int)) or not math.isfinite(min_gain) or min_gain < 0):
        raise ValueError('Invalid split calibration thresholds')
    trace.update(cohesion_threshold=cohesion, minimum_old_child=min_old,
                 minimum_semantic_gain=min_gain)
    old_edges = {e['id']: e for e in previous.view['hyperedges']}
    touched = set(science) - set(old_science)
    for edge in view['hyperedges']:
        if edge['id'] not in old_edges:
            touched.update(set(edge['members']) & set(science))
        elif edge != old_edges[edge['id']]:
            raise ValueError('Existing hyperedge payload changed')
    if old_edges.keys() - {e['id'] for e in view['hyperedges']}:
        raise ValueError('Append-only split cannot remove old hyperedges')
    groups = sorted(set(assignment[n] for n in touched) & set(previous.parents))
    trace['touched_old_group_ids'] = groups
    eligible = [g for g in groups if sum(assignment[n] == g for n in old_science) >= 2 * min_old]
    trace['candidate_count'] = len(eligible)
    if not eligible:
        if groups:
            trace['deferred'].append({'reason': 'too_few_old_members', 'group_ids': groups})
        return result()
    members = {g: sorted(n for n in science if assignment[n] == g) for g in eligible}
    def dispersion(ns):
        return len(ns) - float(np.linalg.norm(np.stack([x[n] for n in ns]).sum(axis=0)))
    target = min(eligible, key=lambda g: (-dispersion(members[g]), str(g)))
    ns = members[target]
    original = {n for n in old_science if previous.assignment[n] == target}
    if len(set(ns) & original) < 2 * min_old:
        trace['deferred'].append({'group_id': target, 'reason': 'too_few_original_old_members'})
        return result()
    xx = np.stack([x[n] for n in ns])
    distance = np.maximum(0., 1 - np.clip(xx @ xx.T, -1., 1.))
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0.)
    codes = cut_tree(linkage(squareform(distance, checks=True), method='complete'), n_clusters=2).ravel()
    children = [[n for n, code in zip(ns, codes, strict=True) if code == i] for i in (0, 1)]
    trace.update(candidate_group_id=target, candidate_members=ns, proposed_children=children)
    if any(len(set(child) & set(old_science)) < min_old for child in children):
        trace['deferred'].append({'group_id': target, 'reason': 'too_few_old_members_in_child'})
        return result()
    if any(len(set(child) & original) < min_old for child in children):
        trace['deferred'].append({'group_id': target, 'reason': 'too_few_original_old_members'})
        return result()
    minimum_cosines = []
    for child in children:
        cx = np.stack([x[n] for n in child])
        sims = np.clip(cx @ cx.T, -1., 1.)
        minimum_cosines.append(float(sims[np.triu_indices(len(child), 1)].min()))
    gain = (dispersion(ns) - sum(dispersion(child) for child in children)) / len(ns)
    trace.update(child_minimum_cosines=minimum_cosines, semantic_gain_per_candidate_node=gain)
    if min(minimum_cosines) < cohesion - 1e-12:
        trace['deferred'].append({'group_id': target, 'reason': 'incoherent_children'})
        return result()
    if gain <= min_gain + 1e-12:
        trace['deferred'].append({'group_id': target, 'reason': 'insufficient_semantic_gain'})
        return result()
    parent = parents[target]
    donor_ids = sorted(g for g in parents if g != target and parents[g] == parent
                       and g in previous.parents)
    merges = [None] if len(parents) < fine_budget else list(itertools.permutations(donor_ids, 2))
    token = hashlib.sha256(json.dumps([target, children], separators=(',', ':')).encode()).hexdigest()[:16]
    born = f'L1-split-{view["cutoff"]}-{token}'
    if born in parents:
        raise ValueError('Split topic ID collision')
    feasible, best, evaluated = 0, None, 0
    for merge in merges:
        for retained in (0, 1):
            proposed, next_parents, next_metadata = dict(assignment), dict(parents), dict(metadata)
            lineage = []
            if merge:
                survivor, retired = merge
                for n in science:
                    if proposed[n] == retired:
                        proposed[n] = survivor
                for n, g in metadata.items():
                    if g == retired:
                        next_metadata[n] = survivor
                del next_parents[retired]
                lineage.append({'kind': 'merge', 'level': 1, 'group_id': survivor,
                                'previous_group_ids': sorted(merge), 'retired_group_ids': [retired],
                                'parent_id': parent})
            next_parents[born] = parent
            moved_child = children[1-retained]
            for n in moved_child:
                proposed[n] = born
            moved = _old_path_moves(previous, proposed, next_parents, next_metadata)
            if len(moved) > max_old_edits:
                continue
            feasible += 1
            cost = topic_edit_criterion(previous, view, vectors, proposed, next_parents,
                                        temporal_weight=temporal_weight, metadata_assignment=next_metadata)
            evaluated += 1
            if cost['total'] < before['total'] - 1e-12 and (best is None or cost['total'] < best[0]['total'] - 1e-12):
                lineage.append({'kind': 'split', 'level': 1, 'group_id': born,
                                'retained_group_id': target, 'new_group_id': born,
                                'previous_group_ids': [target], 'parent_id': parent,
                                'node_ids': list(moved_child),
                                'old_child_counts': [len(set(child) & set(old_science)) for child in children]})
                best = cost, proposed, next_parents, next_metadata, lineage
    trace.update(feasible_transactions=feasible, full_criterion_evaluations=evaluated,
                 candidate_split_attempts=1)
    if best is None:
        reason = 'no_same_parent_donor_pair' if not merges else 'old_edit_budget' if not feasible else 'no_beneficial_transaction'
        trace['deferred'].append({'group_id': target, 'reason': reason})
    else:
        cost, assignment, parents, metadata, events = best
        trace.update(status='accepted', criterion_after=cost,
                     declared_edit_ids=sorted(set(ns) | {n for n, g in starting_assignment.items()
                                                        if g not in parents}
                                              | {n for n in metadata if metadata[n] != metadata_assignment[n]}))
    return result()
