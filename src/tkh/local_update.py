"""Bounded native refinement with fixed outside paths and temporal anchors.

The edit budget concerns old scientific records, not the number of new records.
Validation and exact new-to-old neighbor queries can still scan global inputs.
Original topic count limits are never changed by this updater.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .divisive_hierarchy import (
    DivisiveState, _input, _affinities, _semantic_edges, scientific_edges,
    _restrict, _metadata, _events, _trace_finish,
)
from .divisive_solver import boundary_quotient, objective, quotient, solve_partition
from .temporal_input import delta


def regularized_boundary(edges, editable, assignment, old_assignment, penalty):
    """Exact KM1 contraction plus unary old-label movement penalties.

    Artificial anchor masses are accounted for separately in solver capacities;
    they do not enter scientific endpoint counts or the original cut criterion.
    """
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Movement penalty must be finite and nonnegative')
    contracted, fixed = boundary_quotient(edges, editable, assignment)
    if penalty:
        for node in sorted(set(editable) & old_assignment.keys()):
            part = old_assignment[node]
            anchor = ('__temporal_anchor__', part)
            fixed[anchor] = part
            contracted.append({'id': 'temporal:' + node, 'kind': 'temporal',
                               'members': [node, anchor], 'weight': penalty})
    return contracted, fixed


def select_frontier(old_ids, new_ids, affinity, native, changed_edges, limit):
    """Rank only directly changed incident mass; never close over whole groups."""
    old_ids, new_ids = list(old_ids), list(new_ids)
    if type(limit) is not int or limit < 0:
        raise ValueError('Old edit limit must be a nonnegative integer')
    scores = np.asarray(affinity[:len(old_ids), len(old_ids):].sum(axis=1)).ravel()
    positions = {n:i for i,n in enumerate(old_ids)}
    for edge in native:
        if edge['id'] in changed_edges and edge['weight'] > 0:
            endpoints = set(edge['members']) & positions.keys()
            for node in endpoints:
                scores[positions[node]] += edge['weight'] / max(1, len(endpoints))
    ranked = sorted((n for n in old_ids if scores[positions[n]] > 0),
                    key=lambda n:(-float(scores[positions[n]]), n))
    return ranked[:limit], ranked[limit:], {n:float(scores[positions[n]]) for n in ranked}


def _validate_previous(previous, view, vectors):
    ids, x, science = _input(view, vectors)
    change = delta(previous.view, view)
    nodes = {n['id']:n for n in view['nodes']}
    for record in previous.view['nodes']:
        now = nodes[record['id']]
        if any(now.get(k) != v for k,v in record.items() if k != 'provenance'):
            raise ValueError('Existing node changed')
        a,b = record.get('provenance',{}),now.get('provenance',{})
        if not set(a.get('articles',[])) <= set(b.get('articles',[])) or any(
                b.get(k) != v for k,v in a.items() if k != 'articles'):
            raise ValueError('Existing node provenance removed or rewritten')
    positions = {n:i for i,n in enumerate(ids)}
    if x.shape[1] != previous.vectors.shape[1] or any(
            not np.array_equal(x[positions[n['id']]], previous.vectors[i])
            for i,n in enumerate(previous.view['nodes'])):
        raise ValueError('Existing embedding changed')
    return ids,x,science,change


def _solve_region(edges, domain, active, assignment, previous_assignment, penalty, seed):
    labels = sorted({assignment[n] for n in domain})
    codes = {g:i for i,g in enumerate(labels)}
    start = {n:codes[assignment[n]] for n in domain}
    old = {n:codes[g] for n,g in previous_assignment.items() if n in start and g in codes}
    q,fixed = regularized_boundary(_restrict(edges, domain), active, start, old, penalty)
    local_ids = sorted(active) + sorted(fixed,key=str)
    initial = {n:start[n] for n in active} | fixed
    masses = {n:1 for n in active}
    for terminal,part in fixed.items():
        masses[terminal] = (sum(start[n] == part for n in domain if n not in active)
                            if terminal[0] == '__fixed_terminal__' else 1)
    loads = [sum(masses[n] for n,g in initial.items() if g == i) for i in range(len(labels))]
    usual = math.ceil(1.5 * math.ceil(sum(masses.values()) / len(labels)))
    caps = [max(load,usual) for load in loads]
    result,trace = solve_partition(local_ids,q,len(labels),initial=initial,fixed=fixed,
        masses=masses,block_caps=caps,seed=seed)
    proposed = dict(assignment)
    proposed.update({n:labels[result[n]] for n in active})
    # Occupied solver anchors cannot stand in for a real scientific topic.
    if {proposed[n] for n in domain} != set(labels):
        trace.update(accepted=False,status='rejected_empty_scientific_group')
        proposed = dict(assignment)
    before = objective(edges,assignment)
    after = objective(edges,proposed)
    old_before = sum(assignment[n] != g for n,g in previous_assignment.items())
    old_after = sum(proposed[n] != g for n,g in previous_assignment.items())
    exact = after-before + penalty*(old_after-old_before)
    reduced = objective(q,result)-objective(q,initial)
    if trace['accepted'] and (not math.isclose(exact,reduced,abs_tol=1e-10) or exact > 1e-10):
        raise AssertionError('Local temporal/native objective delta mismatch or increase')
    trace.update(global_regularized_delta=exact,contracted_delta=reduced,
                 scientific_loads=[sum(proposed[n] == g for n in domain) for g in labels],
                 artificial_anchor_mass=sum(masses[t] for t in fixed if t[0]=='__temporal_anchor__'))
    return proposed,trace


def update_local(previous, view, vectors, *, max_old_fraction=.1, movement_strength=.05,
                 mode='bounded', novelty_calibration=None, allow_old_topic_split=False):
    """Refine a declared number of old records without restarting construction."""
    if not math.isfinite(max_old_fraction) or not 0 <= max_old_fraction <= 1:
        raise ValueError('Old edit fraction must lie in [0,1]')
    if not math.isfinite(movement_strength) or movement_strength < 0:
        raise ValueError('Movement strength must be finite and nonnegative')
    if mode not in {'bounded','append','unbounded'}:
        raise ValueError('Unknown temporal control mode')
    if novelty_calibration is not None and mode != 'bounded':
        raise ValueError('Novelty transactions require bounded mode')
    if allow_old_topic_split and novelty_calibration is None:
        raise ValueError('Old topic split requires frozen first-view calibration')
    ids,x,current_science,change = _validate_previous(previous,view,vectors)
    cfg = deepcopy(previous.config)
    old_science = list(cfg['scientific_ids']); old_set = set(old_science)
    new = [n for n in current_science if n not in old_set]
    science = old_science + new
    cfg['scientific_ids'] = science
    limit = (0 if mode=='append' else len(old_science) if mode=='unbounded'
             else math.floor(len(old_science)*max_old_fraction))
    cfg.update(method='bounded-native-km1-v1',max_old_fraction=max_old_fraction,
               movement_strength=movement_strength,temporal_control=mode,
               refinement_scope='ranked_changed_vertices; fixed outside paths')
    if view == previous.view:
        state = deepcopy(previous); state.config = cfg; state.events = []
        state.trace = {'noop':True,'initial_builder_calls':0,'allowed_old_nodes':[],
                       'editable_ids':[],'moved_ids':[],'old_node_moves':[],
                       'resumed_from_saved_state':hasattr(previous,'_loaded_from')}
        return state
    affinity,_ = _affinities(science,ids,x,previous,k=cfg['knn'])
    native,normal = scientific_edges(view,science,cfg['gamma'],cfg['normalizations']['native_papers'])
    old_native,_ = scientific_edges(previous.view,old_science,cfg['gamma'],cfg['normalizations']['native_papers'])
    weights = {e['id']:e['weight'] for e in old_native}
    reweighted = {e['id'] for e in native if e['id'] in weights and e['weight'] != weights[e['id']]}
    fresh = {e['id'] for e in change['hyperedges_added']} | reweighted
    frontier,deferred,scores = select_frontier(old_science,new,affinity,native,fresh,limit)
    declared = set(frontier) | set(new)
    edges = _semantic_edges(science,affinity) + native
    assignment = {n:previous.assignment[n] for n in old_science}
    parents = dict(previous.parents)
    positions = {n:i for i,n in enumerate(ids)}
    old_vectors = previous.vectors[[i for i,n in enumerate(previous.view['nodes']) if n['id'] in old_set]]
    old_aligned = [n['id'] for n in previous.view['nodes'] if n['id'] in old_set]
    for node in new:
        nearest = int(np.argmax(old_vectors @ x[positions[node]]))
        assignment[node] = previous.assignment[old_aligned[nearest]]
    novelty = None; protected = set(); metadata_paths = {
        n:previous.assignment[n] for n in previous.assignment.keys()-old_set}
    if novelty_calibration is not None:
        from .novelty_edits import propose_topic_edits
        novelty = propose_topic_edits(previous,view,x,max_old_edits=limit,
            calibration=novelty_calibration,starting_assignment=assignment,starting_parents=parents)
        if allow_old_topic_split:
            from .topic_split import propose_old_topic_split
            split = propose_old_topic_split(previous,view,x,
                starting_assignment=novelty['assignment'],starting_parents=novelty['parents'],
                metadata_assignment=novelty['metadata_assignment'],max_old_edits=limit,
                calibration=novelty_calibration)
            novelty['trace']['old_topic_split'] = split['trace']
            if split['accepted']:
                original_criterion = novelty['trace']['criterion_before']
                original_events = novelty['events']
                novelty.update({key:split[key] for key in (
                    'assignment','parents','metadata_assignment','moved_old_ids','moved_old_scientific_ids')})
                novelty['accepted'] = True
                novelty['events'] = original_events + split['events']
                novelty['trace'].update(status='accepted',criterion_before=original_criterion,
                    criterion_after=split['trace']['criterion_after'])
        if novelty['accepted']:
            assignment = novelty['assignment']; parents = novelty['parents']
            metadata_paths = novelty['metadata_assignment']
            mandatory = set(novelty['moved_old_ids'])
            # Donor paths consume the SAME budget before local refinement.
            ranked = frontier + deferred
            extra = [n for n in ranked if n not in mandatory][:limit-len(mandatory)]
            frontier = sorted(mandatory | set(extra))
            deferred = [n for n in ranked if n not in frontier]
            declared = set(frontier) | set(new)
            protected = mandatory | {n for event in novelty['events'] if event['kind'] in {'birth','split'}
                                      for n in event.get('node_ids',[])}
            novelty['trace']['moved_old_ids'] = sorted(mandatory)
    penalty = (0. if mode!='bounded' else movement_strength / len(old_science))
    calls=[]
    for parent in sorted(set(parents.values())):
        domain = [n for n in science if parents[assignment[n]] == parent]
        active = (set(domain) & declared) - protected
        if active:
            assignment,trace = _solve_region(edges,domain,active,assignment,
                {n:previous.assignment[n] for n in old_science},penalty,cfg['seed'])
            trace['parent'] = parent; calls.append(trace)
    # Coarse changes charge complete old paths, not just a super-node identity.
    q = quotient(edges,assignment)
    frozen_metadata_topics = set(metadata_paths.values())
    protected_topics = {assignment[n] for n in protected if n in assignment}
    eligible = {g for g in parents if g not in frozen_metadata_topics and
                g not in protected_topics and
                all(n in declared for n in old_science if assignment[n] == g)}
    broad = sorted(set(parents.values())); codes = {g:i for i,g in enumerate(broad)}
    coarse_start = {g:codes[p] for g,p in parents.items()}
    coarse_edges = deepcopy(q)
    coarse_fixed = {g:coarse_start[g] for g in parents if g not in eligible}
    coarse_masses = dict(Counter(assignment.values()))
    if penalty:
        previous_coarse = {n:previous.parents[previous.assignment[n]] for n in old_science}
        for g in sorted(eligible):
            by_parent = Counter(previous_coarse[n] for n in old_science if assignment[n] == g)
            for parent,count in by_parent.items():
                anchor = ('__coarse_anchor__',parent)
                coarse_fixed[anchor] = codes[parent]; coarse_masses[anchor] = 1
                coarse_edges.append({'members':[g,anchor],'weight':penalty*count})
    coarse_ids = list(parents) + sorted(set(coarse_fixed)-parents.keys(),key=str)
    coarse_initial = coarse_start | coarse_fixed
    loads = [sum(coarse_masses[n] for n,p in coarse_initial.items() if p == i) for i in range(len(broad))]
    usual = math.ceil(1.5*math.ceil(sum(coarse_masses.values())/len(broad)))
    result,coarse_trace = solve_partition(coarse_ids,coarse_edges,len(broad),initial=coarse_initial,
        fixed=coarse_fixed,masses=coarse_masses,block_caps=[max(n,usual) for n in loads],seed=cfg['seed'])
    proposed_parents = {g:broad[result[g]] for g in parents}
    if set(proposed_parents.values()) != set(broad):
        proposed_parents = parents
        coarse_trace.update(accepted=False,status='rejected_empty_scientific_coarse_group')
    parents = proposed_parents
    metadata_previous = SimpleNamespace(assignment={**previous.assignment,**metadata_paths},
                                        trace=previous.trace,view=previous.view)
    full,metadata = _metadata(view,x,assignment,metadata_previous,set())
    # Old metadata is kept on its saved path, with no unbudgeted fallback moves.
    for n in previous.assignment.keys()-old_set:
        full[n] = metadata_paths[n]
        if parents[full[n]] != previous.parents[full[n]]:
            raise AssertionError('Frozen metadata parent changed')
    moved = [n for n in previous.assignment if full[n] != previous.assignment[n] or
             parents[full[n]] != previous.parents[previous.assignment[n]]]
    if not set(moved) <= set(frontier) or len(moved) > limit:
        raise AssertionError('Old path edit budget violated')
    invariant = all(full[n] == previous.assignment[n] and
        parents[full[n]] == previous.parents[previous.assignment[n]] for n in previous.assignment.keys()-set(frontier))
    if not invariant:
        raise AssertionError('Outside scientific path changed')
    events = _events(previous,view,full,parents)
    if novelty is not None and novelty['accepted']:
        from .novelty_edits import topic_edit_criterion
        final_criterion = topic_edit_criterion(previous,view,x,assignment,parents,
            temporal_weight=novelty_calibration['temporal_weight'],metadata_assignment=metadata_paths)
        if final_criterion['total'] > novelty['trace']['criterion_before']['total'] + 1e-12:
            # The staged native refinement may not undo the transaction's
            # semantic benefit. Roll back the WHOLE proposal, not half a merge.
            fallback = update_local(previous,view,x,max_old_fraction=max_old_fraction,
                movement_strength=movement_strength,mode=mode)
            fallback.trace['novelty'] = {'accepted':False,'reason':'final_complete_criterion_increased',
                                         'proposal':novelty['trace'],'final_criterion':final_criterion}
            return fallback
        born = {e['group_id'] for e in novelty['events'] if e['kind'] in {'birth','split'}}
        events = [e for e in events if not (e['kind']=='birth' and e['level']==1 and e['group_id'] in born)]
        for event in novelty['events']:
            event = deepcopy(event)
            event.setdefault('node_ids',[n for n,g in full.items() if g == event['group_id']])
            events.append(event)
        novelty['trace']['final_criterion'] = final_criterion
    state = DivisiveState(deepcopy(view),x.copy(),full,parents,cfg,events,{},previous.next_id,affinity)
    state.trace = {'initial_builder_calls':0,'resumed_from_saved_state':hasattr(previous,'_loaded_from'),
        'previous_checkpoint':getattr(previous,'_loaded_from',None),'new_nodes':[n['id'] for n in change['nodes_added']],
        'new_edges':sorted(fresh),'reweighted_native_edges':sorted(reweighted),
        'allowed_old_nodes':sorted(frontier),'editable_ids':sorted(declared),'moved_ids':sorted(moved),
        'old_node_moves':sorted(n for n in old_science if full[n] != previous.assignment[n]),
        'affected_old_nodes_count':len(scores),'deferred_affected_old_ids':deferred,
        'frontier_scores':scores,'old_edit_limit':limit,'realized_old_path_edits':len(moved),
        'movement_penalty_per_old_node':penalty,'region_declared_before_optimization':True,
        'old_old_affinities_unchanged':True,
        'old_metadata_frozen':all(metadata_paths[n] == previous.assignment[n] for n in metadata_paths),
        'solver_calls':calls,'coarse_refinement':coarse_trace,
        'boundary_checks':{'outside_region_unchanged':invariant},
        'metadata_attachment':metadata,'native_normalization':normal,
        'global_input_scan':{'nodes':len(ids),'hyperedges':len(view['hyperedges'])},
        'novelty':({'accepted':novelty['accepted'],**novelty['trace']} if novelty is not None else
                   {'accepted':False,'reason':'not_enabled'}),
        'region_declaration_scope':'Native refinement frontier includes accepted transaction donor paths before its solve.',
        'limitations':['Budgeted optimization is not a sublinear input scan.',
                      'Novelty alarms and edit acceptance are heuristics, not verified discovery of scientific fields.']}
    _trace_finish(state,science,edges,native,q)
    return state
