"""Recursive native bipartitions with explicit nontrivial topic-size guards.

The group-count budgets are the assessment constraint; these additional size
guards are an experimental anti-degeneracy choice, not a scientific ontology.
The same frozen own-text affinities and native connectivity cost are retained.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np

from .divisive_hierarchy import (DivisiveState,_input,_affinities,_semantic_edges,
    scientific_edges,_restrict,_metadata,_events,_trace_finish)
from .divisive_solver import solve_partition,quotient,objective


def build_balanced(view,vectors,budgets=(12,50),gamma=.03,seed=20261019):
    ids,x,science=_input(view,vectors)
    if len(budgets)!=2 or any(type(b)is not int for b in budgets) or not 1<=budgets[0]<=budgets[1]:
        raise ValueError('Expected ordered positive integer budgets')
    top,fine=(min(b,len(science)) for b in budgets)
    affinity,norm=_affinities(science,ids,x)
    native,normal=scientific_edges(view,science,gamma)
    edges=_semantic_edges(science,affinity)+native
    positions={n:i for i,n in enumerate(ids)}
    calls=[];lineage=[]
    def split(groups,target,level):
        while len(groups)<target:
            candidates=[i for i,g in enumerate(groups) if len(g)>1]
            # Split the largest remaining region; dispersion breaks equal-size
            # ties. This prevents repeatedly spending slots on small outliers.
            def priority(i):
                xx=x[[positions[n] for n in groups[i]]]
                dispersion=len(xx)-float(np.square(xx.sum(axis=0)).sum())/len(xx)
                return (-len(groups[i]),-dispersion,min(groups[i]))
            selected=min(candidates,key=priority);domain=groups.pop(selected)
            cap=math.ceil(.55*len(domain))
            result,trace=solve_partition(domain,_restrict(edges,domain),2,seed=seed+len(calls),
                                         epsilon=.1,block_caps=[cap,cap])
            halves=[sorted(n for n in domain if result[n]==g) for g in range(2)]
            halves.sort(key=lambda ns:ns[0])
            if min(map(len,halves))<len(domain)-cap:
                raise AssertionError('Native bisection violated its lower size consequence')
            trace.update(stage='balanced_bisection',level=level,parent_members=len(domain),
                         minimum_child_size=len(domain)-cap)
            calls.append(trace);lineage.append({'kind':'split','level':level,
                'original_members':domain,'child_members':halves})
            groups.extend(halves)
        return sorted(groups,key=lambda ns:ns[0])
    broad_groups=split([sorted(science)],top,0)
    # Fine cuts are globally allocated to the largest remaining region, but
    # every child inherits its own coarse parent from the initial cut.
    fine_groups=split(deepcopy(broad_groups),fine,1)
    broad_assignment={n:f'L0-bal{i:06d}' for i,ns in enumerate(broad_groups) for n in ns}
    assignment={n:f'L1-bal{i:06d}' for i,ns in enumerate(fine_groups) for n in ns}
    parents={assignment[ns[0]]:broad_assignment[ns[0]] for ns in fine_groups}
    q=quotient(edges,assignment);groups=sorted(parents);coarse_ids=sorted(set(parents.values()))
    codes={g:i for i,g in enumerate(coarse_ids)};initial={g:codes[p] for g,p in parents.items()}
    masses=dict(Counter(assignment.values()));caps=[math.ceil(1.5*len(science)/top)]*top
    result,refine=solve_partition(groups,q,top,initial=initial,masses=masses,block_caps=caps,seed=seed)
    proposed={g:coarse_ids[result[g]] for g in groups}
    minimum=max(1,math.floor(.5*len(science)/top))
    loads=Counter(proposed[assignment[n]] for n in science)
    guard=len(loads)==top and min(loads.values())>=minimum
    if not guard:
        proposed=parents
        refine.update(accepted=False,status='rejected_nontrivial_size_guard')
    parents=proposed
    original=objective(edges,{n:parents[assignment[n]] for n in science})
    if not math.isclose(original,objective(q,parents),abs_tol=1e-10):
        raise AssertionError('Counted quotient changed the original objective')
    refine.update(quotient_consumed=True,outside_minimum_guard_passed=True,
                  proposal_minimum_guard_passed=guard,minimum_scientific_topic_size=minimum,
                  final_objective=original)
    full,metadata=_metadata(view,x,assignment)
    cfg={'method':'balanced-recursive-native-v1','budgets':list(budgets),'gamma':gamma,'seed':seed,
         'scientific_ids':science,'knn':15,'epsilon':.1,'normalizations':{'semantic':norm,
         'native_papers':normal['native_papers']},'feature_policy':'own unit sentence vectors; model recorded in run protocol',
         'size_policy':'55% maximum per bisection; coarse minimum half average after refinement',
         'label_status':'scientific explanations require separate measured support'}
    trace={'initial_builder_calls':1,'effective_budgets':[top,fine],
           'budget_shortfall':top<budgets[0] or fine<budgets[1],
           'new_nodes':ids,'new_edges':[e['id'] for e in view['hyperedges']],
           'editable_ids':science,'moved_ids':[],'old_node_moves':[],
           'solver_calls':calls,'coarse_refinement':refine,'lineage':lineage,
           'metadata_attachment':metadata,'native_normalization':normal,
           'limitations':['Size guards prevent trivial topics but can cut a scientifically large field.',
                          'Recursive restricted cuts are heuristic, not a joint optimum.']}
    state=DivisiveState(deepcopy(view),x.copy(),full,parents,cfg,
        _events(None,view,full,parents),trace,fine,affinity)
    _trace_finish(state,science,edges,native,q)
    return state
