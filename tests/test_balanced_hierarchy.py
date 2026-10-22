"""Balanced recursive construction must not spend topic slots on singleton outliers."""
from copy import deepcopy
import importlib

import numpy as np


def instance():
    rng=np.random.default_rng(10)
    x=rng.normal(0,.03,(96,6))
    for i in range(96):x[i,i//16]+=1
    x/=np.linalg.norm(x,axis=1,keepdims=True)
    view={'cutoff':2022,'policy':'test','nodes':[{'id':f'n{i:03d}','type':'method',
        'surface_form':f'concept {i}','observed_at':2022,'provenance':{'articles':[i//16]}}
        for i in range(96)],'hyperedges':[{'id':f'e{i}','relation_type':'claims',
        'members':[f'n{j:03d}' for j in range(i,i+16)],'year':2022,'provenance':{'article_id':i//16}}
        for i in range(0,96,16)]}
    return view,x


def test_recursive_native_partitions_are_nested_cover_all_and_avoid_singleton_topic_slots():
    module=importlib.import_module('tkh.balanced_hierarchy')
    view,x=instance();before=deepcopy(view)
    state=module.build_balanced(view,x,budgets=(6,18))
    hierarchy,_=state.export()
    assert view==before
    assert state.config['budgets']==[6,18]
    assert [len(l['groups']) for l in hierarchy['levels']]==[6,18,96]
    for level,minimum in ((0,8),(1,2)):
        assert min(len(g['members']) for g in hierarchy['levels'][level]['groups'])>=minimum
        assert sorted(n for g in hierarchy['levels'][level]['groups'] for n in g['members'])==sorted(state.assignment)
    assert state.trace['numeric_checks']['quotient_objective_equal']
    assert state.trace['coarse_refinement']['quotient_consumed']
    assert state.trace['coarse_refinement']['outside_minimum_guard_passed']


def test_recursive_builder_replays_identically_and_supports_small_graph_shortfall():
    module=importlib.import_module('tkh.balanced_hierarchy')
    view,x=instance();view['nodes']=view['nodes'][:3];view['hyperedges']=[];x=x[:3]
    a=module.build_balanced(view,x,budgets=(12,50));b=module.build_balanced(view,x,budgets=(12,50))
    assert a.assignment==b.assignment and a.parents==b.parents
    assert len(a.parents)==len(set(a.parents.values()))==3
    assert a.trace['budget_shortfall']
