from copy import deepcopy
import json
import numpy as np
import pytest
from scipy import sparse
from tkh.divisive_hierarchy import build_divisive, update_divisive, DivisiveState, scientific_edges
from tkh.divisive_solver import objective


def fixture(n=8):
    nodes = [{'id':f'n{i}', 'type':'concept','surface_form':f'concept {i}', 'observed_at':2022,
              'provenance':{'articles':[i//4]}} for i in range(n)]
    nodes += [{'id':'article','type':'article','surface_form':'paper', 'observed_at':2022,'provenance':{'articles':[0]}},
              {'id':'author','type':'author','surface_form':'writer','observed_at':2022,'provenance':{'articles':[0]}},
              {'id':'orphan','type':'author','surface_form':'unknown','observed_at':2022,'provenance':{'articles':[99]}}]
    edges = [{'id':'e0','relation_type':'claims','members':['n0','n1','n2'],'year':2022,'provenance':{'article_id':0}},
             {'id':'e1','relation_type':'claims','members':['n4','n5','n6'],'year':2022,'provenance':{'article_id':1}},
             {'id':'present','relation_type':'presents','members':['article','n0','n1'],'year':2022,'provenance':{'article_id':0}}]
    x = np.zeros((len(nodes),3))
    for i in range(n): x[i, i//4] = 1
    x[n:,-1] = 1
    return {'cutoff':2022,'policy':'test','limitations':[], 'nodes':nodes,'hyperedges':edges}, x


def test_build_topdown_metadata_exact_quotient_and_persistence(tmp_path):
    view, x = fixture()
    state = build_divisive(view,x,budgets=(2,4))
    assert set(state.assignment) == {n['id'] for n in view['nodes']}
    assert len(state.parents) == 4 and len(set(state.parents.values())) == 2
    assert state.assignment['article'] == state.assignment['author']
    assert state.trace['metadata_attachment']['no_scientific_anchor'] == ['orphan']
    assert state.config['scientific_ids'] == [f'n{i}' for i in range(8)]
    assert state.trace['numeric_checks']['quotient_objective_equal']
    assert state.trace['coarse_refinement']['quotient_consumed']
    assert state.trace['endpoint_accounting']['full'] == sum(len(e['members']) for e in view['hyperedges'])
    hierarchy, inputs = state.export()
    assert len(hierarchy['levels']) == 3
    state.save(tmp_path/'checkpoint')
    loaded = DivisiveState.load(tmp_path/'checkpoint')
    assert loaded.assignment == state.assignment
    assert (loaded.affinity != state.affinity).nnz == 0
    assert (tmp_path/'checkpoint'/'affinity.npz').exists()
    assert loaded.export() == (hierarchy, inputs)


def test_native_normalization_equal_paper_and_membership_exclusion():
    view, _ = fixture()
    view['hyperedges'].append({'id':'e2','relation_type':'claims','members':['n0','n2'],'provenance':{'article_id':0}})
    edges, info = scientific_edges(view, {f'n{i}' for i in range(8)}, gamma=.3)
    byid = {e['id']:e for e in edges}
    assert byid['e0']['weight'] == byid['e2']['weight'] == pytest.approx(.075)
    assert byid['e1']['weight'] == pytest.approx(.15)
    assert byid['present']['weight'] == 0.
    assert info['native_papers'] == 2


def test_append_and_noop_freeze_old_affinities_boundaries_and_vectors(tmp_path):
    view,x = fixture()
    state=build_divisive(view,x,budgets=(2,4));state.save(tmp_path/'s')
    loaded=DivisiveState.load(tmp_path/'s')
    noop=update_divisive(loaded,deepcopy(view),x.copy())
    assert noop.assignment == loaded.assignment and noop.trace['initial_builder_calls'] == 0
    later=deepcopy(view);later['cutoff']=2023
    later['nodes'].append({'id':'new','type':'concept','surface_form':'new science','observed_at':2023,'provenance':{'articles':[2]}})
    later['hyperedges'].append({'id':'newedge','relation_type':'claims','members':['new','n0'],'year':2023,'provenance':{'article_id':2}})
    current=update_divisive(loaded,later,np.vstack([x,[1.,0.,0.]]))
    assert current.trace['initial_builder_calls'] == 0
    assert (current.affinity[:8,:8] != loaded.affinity).nnz == 0
    editable=set(current.trace['editable_ids'])
    for n,g in state.assignment.items():
        if n.startswith('n') and n not in editable:
            assert current.assignment[n] == g
            assert current.parents[g] == state.parents[g]
    assert set(current.trace['moved_ids']) <= set(current.trace['editable_ids'])
    assert current.trace['boundary_checks']['outside_region_unchanged']
    assert current.config['normalizations'] == state.config['normalizations']
    assert current.trace['unresolved_novelty'] is not None


@pytest.mark.parametrize('mutation', ['remove','edge_change','vector_change','future','node_change'])
def test_reject_nonappend_or_future_input(mutation):
    view,x=fixture();state=build_divisive(view,x,budgets=(2,4));later=deepcopy(view);later['cutoff']=2023
    if mutation=='remove': later['hyperedges'].pop()
    if mutation=='edge_change': later['hyperedges'][0]['members'].append('n3')
    if mutation=='vector_change': x[0]=[0,1,0]
    if mutation=='future': later['nodes'][0]['observed_at']=2024
    if mutation=='node_change': later['nodes'][0]['surface_form']='rewritten'
    with pytest.raises(ValueError): update_divisive(state,later,x)


def test_single_science_degenerate_and_invalid_vectors():
    view,x=fixture();view['nodes']=view['nodes'][:1];view['hyperedges']=[];x=x[:1]
    state=build_divisive(view,x,budgets=(12,50))
    assert len(state.parents)==1
    assert state.trace['effective_budgets']==[1,1]
    assert state.trace['budget_shortfall']
    with pytest.raises(ValueError):build_divisive(view,np.zeros_like(x))


def test_metadata_dependency_closure_is_declared_and_checkpoint_trace_is_truthful(tmp_path):
    view,x=fixture();prior=build_divisive(view,x,budgets=(2,4))
    later=deepcopy(view);later['cutoff']=2023
    later['nodes'].append({'id':'z','type':'concept','surface_form':'extra','observed_at':2023,'provenance':{'articles':[0]}})
    later['hyperedges'].append({'id':'p2','relation_type':'presents','members':['article','z'],'year':2023,'provenance':{'article_id':0}})
    result=update_divisive(prior,later,np.vstack([x,[1.,0.,0.]]))
    assert 'article' in result.trace['dependent_metadata_ids']
    assert 'author' in result.trace['dependent_metadata_ids']
    assert result.trace['region_declared_before_optimization']
    assert result.trace['resumed_from_saved_state'] is False
    prior.save(tmp_path/'s');loaded=DivisiveState.load(tmp_path/'s')
    result=update_divisive(loaded,later,np.vstack([x,[1.,0.,0.]]))
    assert result.trace['resumed_from_saved_state'] is True


def test_checkpoint_corrupt_affinity_rejected(tmp_path):
    view,x=fixture();state=build_divisive(view,x,budgets=(2,4));state.save(tmp_path/'s')
    a=sparse.load_npz(tmp_path/'s'/'affinity.npz');a.data[0]+=.1;sparse.save_npz(tmp_path/'s'/'affinity.npz',a)
    with pytest.raises(ValueError,match='affinity'):DivisiveState.load(tmp_path/'s')
