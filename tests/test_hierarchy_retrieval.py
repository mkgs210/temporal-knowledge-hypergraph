import copy
import numpy as np
import pytest
from tkh.tag_graph_index import IncidenceLayer


def fixture():
    ids=['a','b','c','d']
    nodes={n:{'type':'method'} for n in ids}
    layers={name:IncidenceLayer(ids,edges,name=name) for name,edges in {
        'native':[['a','c']], 'sources':[['b','d']], 'tags':[['a','b']]}.items()}
    hierarchy={'levels':[{'index':0,'groups':[
        {'id':'t0','parent_id':None,'members':['a','b']},
        {'id':'t1','parent_id':None,'members':['c','d']}]},
        {'index':1,'groups':[{'id':'f0','parent_id':'t0','members':['a','b']},
                              {'id':'f1','parent_id':'t1','members':['c','d']}]}]}
    channels={'bm25_own':np.array([1.,0.,0.,0.]),'lexical':np.array([1.,0.,0.,0.]),
              'dense':np.zeros(4),'tags':np.zeros(4)}
    return ids,nodes,layers,hierarchy,channels


def test_flat_overlay_keeps_identity_share_instead_of_replacing_search():
    from tkh.hierarchy_retrieval import PairedHierarchySearch
    ids,nodes,layers,hierarchy,channels=fixture()
    search=PairedHierarchySearch(ids,nodes,layers,hierarchy)
    rows=search.predict(channels,'A',routing_widths=[(1,1)])
    base=rows['bm25_context::baseline'];overlay=rows['bm25_context::hierarchy']
    assert base['ids']==['a']
    assert overlay['ids']==['a','b']
    assert overlay['scores'][1]==pytest.approx(.04875/.4475,abs=2e-9)
    assert sum(overlay['probabilities'])==pytest.approx(1.)


def test_routing_happens_before_diffusion_and_cannot_recover_outside_nodes():
    from tkh.hierarchy_retrieval import PairedHierarchySearch
    ids,nodes,layers,hierarchy,channels=fixture()
    before=copy.deepcopy(layers['native'].edges)
    rows=PairedHierarchySearch(ids,nodes,layers,hierarchy).predict(channels,'A',routing_widths=[(1,1)])
    assert 'c' in rows['bm25__native::baseline']['ids']
    routed=rows['bm25__native::route_1_1']
    assert routed['ids']==['a']
    assert routed['probabilities'][2]==0
    assert routed['trace']['routing']['retained_ids']==['a','b']
    assert routed['trace']['routing']['full_graph_diffusion_before_routing'] is False
    assert layers['native'].edges==before
    assert all(r['trace']['all_branch_replay_checked'] for r in rows.values())


def test_empty_induced_layer_returns_its_weight_to_seed_without_reweighting():
    from tkh.hierarchy_retrieval import fixed_mixture_walk
    ids=['a','b']
    connected=IncidenceLayer(ids,[ids],name='native')
    empty=IncidenceLayer(ids,[],name='sources')
    scores,trace=fixed_mixture_walk(np.array([1.,0.]),[connected,empty])
    assert scores[1]==pytest.approx(.1625,abs=2e-9)
    assert trace['empty_induced_layers']==['sources']


def test_invalid_hierarchy_is_rejected_before_search():
    from tkh.hierarchy_retrieval import PairedHierarchySearch
    ids,nodes,layers,hierarchy,channels=fixture()
    hierarchy['levels'][1]['groups'][1]['members']=['b','d']
    with pytest.raises(ValueError):PairedHierarchySearch(ids,nodes,layers,hierarchy)


def test_null_preserves_cluster_sizes_types_and_laminarity():
    from tkh.hierarchy_retrieval import PairedHierarchySearch
    ids,nodes,layers,hierarchy,channels=fixture()
    nodes['b']['type']=nodes['d']['type']='claim'
    search=PairedHierarchySearch(ids,nodes,layers,hierarchy)
    for null in search.null_hierarchies:
        for original,permuted in zip(hierarchy['levels'],null['levels'],strict=True):
            parents={g['id']:set(g['members']) for g in null['levels'][0]['groups']}
            for before,after in zip(original['groups'],permuted['groups'],strict=True):
                assert sorted(nodes[n]['type'] for n in before['members'])==sorted(nodes[n]['type'] for n in after['members'])
                if permuted['index']==1:assert set(after['members'])<=parents[after['parent_id']]
