"""Coarse-first retrieval must not secretly score discarded leaf documents."""
from copy import deepcopy
import importlib.util

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('tkh.tree_search') is not None, 'coarse-first search is missing'
    from tkh.tree_search import TreeSearch, SelectedBM25
    return TreeSearch, SelectedBM25


def fixture():
    from tkh.tag_graph_index import TagGraphIndex
    graph={'nodes':[{'id':n,'type':'method','surface_form':text,
                    'provenance':{'articles':[source]}}
                   for n,text,source in [('a','quantum forces',1),('b','potential energy',2),
                                          ('c','crystal lattice',2),('d','lattice crystal',3)]],
           'hyperedges':[{'id':'native','members':['a','b'],'relation_type':'uses'}]}
    h={'levels':[{'index':0,'groups':[{'id':'C0','members':['a','b'],'parent_id':None},
                                   {'id':'C1','members':['c','d'],'parent_id':None}]},
                 {'index':1,'groups':[{'id':'F0','members':['a'],'parent_id':'C0'},
                                     {'id':'F1','members':['b'],'parent_id':'C0'},
                                     {'id':'F2','members':['c'],'parent_id':'C1'},
                                     {'id':'F3','members':['d'],'parent_id':'C1'}]}]}
    x=np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    return graph,TagGraphIndex(graph),h,x


def test_selected_bm25_exactly_retains_global_weights_duplicate_tokens_and_oov():
    _,SelectedBM25=api();_,index,_,_=fixture()
    model=index._bm25[True]; selected=SelectedBM25(model)
    for query in ['quantum quantum forces','lattice','unfindabletoken','the and']:
        tokens=index._tokens(query)
        expected=index.bm25_scores(query)
        actual,trace=selected.score(tokens,[0,2])
        np.testing.assert_array_equal(actual,expected[[0,2]])
        assert trace['documents_scored']==2
        assert trace['global_posting_scores_evaluated'] is False


def test_route_precedes_leaf_dots_and_never_invokes_full_leaf_scorer(monkeypatch):
    TreeSearch,_=api();_,index,h,x=fixture();tree=TreeSearch(index,x,h)
    monkeypatch.setattr(index,'bm25_scores',lambda *a,**k:pytest.fail('global leaf scoring'))
    result=tree.predict('quantum',np.array([1.,0.]),broad_width=1,fine_width=1)
    t=result['trace']
    assert t['coarse_scores']==2 and t['fine_scores']==2
    assert t['leaf_scores']==1 and t['leaf_dense_dot_products']==1
    assert t['full_leaf_scores_before_routing']==0
    assert result['ids']==['a'] and t['selected_fine_ids']==['F0']
    assert result['probabilities'][2]==0


def test_all_width_matches_strong_flat_baseline_probabilities_and_top100():
    TreeSearch,_=api();_,index,h,x=fixture();tree=TreeSearch(index,x,h)
    from tkh.repair_assessment import prepare_channel_cache
    from tkh.hierarchy_retrieval import PairedHierarchySearch
    q=np.array([1.,0.]);questions=[{'question_id':'q','question':'quantum','type':'A'}]
    channels=prepare_channel_cache(index,questions,x,np.array([q]))['q']
    expected=PairedHierarchySearch(index.ids,index.nodes,index.layers,h).predict(channels)['hybrid_no_tags::baseline']
    actual=tree.predict('quantum',q,broad_width=None,fine_width=None)
    assert actual['ids']==expected['ids']
    np.testing.assert_allclose(actual['probabilities'],expected['probabilities'],rtol=1e-12,atol=1e-14)


def test_induced_source_walk_crosses_sources_without_modifying_graph():
    TreeSearch,_=api();graph,index,h,x=fixture();before=deepcopy(graph)
    # A activates B through the native edge; B activates C through source 2.
    actual=TreeSearch(index,x,h).predict('quantum',np.array([1.,0.]),broad_width=None,fine_width=None)
    assert actual['probabilities'][2]>0 and 'c' in actual['ids']
    assert graph==before


def test_lexical_only_oov_returns_no_padding_and_invalid_widths_are_rejected():
    TreeSearch,_=api();_,index,h,x=fixture();tree=TreeSearch(index,x,h)
    result=tree.predict('unfindabletoken',None,broad_width=1,fine_width=1)
    assert result['ids']==[] and not result['probabilities'].any()
    for b,f in [(0,1),(1,0),(-1,2),(True,1)]:
        with pytest.raises(ValueError):tree.predict('quantum',None,broad_width=b,fine_width=f)
