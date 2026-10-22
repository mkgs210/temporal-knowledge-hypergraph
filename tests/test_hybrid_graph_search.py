from types import SimpleNamespace
import numpy as np
import pytest
from tkh.hybrid_graph_search import combine_channels, diffuse, shuffle_native_layer, HybridGraphSearch
from tkh.tag_graph_index import IncidenceLayer


def test_channels_normalize_before_combination_and_drop_absent_signal():
    x=combine_channels({'lexical':np.array([10.,0.,0.]),'tags':np.array([0.,2.,0.]),
                        'dense':np.zeros(3)}, {'lexical':.5,'tags':.25,'dense':.25})
    np.testing.assert_allclose(x,[2/3,1/3,0])
    assert not combine_channels({'a':np.zeros(3)},{'a':1}).any()
    with pytest.raises(ValueError):combine_channels({'a':np.array([-1.,1.])},{'a':1})


def test_native_walk_recovers_zero_seed_entity_and_preserves_dangling_mass():
    ids=['evidence','method','isolated']
    layer=IncidenceLayer(ids,[['evidence','method']],name='native')
    seed=np.array([.75,0.,.25])
    result,trace=diffuse(seed,[layer],alpha=.65)
    assert result[1]>0 and result.sum()==pytest.approx(1.)
    assert trace['converged'] and trace['iterations']<=40
    np.testing.assert_allclose(diffuse(seed,[])[0],seed)


def test_shuffle_preserves_native_arity_degree_and_edge_weights():
    ids=[str(i) for i in range(8)]
    layer=IncidenceLayer(ids,[['0','1','2'],['3','4'],['0','5','6'],['2','7']],
                         weights=[1.,2.,3.,4.],name='native')
    shuffled,trace=shuffle_native_layer(layer,seed=2)
    np.testing.assert_array_equal(layer.matrix.sum(axis=0),shuffled.matrix.sum(axis=0))
    np.testing.assert_array_equal(layer.matrix.sum(axis=1),shuffled.matrix.sum(axis=1))
    np.testing.assert_array_equal(layer.weights,shuffled.weights)
    assert trace['successful_swaps']>0
    assert (shuffled.matrix!=layer.matrix).nnz>0


class Index:
    def __init__(self):
        self.ids=['a','b','c','d']
        self.nodes={n:{'id':n,'type':t,'surface_form':n} for n,t in zip(self.ids,['claim','method','author','article'])}
        self.layers={'native':IncidenceLayer(self.ids,[['a','b']],name='native'),
                     'tags':IncidenceLayer(self.ids,[['a','b']],name='tags'),
                     'sources':IncidenceLayer(self.ids,[['a','b','d']],name='sources')}
    def bm25_scores(self,question,context=True):return np.array([1.,0.,0.,0.]) if question=='fracture' else np.zeros(4)
    def tag_scores(self,question):return np.zeros(4)
    def query_tags(self,question):return []


def hierarchy():
    return {'levels':[{'index':0,'groups':[{'id':'top','parent_id':None,'members':['a','b','c','d']}]},
        {'index':1,'groups':[{'id':'fine','parent_id':'top','members':['a','b','c','d']}]}]}


def test_all_arms_share_entity_filter_and_native_graph_can_supply_the_answer():
    search=HybridGraphSearch(Index(),hierarchy=hierarchy())
    out=search.predict('fracture')
    assert out['bm25_own']['ranked']==[]
    assert out['hybrid_graph']['ranked'][0]['node_id']=='b'
    assert all(r['node_id']=='b' for p in out.values() for r in p['ranked'])
    assert all(not p['ranked'] for p in search.predict('unmatched').values())
    source=search.predict('fracture',question_type='B')
    assert all(r['node_id']=='d' for p in source.values() for r in p['ranked'])


def test_hierarchy_membership_and_semantic_vectors_are_authenticated_by_shape():
    broken=hierarchy();broken['levels'][1]['groups'][0]['members'].remove('b')
    with pytest.raises(ValueError):HybridGraphSearch(Index(),hierarchy=broken)
    with pytest.raises(ValueError):HybridGraphSearch(Index(),passage_vectors=np.ones((3,2)))
    vectors=np.tile([1.,0.],(4,1))
    search=HybridGraphSearch(Index(),passage_vectors=vectors,hierarchy=hierarchy())
    with pytest.raises(ValueError):search.predict('fracture',query_vector=np.zeros(2))


def test_iteration_cap_is_reported_instead_of_inventing_convergence():
    layer=IncidenceLayer(['a','b'],[['a','b']],name='native')
    _,trace=diffuse(np.array([1.,0.]),[layer],max_steps=1)
    assert not trace['converged'] and trace['residual_l1']>1e-9


def test_explanations_use_the_actual_arm_seed_and_lexical_field():
    class Fields(Index):
        def bm25_scores(self,question,context=True):
            return np.array([1.,0.,0.,0.]) if context else np.array([0.,1.,0.,0.])
        def tag_scores(self,question):return np.array([0.,1.,0.,0.])
    result=HybridGraphSearch(Fields(),hierarchy=hierarchy()).predict('fracture')
    own=result['bm25_own']['ranked'][0]
    assert own['seed_probability']==1.
    assert own['channel_scores']['bm25']==1.
    assert result['hybrid_no_tags']['ranked'][0]['seed_probability']==0.
