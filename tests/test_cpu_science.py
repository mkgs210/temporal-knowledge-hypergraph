"""Independent held-out citation metric and matched retrieval attribution."""
import pytest


def test_citation_coherence_has_exact_pair_score_and_size_preserving_null():
    from tkh.cpu_science import citation_coherence
    view={'nodes':[{'id':n,'type':'method'} for n in 'abcd'], 'hyperedges':[
        {'id':'c','relation_type':'cites','members':['a','b'], 'provenance':{'article_id':1}},
        {'id':'ignored','relation_type':'uses_component','members':['a','c']}]}
    groups=[{'id':'x','members':['a','b']},{'id':'y','members':['c','d']}]
    r=citation_coherence(view,groups,n_permutations=99,seed=4)
    assert r['observed']==1 and r['citation_edges']==1
    assert r['null_mean']<1 and len(r['null_values'])==99
    assert 0<r['permutation_pvalue']<=1
    assert r['construction_signal_excluded']=='cites'
    assert citation_coherence({**view,'hyperedges':[]},groups)['observed'] is None
    with pytest.raises(ValueError,match='partition'):
        citation_coherence(view,groups[:1])


def test_paired_comparison_counts_target_swaps_not_only_equal_totals():
    from tkh.cpu_science import retrieval_comparison
    def report(ranks):
        return {'questions':[{'question_id':'q','type':'A','targets':[
            {'label':key,'first_rank':rank} for key,rank in ranks.items()]}]}
    result=retrieval_comparison(report({'A':1,'B':31}),report({'A':31,'B':1}),n_bootstrap=20)
    assert result['baseline_hits']==result['hierarchy_hits']==1
    assert result['gained_targets']==[{'question_id':'q','target':'B'}]
    assert result['lost_targets']==[{'question_id':'q','target':'A'}]
    assert result['delta_recall_at_30']==0


def test_unused_citation_links_do_not_change_the_real_initial_partition(tmp_path):
    from copy import deepcopy
    from test_submission_pipeline import _fixture
    from tkh.submission_pipeline import _load_assets,_view_vectors
    from tkh.temporal_input import temporal_view
    from tkh.balanced_hierarchy import build_balanced
    import json
    data,assets,_=_fixture(tmp_path)
    view=temporal_view(json.loads((data/'tkh_collection10.json').read_text()),2022)
    cache,_,_=_load_assets(assets);x=_view_vectors(view,cache)
    changed=deepcopy(view)
    changed['hyperedges'].append({'id':'held-out','relation_type':'cites','members':['n0','n2'],
        'year':2022,'provenance':{'article_id':1,'article_year':2022}})
    a=build_balanced(view,x,budgets=(2,4));b=build_balanced(changed,x,budgets=(2,4))
    assert a.assignment==b.assignment and a.parents==b.parents
