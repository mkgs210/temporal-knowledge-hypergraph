"""Failure attribution must count alias alternatives once and separate routing from rank."""
import hashlib
import json

import pytest


def fixture():
    nodes={n:{'id':n,'type':'method','surface_form':n} for n in 'abcdef'}
    hierarchy={'levels':[
        {'index':0,'groups':[{'id':'C0','members':list('abcd'),'parent_id':None},
                            {'id':'C1','members':list('ef'),'parent_id':None}]},
        {'index':1,'groups':[{'id':'F0','members':list('ab'),'parent_id':'C0'},
                            {'id':'F1','members':list('cd'),'parent_id':'C0'},
                            {'id':'F2','members':list('ef'),'parent_id':'C1'}]}]}
    trace={'selected_coarse_ids':['C0'],'selected_fine_ids':['F0'],'retained_ids':list('ab')}
    targets=[{'label':'A','node_ids':['a']},{'label':'B','node_ids':['b']},
             {'label':'C','node_ids':['c']},{'label':'E','node_ids':['e']},
             {'label':'missing','node_ids':[]},{'label':'alias','node_ids':['a','f']}]
    return nodes,hierarchy,trace,targets


def test_stage_attribution_and_gain_loss_account_for_each_target_once():
    from tkh.retrieval_diagnosis import diagnose_route
    nodes,h,trace,targets=fixture()
    rows,summary=diagnose_route(targets,['c','e','b','a'],['a','b'],h,trace,nodes,k=1)
    assert [r['stage'] for r in rows]==[
        'hit','retained_not_top_k','fine_rejected','coarse_rejected','unmatched','hit']
    assert summary['targets']==6 and summary['matched']==5
    assert summary['hits']==2 and summary['baseline_hits']==1
    assert summary['gained']==2 and summary['lost']==1
    assert summary['stage_counts']=={'hit':2,'retained_not_top_k':1,
        'fine_rejected':1,'coarse_rejected':1,'unmatched':1}
    assert rows[-1]['retained_node_ids']==['a']


def test_equal_hit_counts_can_hide_a_swap_in_target_identity():
    from tkh.retrieval_diagnosis import compare_rankings
    targets=[{'label':'one','node_ids':['a']},{'label':'two','node_ids':['b']}]
    rows,summary=compare_rankings(targets,['a','b'],['b','a'],k=1)
    assert summary=={'targets':2,'before_hits':1,'after_hits':1,'gained':1,'lost':1}
    assert [r['change'] for r in rows]==['lost','gained']


def test_diagnosis_rejects_trace_that_does_not_match_selected_group_members():
    from tkh.retrieval_diagnosis import diagnose_route
    nodes,h,trace,targets=fixture();trace['retained_ids'].append('c')
    with pytest.raises(ValueError,match='retained'):
        diagnose_route(targets,['a'],['a'],h,trace,nodes)


def test_diagnosis_rejects_ranked_record_outside_selected_branch():
    from tkh.retrieval_diagnosis import diagnose_route
    nodes,h,trace,targets=fixture()
    with pytest.raises(ValueError,match='ranking|selected'):
        diagnose_route(targets,['a'],['e'],h,trace,nodes)


def test_saved_ranking_authentication_detects_changed_bytes(tmp_path):
    from tkh.retrieval_diagnosis import verify_predictions
    p=tmp_path/'predictions.json';p.write_text('{"method":{"Q1":["a"]}}')
    lock={'predictions_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (tmp_path/'prediction_lock.json').write_text(json.dumps(lock))
    assert verify_predictions(tmp_path)=={'method':{'Q1':['a']}}
    p.write_text('{"method":{"Q1":["b"]}}')
    with pytest.raises(ValueError,match='hash|changed'):
        verify_predictions(tmp_path)
