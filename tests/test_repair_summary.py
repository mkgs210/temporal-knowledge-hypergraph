"""Summary tables must use exact matched arms and task-ID joins."""
import importlib.util
from pathlib import Path

import pytest


def api():
    path=Path(__file__).resolve().parents[1]/'scripts/summarize_repair.py'
    assert path.exists(),'auditable summary generator is missing'
    spec=importlib.util.spec_from_file_location('repair_summary',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_blind_key_join_preserves_missing_evidence_separate_from_wrong():
    m=api()
    ratings={'tasks':[{'task_id':'b','rating':'insufficient_evidence'},
                      {'task_id':'a','rating':'accurate'},
                      {'task_id':'c','rating':'wrong'}]}
    key=[{'task_id':'c','condition':'cross_group'},{'task_id':'a','condition':'own'},
         {'task_id':'b','condition':'own'}]
    result=m.join_blind(ratings,key)
    assert result['own']=={'accurate':1,'vague':0,'wrong':0,'insufficient_evidence':1,'total':2}
    assert result['cross_group']['wrong']==1


@pytest.mark.parametrize('mutation',['duplicate_rating','duplicate_key','missing_rating','unknown_category'])
def test_blind_summary_rejects_incomplete_or_ambiguous_joins(mutation):
    m=api();r={'tasks':[{'task_id':'a','rating':'accurate'}]};k=[{'task_id':'a','condition':'own'}]
    if mutation=='duplicate_rating':r['tasks']*=2
    elif mutation=='duplicate_key':k*=2
    elif mutation=='missing_rating':r['tasks']=[]
    else:r['tasks'][0]['rating']='supported'
    with pytest.raises(ValueError):m.join_blind(r,k)


def test_exact_retrieval_arm_uses_all63_targets_and_checks_consistent_micro_recall():
    m=api();report={'tracks':{'A':{'questions':14,'metrics':{'30':{
        'hits':28,'targets_all':63,'recall_all_micro':28/63,'targets_reachable':48}}}}}
    assert m.retrieval_cell({'exact':report},'exact')=='28/63 (44.4%)'
    report['tracks']['A']['metrics']['30']['targets_all']=48
    with pytest.raises(ValueError):m.retrieval_cell({'exact':report},'exact')
    with pytest.raises(KeyError):m.retrieval_cell({'other':report},'exact')


def test_temporal_summary_does_not_turn_conditional_significance_into_a_ci():
    m=api();row={'level':0,'point_ari':1.,'alignment_significance':{'p_greater_equal':.0005},
                 'interval':None,'uncertainty_status':'insufficient_source_components',
                 'blocks':{'count':2},'source_delete_one':{'range':[.99,1.]}}
    cell=m.temporal_cell(row)
    assert '1.0000' in cell and 'p=0.0005' in cell and 'CI unavailable' in cell
    assert '0.9900' in cell


def test_old_path_count_comes_from_memberships_including_coarse_and_metadata_changes():
    m=api()
    old={'levels':[{'index':1,'groups':[{'id':'a','parent_id':'P','members':['article','x']},
                                     {'id':'b','parent_id':'P','members':['y']}]}]}
    new={'levels':[{'index':1,'groups':[{'id':'a','parent_id':'Q','members':['x']},
                                     {'id':'b','parent_id':'P','members':['article','y']}]}]}
    assert m.path_change_count(old,new)==2
