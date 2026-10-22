"""Cache replay binds explanatory text to exact fresh cutoff-safe prompts."""
import json
from copy import deepcopy
import pytest
from tkh.scientific_labels import build_label_input
from test_scientific_labels import graph


def fixture():
    view=graph(); members=[n['id'] for n in view['nodes']]
    h={'levels':[{'index':0,'groups':[{'id':'g','members':members,'label':'old','gloss':'old'}]}]}
    inp=build_label_input(view,members,2022,selection_policy='science_evidence')
    label={'label':'Graph model evaluation','gloss':'Graph models connect these methodological statements and their evaluations.','status':'mixed'}
    records=[{'id':'g','level':0,'member_count':len(members),'prompt_sha256':inp['prompt_sha256'],
              'raw_output':json.dumps(label),**label}]
    return view,h,records


def test_label_cache_uses_exact_prompt_and_never_changes_membership_or_input():
    from tkh.label_cache import annotate_labels,validate_annotations
    view,h,records=fixture(); original=deepcopy(h)
    annotated,artifact=annotate_labels(view,h,records,{'generator':'fixture','human':False})
    assert h==original
    assert annotated['levels'][0]['groups'][0]['members']==h['levels'][0]['groups'][0]['members']
    assert annotated['levels'][0]['groups'][0]['label']=='Graph model evaluation'
    assert annotated['levels'][0]['groups'][0]['label_status']=='mixed'
    validate_annotations(view,annotated,artifact)


@pytest.mark.parametrize('what',['hash','id','missing','duplicate','text','count'])
def test_label_cache_rejects_stale_missing_or_inconsistent_outputs(what):
    from tkh.label_cache import annotate_labels
    view,h,r=fixture()
    if what=='hash':r[0]['prompt_sha256']='0'*64
    elif what=='id':r[0]['id']='wrong'
    elif what=='missing':r=[]
    elif what=='duplicate':r+=deepcopy(r)
    elif what=='text':r[0]['gloss']='This changed gloss does not match the recorded generation.'
    else:r[0]['member_count']+=1
    with pytest.raises(ValueError):annotate_labels(view,h,r,{})


def test_annotation_checker_recomputes_prompt_instead_of_trusting_saved_hash():
    from tkh.label_cache import annotate_labels,validate_annotations
    view,h,r=fixture(); h,a=annotate_labels(view,h,r,{})
    a['groups'][0]['input']['prompt']='tampered'
    with pytest.raises(ValueError):validate_annotations(view,h,a)
    h,a=annotate_labels(view,fixture()[1],r,{})
    h['levels'][0]['groups'][0]['gloss']='Changed scientific claim.'
    with pytest.raises(ValueError):validate_annotations(view,h,a)
