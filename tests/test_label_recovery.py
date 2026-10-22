"""One-time, offline recovery never silently relaxes live response parsing."""
import hashlib
import json

import pytest

from test_gemini_labels import batch_fixture
from test_nvidia_labels import reply


def saved_legacy_batch(tmp_path):
    from tkh.gemini_labels import run_labels,batch_prompt,write
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'new';source=tmp_path/'old'
    run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',prepare_only=True,decoding='json_schema')
    protocol=json.loads((out/'protocol.json').read_text())
    protocol['schema_version']='api-labels-v3';protocol['generation'].pop('response_format')
    write(source/'protocol.json',protocol)
    jobs=json.loads((out/'prompts.json').read_text());write(source/'prompts.json',jobs)
    batch=[j for j in jobs if j['level']==1];text=batch_prompt(batch)
    sha=hashlib.sha256(text.encode()).hexdigest()
    write(source/'requests'/f'{sha}.json',{'prompt':text,'prompt_sha256':sha,
        'year':2022,'level':1,'group_ids':[j['id'] for j in batch]})
    rows=[{'group_id':j['id'],'label':'Graph model evaluation',
        'gloss':'Graph models connect the supplied methodological statements with their evaluations.',
        'status':'mixed'} for j in batch]
    response_path=source/'responses'/f'{sha}-123.json'
    write(response_path,reply('Here are the groups formatted as requested:\n\n'+json.dumps(rows)))
    return run,prompt,source,out,response_path


def test_legacy_recovery_preserves_originals_and_resume_skips_recovered_groups(tmp_path,monkeypatch):
    from tkh.label_recovery import recover_legacy_labels
    from tkh.gemini_labels import run_labels
    run,prompt,source,out,_=saved_legacy_batch(tmp_path)
    before={p:p.read_bytes() for p in source.rglob('*') if p.is_file()}
    result=recover_legacy_labels(source,out)
    assert result['imported_labels']==2 and result['network_requests']==0
    assert all(p.read_bytes()==data for p,data in before.items())
    recovered=json.loads((out/'snapshots/2022/labels.json').read_text())
    assert {r['id'] for r in recovered}=={'fine-a','fine-b'}
    assert all(r['response_format']=='legacy-prompt-only-imported' for r in recovered)
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');calls=[]
    def send(model,prompt,key):calls.append(prompt);return reply()
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',send=send,rpm=100000,decoding='json_schema')
    assert result['completed']==3 and result['imported_legacy_labels']==2 and len(calls)==1


@pytest.mark.parametrize('corruption',['missing','changed_evidence','different_model'])
def test_legacy_recovery_rejects_mismatched_ids_evidence_or_model_before_import(tmp_path,corruption):
    from tkh.label_recovery import recover_legacy_labels
    run,prompt,source,out,response_path=saved_legacy_batch(tmp_path)
    if corruption=='missing':
        r=json.loads(response_path.read_text());r['choices'][0]['message']['content']='[]';response_path.write_text(json.dumps(r))
    elif corruption=='changed_evidence':
        p=next((source/'requests').glob('*.json'));r=json.loads(p.read_text());r['prompt']+='changed';p.write_text(json.dumps(r))
    else:
        p=source/'protocol.json';r=json.loads(p.read_text());r['model']='nvidia/different';p.write_text(json.dumps(r))
    with pytest.raises(ValueError):recover_legacy_labels(source,out)
    assert not list((out/'completed').glob('*.json'))


def structured_response(tmp_path,*,decoding='json_schema',multi_sentence=False):
    from tkh.gemini_labels import run_labels,batch_prompt,write
    from tkh.nvidia_labels import nvidia_response_format
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',prepare_only=True,decoding=decoding)
    jobs=json.loads((out/'prompts.json').read_text());batch=[j for j in jobs if j['level']==1]
    text=batch_prompt(batch,provider='nvidia');sha=hashlib.sha256(text.encode()).hexdigest()
    schema=nvidia_response_format(['fine-a','fine-b'])
    constraint=({'guided_json':schema['json_schema']['schema']} if decoding=='guided_json' else {'response_format':schema})
    write(out/'requests'/f'{sha}.json',{'prompt':text,'prompt_sha256':sha,'year':2022,'level':1,
        'group_ids':[j['id'] for j in batch],'decoding':decoding,
        'validation_schema':schema['json_schema']['schema'],**constraint})
    gloss=('One sentence: 34 words. The group covers interatomic potentials, ML potentials, '
           'and related components, problems, and tasks in materials modeling.' if multi_sentence else
           'A collection of methods centered on first-principles molecular dynamics and electronic structure simulation.')
    rows=[{'group_id':j['id'],'label':'First principles molecular dynamics',
        'gloss':gloss,
        'status':'coherent'} for j in batch]
    path=out/'responses'/f'{sha}-123.json';write(path,reply(json.dumps({'groups':rows})))
    return run,prompt,out,path


def test_saved_structured_reply_recovers_flagged_labels_without_network_or_text_edits(tmp_path,monkeypatch):
    from tkh.label_recovery import recover_saved_response
    from tkh.gemini_labels import run_labels
    run,prompt,out,path=structured_response(tmp_path);original=path.read_bytes()
    result=recover_saved_response(out,path)
    assert result['imported_labels']==2 and result['network_requests']==0
    assert path.read_bytes()==original
    summary=json.loads((out/'summary.json').read_text())
    assert summary['completed']==2 and summary['labels_with_quality_warnings']==2
    assert (out/'label_review.json').exists()
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');calls=[]
    def send(*args):calls.append(args);return reply()
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',send=send,rpm=100000,decoding='json_schema')
    assert result['completed']==3 and len(calls)==1


def test_saved_reply_recovery_rejects_wrong_group_ids_before_saving_any_label(tmp_path):
    from tkh.label_recovery import recover_saved_response
    run,prompt,out,path=structured_response(tmp_path)
    response=json.loads(path.read_text());rows=json.loads(response['choices'][0]['message']['content'])
    rows['groups'][1]['group_id']='fine-a'
    response['choices'][0]['message']['content']=json.dumps(rows);path.write_text(json.dumps(response))
    with pytest.raises(ValueError):recover_saved_response(out,path)
    assert not list((out/'completed').glob('*.json'))


def test_guided_multi_sentence_recovery_preserves_text_format_and_resumes_without_repeating(tmp_path,monkeypatch):
    from tkh.label_recovery import recover_saved_response
    from tkh.gemini_labels import run_labels
    run,prompt,out,path=structured_response(tmp_path,decoding='guided_json',multi_sentence=True)
    before={p:p.read_bytes() for folder in ('requests','responses') for p in (out/folder).glob('*.json')}
    result=recover_saved_response(out,path)
    assert result['imported_labels']==2 and result['network_requests']==0
    for p,content in before.items():assert p.read_bytes()==content
    rows=json.loads((out/'snapshots/2022/labels.json').read_text())
    assert all(r['response_format']=='guided_json' and r['quality_warnings'] for r in rows)
    assert all(r['gloss'].startswith('One sentence: 34 words. The group') for r in rows)
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');calls=[]
    def send(*args):calls.append(args);return reply()
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',send=send,rpm=100000)
    assert result['completed']==3 and len(calls)==1
