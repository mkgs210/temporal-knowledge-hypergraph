"""Offline checks: source separation, cache reuse and actual Gemini wire format."""
import json
from pathlib import Path
from copy import deepcopy
from urllib.error import HTTPError
from io import BytesIO

import pytest

from test_scientific_labels import graph


def fixture(tmp_path):
    run=tmp_path/'run'; snap=run/'snapshots/2022';snap.mkdir(parents=True)
    view=graph()
    hierarchy={'cutoff':2022,'levels':[
        {'index':0,'groups':[{'id':'broad','parent_id':None,'members':[f'n{i}' for i in range(8)],'label':'OLD LABEL','gloss':'OLD GLOSS'}]},
        {'index':1,'groups':[{'id':'fine','parent_id':'broad','members':[f'n{i}' for i in range(8)],'label':'OLD LABEL','gloss':'OLD GLOSS'}]},
        {'index':2,'groups':[]}]}
    (snap/'input.json').write_text(json.dumps(view));(snap/'hierarchy.json').write_text(json.dumps(hierarchy))
    prompt=tmp_path/'prompt.md';prompt.write_text('Name the scientific theme using supplied evidence only.')
    return run,prompt,hierarchy


def reply(text=None,**kwargs):
    label=text or json.dumps({'label':'Graph model evaluation','gloss':'Graph models connect the supplied methodological statements with their evaluations.','status':'mixed'})
    return {'candidates':[{'content':{'parts':[{'thought':True,'text':'hidden reasoning'}, {'text':label}]},'finishReason':'STOP'}],
            'modelVersion':'test-gemini-version','usageMetadata':{'promptTokenCount':50,'candidatesTokenCount':30},**kwargs}


def test_prompt_withholds_sources_and_never_passes_old_labels_or_metadata(tmp_path):
    from tkh.gemini_labels import prepare_jobs
    run,prompt,_=fixture(tmp_path)
    jobs=prepare_jobs(run,prompt)
    assert len(jobs)==2
    for job in jobs:
        assert 'OLD LABEL' not in job['prompt'] and 'OLD GLOSS' not in job['prompt']
        assert set(job['input']['generation']['source_ids']).isdisjoint(job['input']['support']['source_ids'])
        evidence=json.loads(job['prompt'].split('EVIDENCE:\n')[1])
        assert len(evidence['members'])==4
        visible={n['id'] for n in evidence['members']}
        assert not visible & set(job['input']['support']['member_ids'])


def test_prepare_only_never_reads_a_key_or_calls_the_network(tmp_path):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path)
    def forbidden(*a,**kw):raise AssertionError('Network used in prepare-only')
    result=run_labels(run,tmp_path/'out',prompt_path=prompt,prepare_only=True,
        env_file=tmp_path/'absent',send=forbidden)
    assert result['total']==2 and result['completed']==0
    assert (tmp_path/'out/prompts.json').exists()


def test_resume_skips_completed_requests_and_exports_only_changed_labels(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt,original=fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-private-key')
    calls=[]
    def send(model,prompt,key):calls.append((model,prompt,key));return reply()
    first=run_labels(run,out,prompt_path=prompt,limit=1,rpm=100000,send=send)
    assert first['completed']==1 and len(calls)==1
    second=run_labels(run,out,prompt_path=prompt,rpm=100000,send=send)
    assert second['completed']==2 and len(calls)==2
    assert json.loads((run/'snapshots/2022/hierarchy.json').read_text())==original
    exported=json.loads((out/'snapshots/2022/hierarchy.json').read_text())
    assert exported['levels'][0]['groups'][0]['members']==original['levels'][0]['groups'][0]['members']
    assert exported['levels'][0]['groups'][0]['label']=='Graph model evaluation'
    monkeypatch.delenv('GEMINI_API_KEY')
    run_labels(run,out,prompt_path=prompt,env_file=tmp_path/'absent',send=send)
    assert len(calls)==2
    for file in out.rglob('*.json'):assert 'fake-private-key' not in file.read_text()


def test_changed_prompts_or_model_cannot_silently_reuse_old_outputs(tmp_path):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path);out=tmp_path/'out'
    run_labels(run,out,prompt_path=prompt,prepare_only=True)
    with pytest.raises(ValueError,match='different|changed'):
        run_labels(run,out,prompt_path=prompt,model='different-model',prepare_only=True)
    prompt.write_text('Changed instructions')
    with pytest.raises(ValueError,match='different|changed'):
        run_labels(run,out,prompt_path=prompt,prepare_only=True)


def test_truncated_or_blocked_generation_is_not_counted_as_a_name(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-key')
    broken=reply();broken['candidates'][0]['finishReason']='MAX_TOKENS'
    result=run_labels(run,out,prompt_path=prompt,send=lambda *a:broken,rpm=100000,max_attempts=1)
    assert result['completed']==0 and result['rejected_this_invocation']==2
    rejected=[json.loads(p.read_text()) for p in (out/'rejected').glob('*.json')]
    assert len(rejected)==2 and all('MAX_TOKENS' in r['error'] for r in rejected)
    assert list((out/'responses').rglob('*.json'))
    assert not (out/'snapshots/2022/hierarchy.json').exists()


def test_key_loader_reads_data_without_executing_shell_and_prefers_environment(tmp_path):
    from tkh.gemini_labels import load_api_key
    env=tmp_path/'.env';env.write_text('# comment\nGEMINI_API_KEY="$(touch should-not-exist)"\n')
    assert load_api_key(env,environ={})=='$(touch should-not-exist)'
    assert load_api_key(env,environ={'GEMINI_API_KEY':'override'})=='override'
    assert not (tmp_path/'should-not-exist').exists()


def test_rest_request_keeps_key_out_of_url_and_returns_only_final_json():
    from tkh.gemini_labels import request_gemini, extract_label
    captured=[]
    class Response(BytesIO):
        def __enter__(self):return self
        def __exit__(self,*args):self.close()
    def opener(req,timeout):
        captured.append(req)
        return Response(json.dumps(reply()).encode())
    payload=request_gemini('gemini-3.8-flash','one prompt','super-secret',opener=opener)
    request=captured[0]
    assert request.full_url=='https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent'
    assert request.get_header('X-goog-api-key')=='super-secret'
    body=json.loads(request.data)
    assert body['contents'][0]['parts'][0]['text']=='one prompt'
    assert body['generationConfig']['responseMimeType']=='application/json'
    assert 'super-secret' not in request.full_url and 'super-secret' not in request.data.decode()
    assert extract_label(payload)[0]['label']=='Graph model evaluation'


def test_rate_limits_retry_but_auth_failure_stops_and_redacts_key():
    from tkh.gemini_labels import request_gemini
    calls=[];sleeps=[]
    def limited(req,timeout):
        calls.append(req)
        if len(calls)==1:raise HTTPError(req.full_url,429,'quota',{'Retry-After':'2'},BytesIO(b'{"error":{"message":"quota"}}'))
        return BytesIO(json.dumps(reply()).encode())
    assert request_gemini('gemini-3.8-flash','prompt','private',opener=limited,sleep=sleeps.append)['candidates']
    assert sleeps==[2.] and len(calls)==2
    calls.clear()
    def denied(req,timeout):
        calls.append(req)
        raise HTTPError(req.full_url,403,'denied',{},BytesIO(b'{"error":{"message":"invalid private"}}'))
    with pytest.raises(RuntimeError) as error:
        request_gemini('gemini-3.8-flash','prompt','private',opener=denied,sleep=sleeps.append)
    assert len(calls)==1 and 'private' not in str(error.value)


def batch_fixture(tmp_path):
    run,prompt,h=fixture(tmp_path)
    h['levels'][1]['groups']=[{'id':'fine-a','parent_id':'broad','members':[f'n{i}' for i in range(4)]},
                             {'id':'fine-b','parent_id':'broad','members':[f'n{i}' for i in range(4,8)]}]
    (run/'snapshots/2022/hierarchy.json').write_text(json.dumps(h))
    return run,prompt


def test_batches_keep_cutoffs_and_levels_separate_and_never_expose_heldout_sources(tmp_path):
    from tkh.gemini_labels import prepare_jobs, plan_batches
    run,prompt=batch_fixture(tmp_path)
    jobs=prepare_jobs(run,prompt)
    # A second group in another snapshot and one conflicting heldout source.
    other=deepcopy(jobs[-1]);other.update(id='later',year=2023)
    bad=deepcopy(jobs[-1]);bad['id']='conflict'
    bad['input']['support']['source_ids']=jobs[-2]['input']['generation']['source_ids']
    batches=plan_batches([*jobs,other,bad],batch_size=10,max_chars=100000)
    assert sorted(len(b) for b in batches)==[1,1,1,2]
    assert sum(map(len,batches))==5
    for batch in batches:
        assert len({(j['year'],j['level']) for j in batch})==1
        sent=set().union(*(set(j['input']['generation']['source_ids']) for j in batch))
        support=set().union(*(set(j['input']['support']['source_ids']) for j in batch))
        assert sent.isdisjoint(support)


def test_batch_response_is_bound_to_exact_ids_not_return_order(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-key');calls=[]
    def send(model,prompt,key):
        calls.append(prompt)
        if 'BATCH_EVIDENCE:\n' not in prompt:return reply()
        groups=json.loads(prompt.split('BATCH_EVIDENCE:\n')[1])
        return reply(json.dumps([{'group_id':g['group_id'],'label':'Atomistic graph models',
            'gloss':'Graph models connect the supplied methodological statements with their evaluations.',
            'status':'mixed'} for g in reversed(groups)]))
    result=run_labels(run,out,prompt_path=prompt,batch_size=10,rpm=100000,send=send)
    assert result['completed']==3 and len(calls)==2
    records=json.loads((out/'snapshots/2022/labels.json').read_text())
    assert {r['id'] for r in records}=={'broad','fine-a','fine-b'}
    fine=[r for r in records if r['level']==1]
    assert len({r['batch_prompt_sha256'] for r in fine})==1
    assert len({r['response_file'] for r in fine})==1
    def forbidden(*a):raise AssertionError('Completed groups were sent again')
    run_labels(run,out,prompt_path=prompt,batch_size=10,send=forbidden)


@pytest.mark.parametrize('corruption',['duplicate','missing','extra','invalid'])
def test_batch_cannot_partially_commit_wrong_group_mapping(corruption):
    from tkh.gemini_labels import extract_batch_labels
    rows=[{'group_id':g,'label':'Graph model evaluation',
        'gloss':'Graph models connect the supplied methodological statements with their evaluations.',
        'status':'mixed'} for g in ['a','b']]
    if corruption=='duplicate':rows[1]['group_id']='a'
    elif corruption=='missing':rows.pop()
    elif corruption=='extra':rows[1]['group_id']='unexpected'
    else:rows[1]['gloss']=''
    with pytest.raises(ValueError):extract_batch_labels(reply(json.dumps(rows)),{'a','b'})


def test_batch_limit_counts_groups_and_preparation_records_request_count(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-key');calls=[]
    def send(*args):calls.append(args);return reply()
    prepared=run_labels(run,out,prompt_path=prompt,batch_size=10,prepare_only=True)
    assert prepared['planned_requests']==2
    result=run_labels(run,out,prompt_path=prompt,batch_size=10,limit=1,send=send)
    assert result['completed']==1 and len(calls)==1


def test_large_batch_is_split_without_truncating_group_evidence(tmp_path):
    from tkh.gemini_labels import prepare_jobs,plan_batches,batch_prompt
    run,prompt=batch_fixture(tmp_path)
    jobs=[j for j in prepare_jobs(run,prompt) if j['level']==1]
    maximum=max(len(batch_prompt([j])) for j in jobs)
    assert len(plan_batches(jobs,batch_size=10,max_chars=maximum))==2
    with pytest.raises(ValueError,match='large|size|characters'):
        plan_batches(jobs,batch_size=10,max_chars=10)


def test_batch_wire_schema_accepts_an_array_with_group_ids():
    from tkh.gemini_labels import request_gemini_batch
    captured=[]
    def opener(req,timeout):captured.append(req);return BytesIO(json.dumps(reply('[]')).encode())
    request_gemini_batch('gemini-3.8-flash','batch prompt','fake-key',opener=opener)
    config=json.loads(captured[0].data)['generationConfig']
    assert config['responseSchema']['type']=='ARRAY'
    assert 'group_id' in config['responseSchema']['items']['required']


def test_length_warning_mode_keeps_original_text_but_still_rejects_invalid_content(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels,extract_label,extract_batch_labels
    run,prompt,_=fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-key')
    label={'label':'Graph neural network models for predicting properties and forces of atomic materials systems',
        'gloss':' '.join(['Scientific']*45+['models.']),'status':'coherent'}
    response=reply(json.dumps(label))
    with pytest.raises(ValueError):extract_label(response)
    assert extract_label(response,length_policy='warn')[0]==label
    rows=[dict(label,group_id='fine')]
    assert extract_batch_labels(reply(json.dumps(rows)),{'fine'},length_policy='warn')['fine'][0]==label
    result=run_labels(run,out,prompt_path=prompt,length_policy='warn',send=lambda *a:response,rpm=100000)
    assert result['completed']==2 and result['labels_with_length_warnings']==2
    saved=json.loads((out/'snapshots/2022/labels.json').read_text())[0]
    assert saved['label']==label['label'] and saved['gloss']==label['gloss']
    assert len(saved['length_warnings'])==2
    assert json.loads((out/'protocol.json').read_text())['length_policy']=='warn'
    def forbidden(*a):raise AssertionError('API used for a cached label')
    run_labels(run,out,prompt_path=prompt,length_policy='warn',send=forbidden)
    for bad in [dict(label,gloss=''),dict(label,status='invented')]:
        with pytest.raises(ValueError):extract_label(reply(json.dumps(bad)),length_policy='warn')


def test_multi_sentence_gloss_is_preserved_and_reviewed_without_rejecting_the_batch(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    from test_nvidia_labels import reply as nvidia_reply
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    gloss=('One sentence: 34 words. The group covers interatomic potentials, ML potentials, '
           'and related components, problems, and tasks in materials modeling.')
    def send(model,prompt,key):
        if 'BATCH_EVIDENCE:\n' not in prompt:return nvidia_reply()
        groups=json.loads(prompt.split('BATCH_EVIDENCE:\n')[1])
        return nvidia_reply(json.dumps({'groups':[
            {'group_id':g['group_id'],'label':'Interatomic potential models','gloss':gloss,'status':'coherent'}
            for g in groups]}))
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',send=send,rpm=100000)
    assert result['completed']==3 and result['rejected_this_invocation']==0
    assert result['labels_with_quality_warnings']==2
    rows=json.loads((out/'snapshots/2022/labels.json').read_text())
    for row in rows:
        if row['level']==1:
            assert row['gloss']==gloss and any('sentence' in w for w in row['quality_warnings'])
    assert len(json.loads((out/'label_review.json').read_text())['items'])==2
    def forbidden(*a):raise AssertionError('Style warnings caused additional API requests')
    assert run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',send=forbidden)['completed']==3


def test_nvidia_batches_use_an_object_envelope_and_record_the_schema(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    httpx=pytest.importorskip('httpx');openai=pytest.importorskip('openai')
    from test_nvidia_labels import reply as nvidia_reply
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    requests=[];real_client=openai.OpenAI
    def handler(request):
        body=json.loads(request.content);requests.append(body)
        prompt=body['messages'][0]['content']
        if 'BATCH_EVIDENCE:\n' not in prompt:return httpx.Response(200,json=nvidia_reply())
        groups=json.loads(prompt.split('BATCH_EVIDENCE:\n')[1])
        assert '{"groups":' in prompt
        content=json.dumps({'groups':[{'group_id':g['group_id'],'label':'Graph model evaluation',
            'gloss':'Graph models connect the supplied methodological statements with their evaluations.',
            'status':'mixed'} for g in groups]})
        return httpx.Response(200,json=nvidia_reply(content))
    def client_factory(**kwargs):
        kwargs['http_client']=httpx.Client(transport=httpx.MockTransport(handler))
        return real_client(**kwargs)
    monkeypatch.setattr(openai,'OpenAI',client_factory)
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',rpm=100000)
    assert result['completed']==3 and result['completed_requests']==2
    assert len(requests)==2
    for path in (out/'requests').glob('*.json'):
        saved=json.loads(path.read_text())
        sent=next(r for r in requests if r['messages'][0]['content']==saved['prompt'])
        assert saved['guided_json']==sent['guided_json']==saved['validation_schema']
        assert saved['decoding']=='guided_json' and 'response_format' not in sent
    assert json.loads((out/'protocol.json').read_text())['generation']['response_format']=='guided_json'
    rows=json.loads((out/'snapshots/2022/labels.json').read_text())
    assert all(r['response_format']=='guided_json' for r in rows)


def test_nvidia_old_json_mode_run_cannot_be_reused_as_guided_generation(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path);out=tmp_path/'old'
    run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',prepare_only=True)
    path=out/'protocol.json';protocol=json.loads(path.read_text())
    protocol['schema_version']='api-labels-v4';protocol['generation']['response_format']='json_object'
    path.write_text(json.dumps(protocol));original=path.read_bytes()
    def forbidden(*args,**kwargs):raise AssertionError('Old run reused or API accessed')
    monkeypatch.setattr('tkh.gemini_labels.load_api_key',forbidden)
    with pytest.raises(ValueError,match='Protocol changed'):
        run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test')
    assert path.read_bytes()==original


def test_progress_counts_remaining_requests_and_cached_groups(tmp_path,monkeypatch,capsys):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-key')
    run_labels(run,out,prompt_path=prompt,limit=1,send=lambda *a:reply(),rpm=100000)
    capsys.readouterr()
    def send(model,prompt,key):
        groups=json.loads(prompt.split('BATCH_EVIDENCE:\n')[1])
        return reply(json.dumps([{'group_id':g['group_id'],'label':'Graph model evaluation',
            'gloss':'Graph models connect the supplied methodological statements with their evaluations.',
            'status':'mixed'} for g in groups]))
    run_labels(run,out,prompt_path=prompt,send=send,rpm=100000)
    progress=capsys.readouterr().err
    assert '1/1 requests' in progress and 'groups=3/3' in progress
    assert 'elapsed' in progress and 'ETA' in progress


def test_failed_batch_advances_attempt_count_but_not_completed_groups(tmp_path,monkeypatch,capsys):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-key')
    def send(model,prompt,key):return reply('not JSON') if 'BATCH_EVIDENCE:\n' in prompt else reply()
    result=run_labels(run,out,prompt_path=prompt,send=send,rpm=100000,max_attempts=1)
    progress=capsys.readouterr().err
    assert '2/2 requests' in progress and 'groups=1/3' in progress
    assert result['completed']==1 and result['rejected_this_invocation']==1


@pytest.mark.parametrize('previous_client',[
    '3209bbc1acaa670fde6659c88ce04064f21262316eea2fa57a520d40679ccb2a',
    '7d374741af300382aaa17db2d8b9143747534bfc455fd9d193bc2d0c5bbda3b1',
    '47dd61a36948a78a997bdb8e4a17bd07104afe9b044d73bc227cb608e76c26a1'])
def test_progress_update_can_resume_the_previous_client_without_changing_its_protocol(tmp_path,monkeypatch,previous_client):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','fake-key')
    run_labels(run,out,prompt_path=prompt,send=lambda *a:reply(),rpm=100000)
    path=out/'protocol.json';protocol=json.loads(path.read_text())
    protocol['source_sha256']=previous_client
    path.write_text(json.dumps(protocol));before=path.read_bytes()
    monkeypatch.delenv('GEMINI_API_KEY')
    def forbidden(*a):raise AssertionError('Cached groups were regenerated')
    result=run_labels(run,out,prompt_path=prompt,send=forbidden,env_file=tmp_path/'absent')
    assert result['completed']==2 and path.read_bytes()==before
    assert (out/'client_code_versions.json').exists()
    protocol['source_sha256']='unreviewed-code-change';path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='Protocol changed'):
        run_labels(run,out,prompt_path=prompt,send=forbidden)


def test_template_phrase_flags_review_and_does_not_stop_later_batches(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out';calls=[]
    monkeypatch.setenv('GEMINI_API_KEY','fake-key')
    def send(model,prompt,key):
        calls.append(prompt)
        if 'BATCH_EVIDENCE:\n' not in prompt:
            return reply(json.dumps({'label':'First principles molecular dynamics',
                'gloss':'A collection of methods centered on first-principles molecular dynamics and electronic structure simulation.',
                'status':'coherent'}))
        groups=json.loads(prompt.split('BATCH_EVIDENCE:\n')[1])
        return reply(json.dumps([{'group_id':g['group_id'],'label':'Graph model evaluation',
            'gloss':'Graph models connect the supplied methodological statements with their evaluations.',
            'status':'mixed'} for g in groups]))
    result=run_labels(run,out,prompt_path=prompt,send=send,rpm=100000)
    assert result['completed']==3 and len(calls)==2
    assert result['labels_with_quality_warnings']==1
    flagged=json.loads((out/'label_review.json').read_text())['items']
    assert len(flagged)==1 and flagged[0]['id']=='broad'
    assert 'collection of methods' in flagged[0]['quality_warnings'][0]
    rows=json.loads((out/'snapshots/2022/labels.json').read_text())
    broad=next(r for r in rows if r['id']=='broad')
    assert broad['status']=='coherent' and broad['quality_warnings']
    def forbidden(*a):raise AssertionError('Review warnings caused repeated API requests')
    assert run_labels(run,out,prompt_path=prompt,send=forbidden)['completed']==3


def test_invalid_response_is_saved_and_later_groups_are_still_processed(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out';calls=[]
    monkeypatch.setenv('GEMINI_API_KEY','fake-key')
    def send(model,prompt,key):
        group=json.loads(prompt.split('EVIDENCE:\n')[1])['group_id'];calls.append(group)
        return reply('not JSON') if group=='fine-a' else reply()
    result=run_labels(run,out,prompt_path=prompt,send=send,rpm=100000,batch_size=1,max_attempts=1)
    assert calls==['broad','fine-a','fine-b']
    assert result['completed']==2 and result['remaining']==1 and result['rejected_this_invocation']==1
    rejection=json.loads(next((out/'rejected').glob('*.json')).read_text())
    assert rejection['group_ids']==['fine-a'] and rejection['error']
    assert (out/rejection['response_file']).exists()
