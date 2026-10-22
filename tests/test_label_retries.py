"""Bounded user-run generation retries, using the real SDK without a network."""
import json

import pytest

from test_gemini_labels import batch_fixture
from test_nvidia_labels import reply


def wire(monkeypatch,handler):
    httpx=pytest.importorskip('httpx');openai=pytest.importorskip('openai')
    real_client=openai.OpenAI
    def factory(**kwargs):
        kwargs['http_client']=httpx.Client(transport=httpx.MockTransport(handler))
        return real_client(**kwargs)
    monkeypatch.setattr(openai,'OpenAI',factory)
    monkeypatch.setenv('NVIDIA_API_KEY','private-key')
    return httpx


def response_for(body):
    prompt=body['messages'][0]['content']
    if 'BATCH_EVIDENCE:\n' not in prompt:return reply()
    groups=json.loads(prompt.split('BATCH_EVIDENCE:\n')[1])
    return reply(json.dumps({'groups':[{'group_id':g['group_id'],
        'label':'Graph model evaluation','gloss':'Graph models connect supplied methods and evaluations.',
        'status':'mixed'} for g in groups]}))


@pytest.mark.parametrize('failure',['length','duplicate_ids','not_json'])
def test_bad_batch_retries_as_individual_groups_and_keeps_every_attempt(tmp_path,monkeypatch,failure):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out';requests=[]
    def handler(req):
        body=json.loads(req.content);requests.append(body)
        response=response_for(body)
        if 'BATCH_EVIDENCE:\n' in body['messages'][0]['content']:
            if failure=='length':response['choices'][0]['finish_reason']='length'
            elif failure=='not_json':response['choices'][0]['message']['content']='not JSON'
            else:
                data=json.loads(response['choices'][0]['message']['content'])
                data['groups'][1]['group_id']=data['groups'][0]['group_id']
                response['choices'][0]['message']['content']=json.dumps(data)
        return httpx.Response(200,json=response)
    httpx=wire(monkeypatch,handler)
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',max_attempts=3,rpm=100000,sleep=lambda _:None)
    assert result['completed']==3 and result['failed_groups_this_invocation']==0
    assert len(requests)==4 and all(r['max_tokens']==512 for r in requests)
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['batch_size']==3
    assert all('guided_json' in r and 'response_format' not in r for r in requests)
    assert result['rejected_this_invocation']==1 and result['retried_requests_this_invocation']==2
    assert len(list((out/'responses').glob('*.json')))==4
    assert len(list((out/'rejected').glob('*.json')))==1
    records=[json.loads(p.read_text()) for p in (out/'completed').glob('*.json')]
    assert {r['id'] for r in records}=={'broad','fine-a','fine-b'}
    for row in records:
        saved=json.loads((out/row['request_file']).read_text())
        assert saved['max_output_tokens']==512
        if row['level']==1:
            assert saved['attempt']==2 and saved['retry_of']
            assert len(saved['group_ids'])==1
    def forbidden(*a,**kw):raise AssertionError('Completed labels were requested again')
    monkeypatch.setattr('tkh.nvidia_labels.request_nvidia',forbidden)
    assert run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',max_output_tokens=512,max_attempts=3)['completed']==3


@pytest.mark.parametrize('status',[429,500])
def test_transient_api_error_retries_same_batch_and_honors_backoff(tmp_path,monkeypatch,status):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out';requests=[];pauses=[]
    def handler(req):
        body=json.loads(req.content);requests.append(body)
        if len(requests)==1:
            return httpx.Response(status,json={'error':'try later'},headers={'retry-after':'4'})
        return httpx.Response(200,json=response_for(body))
    httpx=wire(monkeypatch,handler)
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        length_policy='warn',max_attempts=3,rpm=100000,sleep=pauses.append)
    assert len(requests)==3 and result['completed']==3
    assert result['retried_requests_this_invocation']==1 and 4.0 in pauses
    assert len(list((out/'request_errors').glob('*.json')))==1
    for p in out.rglob('*.json'):assert 'private-key' not in p.read_text()


@pytest.mark.parametrize('status',[400,401,403,404,422])
def test_permanent_api_error_stops_without_restarting(tmp_path,monkeypatch,status):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out';requests=[]
    def handler(req):requests.append(req);return httpx.Response(status,json={'error':'private-key invalid request'})
    httpx=wire(monkeypatch,handler)
    with pytest.raises(RuntimeError,match=str(status)):
        run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
            max_attempts=3,rpm=100000,sleep=lambda _:None)
    assert len(requests)==1
    assert json.loads((out/'summary.json').read_text())['status']=='interrupted_or_error'
    for p in out.rglob('*.json'):assert 'private-key' not in p.read_text()


def test_retry_exhaustion_is_bounded_and_does_not_complete_bad_groups(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out';requests=[]
    def handler(req):requests.append(req);return httpx.Response(200,json=reply('not JSON'))
    httpx=wire(monkeypatch,handler)
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
        max_attempts=3,rpm=100000,sleep=lambda _:None)
    assert len(requests)==8
    assert result['completed']==0 and result['remaining']==3
    assert result['failed_groups_this_invocation']==3
    assert result['attempted_requests_this_invocation']==8
    assert len(list((out/'requests').glob('*.json')))==8
    assert not list((out/'completed').glob('*.json'))


def test_persistent_api_outage_stops_after_three_calls_instead_of_trying_every_group(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out';requests=[]
    def handler(req):requests.append(req);return httpx.Response(503,json={'error':'unavailable'})
    httpx=wire(monkeypatch,handler)
    with pytest.raises(RuntimeError,match='attempt limit'):
        run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',
            max_attempts=3,rpm=100000,sleep=lambda _:None)
    assert len(requests)==3
    assert json.loads((out/'summary.json').read_text())['status']=='interrupted_or_error'


def test_cli_returns_success_after_a_rejected_attempt_is_recovered(tmp_path,monkeypatch):
    from tkh.gemini_labels import main
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'out'
    def handler(req):
        body=json.loads(req.content);response=response_for(body)
        if 'BATCH_EVIDENCE:\n' in body['messages'][0]['content']:
            response['choices'][0]['finish_reason']='length'
        return httpx.Response(200,json=response)
    httpx=wire(monkeypatch,handler)
    monkeypatch.setattr('sys.argv',['name_groups.py','--provider','nvidia','--model','nvidia/test',
        '--run',str(run),'--prompt',str(prompt),'--out',str(out),'--length-policy','warn',
        '--max-attempts','3','--retry-delay','0','--rpm','100000'])
    assert main()==0
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='complete' and summary['rejected_this_invocation']==1


def test_short_gloss_is_valid_in_strict_mode():
    from tkh.gemini_labels import extract_label,label_length_warnings
    label={'label':'Atomistic force prediction','gloss':'Models predict forces in atomic systems.','status':'coherent'}
    parsed,_=extract_label(reply(json.dumps(label)),structured=True)
    assert parsed==label and label_length_warnings(parsed)==[]


def test_default_generation_sends_batches_without_artificial_rate_pauses(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path);seen=[];pauses=[]
    def handler(req):
        body=json.loads(req.content);seen.append(body)
        return httpx.Response(200,json=response_for(body))
    httpx=wire(monkeypatch,handler)
    result=run_labels(run,tmp_path/'labels',prompt_path=prompt,provider='nvidia',
                      model='nvidia/test',sleep=pauses.append)
    assert result['completed']==3
    assert len(seen)==2  # One coarse request, one actual two-group fine batch.
    assert 'BATCH_EVIDENCE' in seen[1]['messages'][0]['content']
    assert pauses==[]


def test_zero_rpm_means_no_client_rate_cap(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt=batch_fixture(tmp_path)
    httpx=wire(monkeypatch,lambda req:httpx.Response(200,json=response_for(json.loads(req.content))))
    result=run_labels(run,tmp_path/'labels',prompt_path=prompt,provider='nvidia',
                      model='nvidia/test',rpm=0)
    assert result['completed']==3
