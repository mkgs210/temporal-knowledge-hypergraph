"""NVIDIA API wiring and small paired comparison, with no real API requests."""
from io import BytesIO
import json
from urllib.error import HTTPError

import pytest

from test_gemini_labels import fixture, batch_fixture


def test_model_comparison_has_no_default_request_rate_pause(tmp_path,monkeypatch):
    from tkh.nvidia_labels import compare_models
    run,prompt=batch_fixture(tmp_path);pauses=[]
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    compare_models(run,tmp_path/'comparison',year=2022,prompt_path=prompt,
                   models=['nvidia/a','nvidia/b'],send=lambda *a:reply(),sleep=pauses.append)
    assert pauses==[]


def reply(text=None):
    return {'id':'chatcmpl-test','model':'nvidia/test-version',
        'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','reasoning_content':'not a label',
            'content':text or json.dumps({'label':'Graph model evaluation',
                'gloss':'Graph models connect the supplied methodological statements with their evaluations.','status':'mixed'})}}],
        'usage':{'prompt_tokens':30,'completion_tokens':20,'total_tokens':50}}


@pytest.mark.parametrize('model',[
    'nvidia/nemotron-3.5-lightning-30b-a3b',
    'nvidia/nemotron-3.5-super-vl-120b-a12b'])
@pytest.mark.parametrize('decoding',['guided_json','json_schema'])
def test_nvidia_sdk_sends_schema_constraints_and_bearer_header(model,decoding):
    from tkh.nvidia_labels import request_nvidia
    httpx=pytest.importorskip('httpx');pytest.importorskip('openai')
    requests=[]
    def handler(req):requests.append(req);return httpx.Response(200,json=reply())
    result=request_nvidia(model,'prompt','private-key',decoding=decoding,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    req=requests[0];body=json.loads(req.content)
    assert str(req.url)=='https://integrate.api.nvidia.com/v1/chat/completions'
    assert req.headers['Authorization']=='Bearer private-key'
    assert body['messages']==[{'role':'user','content':'prompt'}]
    assert body['chat_template_kwargs']['enable_thinking'] is False
    assert body['stream'] is False
    assert body['max_tokens']==512
    if decoding=='guided_json':
        assert 'response_format' not in body
        schema=body['guided_json']
    else:
        assert 'guided_json' not in body
        spec=body['response_format']['json_schema']
        assert body['response_format']['type']=='json_schema' and spec['strict'] is True
        schema=spec['schema']
    assert schema['additionalProperties'] is False
    assert set(schema['required'])=={'label','gloss','status'}
    assert 'private-key' not in req.content.decode() and 'private-key' not in str(req.url)
    assert result['model']=='nvidia/test-version'


def test_shared_parser_accepts_nvidia_content_but_rejects_truncated_reasoning():
    from tkh.gemini_labels import extract_label
    r=reply()
    assert extract_label(r)[0]['label']=='Graph model evaluation'
    r['choices'][0]['finish_reason']='length'
    with pytest.raises(ValueError,match='length'):extract_label(r)


def test_nvidia_key_is_separate_and_all_groups_keep_raw_provider_provenance(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('GEMINI_API_KEY','wrong-provider-key')
    monkeypatch.setenv('NVIDIA_API_KEY','private-nvidia-key')
    calls=[]
    def send(model,prompt,key):calls.append(key);return reply()
    result=run_labels(run,out,prompt_path=prompt,provider='nvidia',
        model='nvidia/nemotron-3.5-lightning-30b-a3b',send=send,rpm=100000)
    assert result['completed']==2 and calls==['private-nvidia-key','private-nvidia-key']
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['provider']=='nvidia' and protocol['generation']['enable_thinking'] is False
    rows=json.loads((out/'snapshots/2022/labels.json').read_text())
    assert rows[0]['model_version']=='nvidia/test-version'
    assert rows[0]['usage']['total_tokens']==50
    for p in out.rglob('*.json'):assert 'private-nvidia-key' not in p.read_text()


def test_guided_run_refuses_labels_imported_from_another_generation_mode(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path);out=tmp_path/'out'
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',send=lambda *a:reply(),rpm=100000)
    path=next((out/'completed').glob('*.json'));row=json.loads(path.read_text())
    row['response_format']='legacy-prompt-only-imported';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='Cached.*decoding'):
        run_labels(run,out,prompt_path=prompt,provider='nvidia',model='nvidia/test',prepare_only=True)


def test_small_comparison_uses_identical_inputs_and_resumes_without_extra_calls(tmp_path,monkeypatch):
    from tkh.nvidia_labels import compare_models
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'comparison'
    monkeypatch.setenv('NVIDIA_API_KEY','private-key');calls=[]
    def send(model,prompt,key):calls.append((model,prompt));return reply()
    result=compare_models(run,out,year=2022,prompt_path=prompt,models=['nvidia/a','nvidia/b'],send=send,rpm=100000)
    assert len(calls)==6 and result['valid_outputs']==6
    generation=json.loads((out/'protocol.json').read_text())['generation']
    assert generation['decoding']=='guided_json'
    assert all('guided_json' in spec and 'response_format' not in spec
               for spec in generation['constraints'].values())
    assert [p for m,p in calls if m=='nvidia/a']==[p for m,p in calls if m=='nvidia/b']
    assert (out/'comparison.md').exists() and (out/'blind_packet.json').exists()
    result=compare_models(run,out,year=2022,prompt_path=prompt,models=['nvidia/a','nvidia/b'],send=send,rpm=100000)
    assert len(calls)==6
    assert result['quality_status']=='not independently assessed'


def test_comparison_dry_run_needs_no_key_and_no_failed_output_becomes_valid(tmp_path,monkeypatch):
    from tkh.nvidia_labels import compare_models
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'comparison'
    monkeypatch.delenv('NVIDIA_API_KEY',raising=False)
    def forbidden(*a):raise AssertionError('API used during preparation')
    prepared=compare_models(run,out,year=2022,prompt_path=prompt,models=['nvidia/a','nvidia/b'],
        prepare_only=True,send=forbidden,env_file=tmp_path/'absent')
    assert prepared['planned_requests']==6 and prepared['valid_outputs']==0
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    def malformed(model,prompt,key):return reply('not JSON')
    result=compare_models(run,out,year=2022,prompt_path=prompt,models=['nvidia/a','nvidia/b'],send=malformed,rpm=100000)
    assert result['valid_outputs']==0 and result['invalid_outputs']==6 and result['failed_requests']==0
    assert list((out/'responses').glob('*.json'))


@pytest.mark.parametrize('status',[400,401,422,500])
def test_nvidia_failed_sdk_request_is_not_retried_or_leaked_to_error(status):
    from tkh.nvidia_labels import request_nvidia
    httpx=pytest.importorskip('httpx');pytest.importorskip('openai')
    calls=[]
    def denied(req):
        calls.append(req)
        return httpx.Response(status,json={'error':'private-key unsupported guided_json'})
    with pytest.raises(RuntimeError) as exc:
        request_nvidia('nvidia/test','prompt','private-key',
            http_client=httpx.Client(transport=httpx.MockTransport(denied)))
    assert len(calls)==1 and 'private-key' not in str(exc.value)
    assert str(status) in str(exc.value) and 'fallback' in str(exc.value)


@pytest.mark.parametrize('kwargs',[
    {'response_format':{'type':'json_object'}}, {'decoding':'json_object'}])
def test_nvidia_rejects_weak_json_mode_before_sending_a_request(kwargs):
    from tkh.nvidia_labels import request_nvidia
    httpx=pytest.importorskip('httpx');pytest.importorskip('openai')
    def forbidden(req):raise AssertionError('Weak JSON mode reached the API')
    with httpx.Client(transport=httpx.MockTransport(forbidden)) as client:
        with pytest.raises(ValueError,match='schema|Schema|decoding'):
            request_nvidia('nvidia/test','prompt','private-key',http_client=client,**kwargs)


def test_nvidia_batch_schema_reaches_the_sdk_and_exact_group_ids_are_checked():
    from tkh.nvidia_labels import request_nvidia,nvidia_response_format
    from tkh.gemini_labels import extract_batch_labels
    httpx=pytest.importorskip('httpx');pytest.importorskip('openai')
    body=[]
    rows=[{'group_id':g,'label':'Graph model evaluation',
        'gloss':'Graph models connect the supplied methodological statements with their evaluations.',
        'status':'mixed'} for g in ['b','a']]
    def handler(req):body.append(json.loads(req.content));return httpx.Response(200,json=reply(json.dumps({'groups':rows})))
    response=request_nvidia('nvidia/test','Name groups a and b.','private-key',
        response_format=nvidia_response_format(['a','b']),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert 'response_format' not in body[0]
    schema=body[0]['guided_json']
    assert schema['type']=='object' and schema['required']==['groups']
    assert schema['properties']['groups']['items']['properties']['group_id']['enum']==['a','b']
    assert set(extract_batch_labels(response,{'a','b'},envelope=True))=={'a','b'}
    rows[0]['group_id']='a'
    with pytest.raises(ValueError,match='duplicate|missing'):
        extract_batch_labels(reply(json.dumps({'groups':rows})),{'a','b'},envelope=True)
    with pytest.raises(ValueError,match='JSON'):
        extract_batch_labels(reply('Here are the groups formatted as requested:\n\n'+json.dumps({'groups':rows})),
            {'a','b'},envelope=True)


def test_truncated_batch_keeps_the_real_failure_reason():
    from tkh.gemini_labels import extract_batch_labels
    response=reply('not JSON');response['choices'][0]['finish_reason']='length'
    with pytest.raises(ValueError,match='length'):extract_batch_labels(response,{'a'},envelope=True)


def test_structured_single_response_rejects_markdown_instead_of_repairing_it():
    from tkh.gemini_labels import extract_label
    text=reply()['choices'][0]['message']['content']
    with pytest.raises(ValueError,match='JSON'):
        extract_label(reply('```json\n'+text+'\n```'),structured=True)


def test_sdk_queued_response_is_not_parsed_as_a_completion():
    from tkh.nvidia_labels import request_nvidia
    httpx=pytest.importorskip('httpx');pytest.importorskip('openai')
    with pytest.raises(RuntimeError,match='202'):
        request_nvidia('nvidia/test','prompt','private-key',http_client=httpx.Client(
            transport=httpx.MockTransport(lambda req:httpx.Response(202,json={'requestId':'queued'}))))


def test_comparison_does_not_accept_edited_cached_labels(tmp_path,monkeypatch):
    from tkh.nvidia_labels import compare_models
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'comparison'
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    compare_models(run,out,year=2022,prompt_path=prompt,models=['nvidia/a'],send=lambda *a:reply(),rpm=100000)
    cache=next((out/'completed').glob('*.json'));r=json.loads(cache.read_text())
    r['label']['gloss']='This edited claim does not match the original response.'
    cache.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Cached|cached'):
        compare_models(run,out,year=2022,prompt_path=prompt,models=['nvidia/a'],send=lambda *a:reply(),rpm=100000)


def test_auth_failure_stops_pilot_before_sending_the_other_five_calls(tmp_path,monkeypatch):
    from tkh.nvidia_labels import compare_models
    run,prompt=batch_fixture(tmp_path)
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');calls=[]
    def denied(*args):calls.append(args);raise RuntimeError('NVIDIA HTTP 401: invalid key')
    r=compare_models(run,tmp_path/'out',year=2022,prompt_path=prompt,models=['nvidia/a','nvidia/b'],send=denied,rpm=100000)
    assert len(calls)==1 and r['failed_requests']==1 and r['completed_requests']==1


def test_format_rejections_do_not_hide_slow_replies_or_discard_their_text(tmp_path):
    from tkh.gemini_labels import digest,write
    from tkh.nvidia_labels import _outputs
    import hashlib
    jobs=[{'id':f'g{i}','year':2022,'level':1,'input':{'support':{'nodes':[],'claim_context':[]}}} for i in range(3)]
    records={}
    for job,seconds in zip(jobs,[2.,14.,44.]):
        token=digest(['nvidia/test',2022,1,job['id']]);path=tmp_path/'responses'/f'{token}.json'
        label={'label':'Graph neural network models for predicting properties of atomic materials',
            'gloss':'Graph models connect the supplied methodological statements with their evaluations.','status':'coherent'}
        write(path,reply(json.dumps(label)))
        records[token]={**job,'requested_model':'nvidia/test','elapsed_seconds':seconds,'status':'error',
            'error':'label/gloss length does not explain a theme','response_file':str(path.relative_to(tmp_path)),
            'response_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    result=_outputs(tmp_path,jobs,['nvidia/test'],records)
    assert result['models']['nvidia/test']['median_request_seconds']==14.
    assert result['responses_received']==3 and result['invalid_outputs']==3 and result['failed_requests']==0
    assert label['label'] in (tmp_path/'comparison.md').read_text()
    assert len(json.loads((tmp_path/'blind_packet.json').read_text())['tasks'])==3


def test_saved_inspection_needs_no_key_or_network_and_preserves_original_files(tmp_path,monkeypatch):
    from tkh.nvidia_labels import compare_models,inspect_saved_comparison
    run,prompt=batch_fixture(tmp_path);out=tmp_path/'comparison'
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    compare_models(run,out,year=2022,prompt_path=prompt,models=['nvidia/a'],send=lambda *a:reply(),rpm=100000)
    before={p:p.read_bytes() for p in out.rglob('*') if p.is_file()}
    # An older source-code fingerprint must not prevent inspecting saved evidence.
    protocol=json.loads((out/'protocol.json').read_text());protocol['source_sha256']='older-code'
    (out/'protocol.json').write_text(json.dumps(protocol));before[out/'protocol.json']=(out/'protocol.json').read_bytes()
    monkeypatch.delenv('NVIDIA_API_KEY')
    def forbidden(*a,**kw):raise AssertionError('Network or key access during inspection')
    monkeypatch.setattr('tkh.nvidia_labels.request_nvidia',forbidden)
    monkeypatch.setattr('tkh.nvidia_labels.load_api_key',forbidden)
    result=inspect_saved_comparison(out)
    assert result['valid_outputs']==3
    assert (out/'inspection/comparison.md').exists()
    assert all(p.read_bytes()==data for p,data in before.items())
    response=next((out/'responses').glob('*.json'));response.write_text('{}')
    with pytest.raises(ValueError,match='response|Response'):inspect_saved_comparison(out)
