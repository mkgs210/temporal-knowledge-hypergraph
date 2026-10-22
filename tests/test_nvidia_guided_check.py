"""Exercise the user-run endpoint probe through the real SDK, without network."""
import json

import pytest

from test_nvidia_labels import reply


@pytest.mark.parametrize('answer,passes',[
    ('constrained',True),('NOT_JSON',False),('wrong_nonce',False)])
def test_probe_checks_hidden_schema_value_instead_of_http_success(tmp_path,monkeypatch,answer,passes):
    from tkh.nvidia_guided_check import check_guided
    httpx=pytest.importorskip('httpx');pytest.importorskip('openai')
    monkeypatch.setenv('NVIDIA_API_KEY','private-key');requests=[]
    def handler(req):
        body=json.loads(req.content);requests.append(body)
        nonce=body['guided_json']['properties']['proof']['enum'][0]
        assert nonce not in body['messages'][0]['content']
        assert 'NOT_JSON' in body['messages'][0]['content']
        assert body['max_tokens']==256 and 'response_format' not in body
        text=json.dumps({'proof':nonce}) if answer=='constrained' else (
            'NOT_JSON' if answer=='NOT_JSON' else '{"proof":"wrong"}')
        return httpx.Response(200,json=reply(text))
    result=check_guided(tmp_path,env_file=tmp_path/'absent',
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert len(requests)==1 and result['passed'] is passes
    assert json.loads((tmp_path/'summary.json').read_text())==result
    request=json.loads((tmp_path/result['request_file']).read_text())
    assert request['guided_json']==requests[0]['guided_json']
    assert (tmp_path/result['response_file']).is_file()
    for p in tmp_path.rglob('*.json'):assert 'private-key' not in p.read_text()


def test_probe_preparation_needs_no_key_or_api(tmp_path,monkeypatch):
    from tkh.nvidia_guided_check import check_guided
    def forbidden(*args,**kwargs):raise AssertionError('Probe used key or API during preparation')
    monkeypatch.setattr('tkh.nvidia_guided_check.load_api_key',forbidden)
    monkeypatch.setattr('tkh.nvidia_guided_check.request_nvidia',forbidden)
    result=check_guided(tmp_path,prepare_only=True)
    assert result['status']=='prepared' and result['passed'] is False
    assert result['api_requests']==0 and not list((tmp_path/'responses').glob('*.json'))
