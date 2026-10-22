"""Offline deadlines, parallelism, shared quota and preserved judge caches."""
import asyncio
import json
import time

import pytest

from test_label_judging import packet
from test_nvidia_labels import reply


def judgment():
    return reply(json.dumps({'rating':'accurate','coverage':'representative',
        'reason':'The scientific subject is supported.','evidence_ids':[]}))


def test_async_sdk_preserves_schema_tokens_and_disabled_reasoning():
    from tkh.nvidia_async import request_nvidia_async
    httpx=pytest.importorskip('httpx');seen=[]
    async def handler(req):
        seen.append(req);return httpx.Response(200,json=judgment())
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await request_nvidia_async('nvidia/test','prompt','secret',http_client=client)
    assert asyncio.run(run())['model']=='nvidia/test-version'
    body=json.loads(seen[0].content)
    assert body['stream'] is False and body['max_tokens']==512
    assert body['chat_template_kwargs']['enable_thinking'] is False
    assert body['response_format']['json_schema']['strict'] is True
    assert 'guided_json' not in body and 'secret' not in seen[0].content.decode()


def test_async_sdk_deadline_cancels_an_unresponsive_transport():
    from tkh.nvidia_async import request_nvidia_async
    httpx=pytest.importorskip('httpx');cancelled=[]
    async def handler(req):
        try:await asyncio.sleep(30)
        finally:cancelled.append(True)
        return httpx.Response(200,json=judgment())
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RuntimeError,match='deadline'):
                await asyncio.wait_for(request_nvidia_async('nvidia/test','prompt','secret',
                    http_client=client,timeout=.03),timeout=1.)
    asyncio.run(run())
    assert cancelled==[True]


def test_parallel_judge_caps_inflight_calls_and_persists_each_answer(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');active=peak=0
    async def send(*a,**kw):
        nonlocal active,peak
        active+=1;peak=max(peak,active)
        try:await asyncio.sleep(.01);return judgment()
        finally:active-=1
    result=run_judging(packet(tmp_path),tmp_path/'judge',send=send,sleep=lambda _:None,
        rpm=100000,workers=3)
    assert result['completed']==5 and peak==3 and active==0
    assert len(list((tmp_path/'judge/completed').glob('*.json')))==5


def test_total_budget_cancels_pending_work_and_keeps_completed_answers(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');calls=active=0
    async def send(*a,**kw):
        nonlocal calls,active
        calls+=1
        if calls==1:return judgment()
        active+=1
        try:await asyncio.sleep(30)
        finally:active-=1
    with pytest.raises(RuntimeError,match='time budget'):
        run_judging(packet(tmp_path),tmp_path/'judge',send=send,sleep=lambda _:None,
            rpm=100000,workers=2,request_timeout=1.,time_budget=.05)
    summary=json.loads((tmp_path/'judge/summary.json').read_text())
    assert summary['completed']==1 and active==0
    assert summary['sample_size']<summary['planned_requests']
    assert list((tmp_path/'judge/executions').glob('*.json'))


def test_operational_upgrade_resumes_exact_old_scientific_protocol(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');run=packet(tmp_path);out=tmp_path/'judge'
    run_judging(run,out,send=lambda *a,**kw:judgment(),sleep=lambda _:None,rpm=100000)
    path=out/'protocol.json';protocol=json.loads(path.read_text())
    protocol['source_sha256']='628af1208ff67eef51efc5f7ebb82f6d906de65d8bd0dd45a93e131eef0823e2'
    path.write_text(json.dumps(protocol));original=path.read_bytes()
    def forbidden(*a,**kw):raise AssertionError('Cached answers or credentials accessed again')
    monkeypatch.setattr('tkh.label_judging.load_api_key',forbidden)
    result=run_judging(run,out,send=forbidden,workers=4,request_timeout=10.)
    assert result['completed']==5 and path.read_bytes()==original
    assert (out/'client_code_versions.json').exists()
    protocol['task_inputs_sha256']='different-evidence';path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='protocol changed|Protocol changed'):
        run_judging(run,out,prepare_only=True)


def test_default_judge_has_no_artificial_rate_pause(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');pauses=[]
    result=run_judging(packet(tmp_path),tmp_path/'judge',send=lambda *a,**kw:judgment(),
                       sleep=pauses.append)
    assert result['completed']==5 and pauses==[]
