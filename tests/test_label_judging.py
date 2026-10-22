"""Offline checks for a separate, evidence-bound label judge."""
import json

import pytest

from test_nvidia_labels import reply


def packet(tmp_path):
    run=tmp_path/'labels';(run/'audit').mkdir(parents=True)
    (run/'protocol.json').write_text(json.dumps({'model':'nvidia/generator'}))
    tasks=[{'task_id':'sample-000','cutoff':2026,'level':1,'label':'Atomic force prediction',
        'gloss':'Models predict atomic forces.','held_out_members':[
            {'id':'n1','type':'claim','surface_form':'The graph model predicts atomic forces.'}],
        'held_out_claims':[],'rating':None}]
    (run/'audit/diagnostic_packet.json').write_text(json.dumps(tasks))
    return run


def test_empty_evidence_rule_has_own_provenance_and_does_not_forge_a_model_call(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    monkeypatch.setenv('NVIDIA_API_KEY','fake');calls=[]
    def send(model,prompt,key,**kw):
        data=json.loads(prompt.split('EVIDENCE:\n')[1]);calls.append(data)
        assert data['held_out_members'] or data['held_out_claims']
        return reply(json.dumps({'rating':'accurate','coverage':'representative','reason':'Supported subject.','evidence_ids':[]}))
    run=packet(tmp_path);out=tmp_path/'judge'
    result=run_judging(run,out,send=send,empty_evidence_rule=True)
    assert len(calls)==4 and result['completed']==5 and result['rule_assessments']==1
    row=next(r for r in json.loads((out/'judgments.json').read_text()) if r.get('assessment_source')=='empty_evidence_rule')
    assert row['request_file'] is None and row['model_version'] is None
    raw=json.loads((out/row['response_file']).read_text())
    assert 'choices' not in raw and raw['judgment']['rating']=='insufficient_evidence'
    assert run_judging(run,out,prepare_only=True,empty_evidence_rule=True)['rule_assessments']==1


def test_judge_preparation_does_not_read_keys_or_call_a_model(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    def forbidden(*a,**kw):raise AssertionError('Inference or key access in preparation')
    monkeypatch.setattr('tkh.label_judging.load_api_key',forbidden)
    monkeypatch.setattr('tkh.label_judging.request_nvidia_async',forbidden)
    result=run_judging(packet(tmp_path),tmp_path/'judge',prepare_only=True)
    assert result['planned_requests']==5 and result['completed']==0
    assert result['scientific_quality_established'] is False


def test_judge_rejects_invented_evidence_and_sensitivity_controls_detect_constant_answers(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    run=packet(tmp_path);out=tmp_path/'judge';calls=[]
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    def send(model,prompt,key,**kwargs):
        calls.append(kwargs)
        data=json.loads(prompt.split('EVIDENCE:\n')[1])
        ids=[n['id'] for n in data['held_out_members']]
        result={'rating':'accurate','coverage':'representative','reason':'The supplied evidence supports this explanation.',
                'evidence_ids':['invented'] if len(calls)==1 else ids[:1]}
        return reply(json.dumps(result))
    r=run_judging(run,out,send=send,sleep=lambda _:None,rpm=100000,max_attempts=2)
    assert r['completed']==5 and len(calls)==6 and r['rejected_attempts']==1
    assert all(c['max_tokens']==512 and c['thinking'] is False and c['decoding']=='json_schema' for c in calls)
    assert r['control_gate_passed'] is False and r['scientific_quality_established'] is False
    assert r['sample_size']==1 and sum(r['sample_ratings'].values())==1
    def forbidden(*a,**kw):raise AssertionError('Cached judgments were sent again')
    assert run_judging(run,out,send=forbidden,rpm=100000,max_attempts=2)['completed']==5


def test_judge_cannot_be_the_label_generator(tmp_path):
    from tkh.label_judging import run_judging
    with pytest.raises(ValueError,match='different'):
        run_judging(packet(tmp_path),tmp_path/'judge',model='nvidia/generator',prepare_only=True)


def test_judge_cli_returns_failure_for_missing_inputs(tmp_path,monkeypatch):
    from tkh.label_judging import main
    monkeypatch.setattr('sys.argv',['check_group_labels.py','--run',str(tmp_path/'missing')])
    assert main()==1


def test_judge_cli_interrupt_preserves_preparation_and_returns_130(tmp_path,monkeypatch):
    from tkh.label_judging import main
    run=packet(tmp_path);out=tmp_path/'judge'
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    monkeypatch.setattr('sys.argv',['check_group_labels.py','--run',str(run),'--out',str(out)])
    def interrupted(*a,**kw):raise KeyboardInterrupt
    monkeypatch.setattr('tkh.label_judging.request_nvidia_async',interrupted)
    assert main()==130 and (out/'protocol.json').exists()


def test_judge_stops_after_persistent_api_failure(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    from tkh.gemini_labels import LabelRequestError
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key');calls=[]
    def unavailable(*a,**kw):
        calls.append(a)
        raise LabelRequestError('NVIDIA HTTP503: unavailable',retryable=True)
    with pytest.raises(RuntimeError,match='503'):
        run_judging(packet(tmp_path),tmp_path/'judge',send=unavailable,sleep=lambda _:None,rpm=100000)
    assert len(calls)==3


def test_cli_fails_for_completed_cached_judgments_when_sensitivity_controls_fail(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging,main
    run=packet(tmp_path);out=tmp_path/'judge'
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    def constant(*a,**kw):
        return reply(json.dumps({'rating':'accurate','coverage':'representative',
            'reason':'The topic is supported.','evidence_ids':[]}))
    run_judging(run,out,send=constant,sleep=lambda _:None,rpm=100000)
    def forbidden(*a,**kw):raise AssertionError('Completed judgments requested again')
    monkeypatch.setattr('tkh.label_judging.request_nvidia_async',forbidden)
    monkeypatch.setattr('sys.argv',['check_group_labels.py','--run',str(run),'--out',str(out)])
    assert main()==1


def test_controls_first_stops_before_sample_and_preserves_failed_controls_on_resume(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    run=packet(tmp_path);out=tmp_path/'judge';calls=[]
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    def constant(model,prompt,key,**kw):
        calls.append(prompt)
        return reply(json.dumps({'rating':'accurate','coverage':'representative',
            'reason':'The topic is supported.','evidence_ids':[]}))
    result=run_judging(run,out,send=constant,sleep=lambda _:None,rpm=100000,controls_first=True)
    assert result['sample_size']==0 and result['control_gate_passed'] is False
    assert result['completed']==len(calls)==7
    assert result['planned_requests']==8
    def forbidden(*a,**kw):raise AssertionError('Failed control gate was rerolled or credentials accessed')
    monkeypatch.setattr('tkh.label_judging.load_api_key',forbidden)
    assert run_judging(run,out,send=forbidden,controls_first=True)['completed']==7


def test_control_gate_checks_coverage_as_well_as_rating(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    def weak_scope_checker(model,prompt,key,**kw):
        data=json.loads(prompt.split('EVIDENCE:\n')[1])
        ratings={'Atomic force prediction':'accurate','Language translation':'wrong',
            'Scientific methods':'vague','Unknown scientific theme':'insufficient_evidence',
            'Computational methods for molecules':'vague',
            'Thermal expansion prediction':'accurate','Thermal and optical property prediction':'accurate'}
        return reply(json.dumps({'rating':ratings[data['label']],'coverage':'representative',
            'reason':'Correct facts but coverage is not checked.','evidence_ids':[]}))
    result=run_judging(packet(tmp_path),tmp_path/'judge',send=weak_scope_checker,
        sleep=lambda _:None,rpm=100000,controls_first=True)
    assert result['control_gate_passed'] is False and result['sample_size']==0
    assert all(c['expected']==c['observed'] for c in result['controls'])
    assert any(c.get('expected_coverage')!=c.get('observed_coverage')
        for c in result['controls'] if 'expected_coverage' in c)


def test_incomplete_controls_do_not_allow_sample_calls(tmp_path,monkeypatch):
    from tkh.label_judging import run_judging
    monkeypatch.setenv('NVIDIA_API_KEY','fake-key')
    decisions={'Atomic force prediction':('accurate','representative'),
        'Language translation':('wrong','too_narrow'),
        'Scientific methods':('vague','too_generic'),
        'Unknown scientific theme':('insufficient_evidence','insufficient_evidence'),
        'Computational methods for molecules':('vague','too_generic'),
        'Thermal expansion prediction':('accurate','too_narrow'),
        'Thermal and optical property prediction':('accurate','representative')}
    def send(model,prompt,key,**kw):
        data=json.loads(prompt.split('EVIDENCE:\n')[1])
        assert not any(n['id']=='n1' for n in data['held_out_members']), 'Sample called before controls completed'
        rating,coverage=decisions[data['label']]
        result=reply(json.dumps({'rating':rating,'coverage':coverage,'reason':'Control explanation.','evidence_ids':[]}))
        if data['label']=='Scientific methods':result['choices'][0]['finish_reason']='length'
        return result
    result=run_judging(packet(tmp_path),tmp_path/'judge',send=send,
        sleep=lambda _:None,rpm=100000,controls_first=True)
    assert result['sample_size']==0 and result['completed']==6
    assert result['control_gate_passed'] is False and result['rejected_attempts']==3
