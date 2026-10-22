"""Raw-data-only CPU flow; real graph solvers/search, fake neural boundary."""
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from test_cpu_embeddings import factory
from test_submission_pipeline import _fixture


def test_raw_data_only_run_builds_bounded_hierarchy_and_scores_new_predictions(tmp_path, monkeypatch):
    from tkh.cpu_embeddings import CPUEncoder
    from tkh.cpu_pipeline import run_cpu
    from tkh.validation import validate_run
    data, _, _ = _fixture(tmp_path)
    (data/'questions.csv').write_text('question_id;question;type\nq;Name n0?;A\n')
    (data/'ground_truth.json').write_text(json.dumps({'q': {'type': 'A', 'expected_methods': ['Name n0']}}))
    # _fixture uses concept nodes; retrieval answer types deliberately include methods.
    path = data/'tkh_collection10.json'
    graph = json.loads(path.read_text())
    for n in graph['nodes']:
        if n['type'] == 'concept':
            n['type'] = 'method'
    path.write_text(json.dumps(graph))
    calls = []
    out = tmp_path/'cpu'
    read_text = Path.read_text
    def guard_reference(path, *args, **kwargs):
        if path == data/'ground_truth.json':
            assert (out/'prediction_lock.json').is_file()
            assert (out/'predictions.json').is_file()
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guard_reference)
    result = run_cpu(data, out, encoder=CPUEncoder(tmp_path/'cache', model_factory=factory(calls)))
    assert validate_run(out/'hierarchy') == []
    assert result['encoder']['model_id'] == 'sentence-transformers/all-MiniLM-L6-v2'
    assert result['encoder']['device'] == 'cpu'
    assert result['construction']['protocol']['method'] == 'bounded'
    assert result['construction']['protocol']['initial_strategy'] == 'balanced'
    assert result['construction']['snapshots'][1]['growth_and_cost_trace']['initial_builder_calls'] == 0
    arms = result['retrieval']['arms']
    assert arms['bm25']['tracks']['A']['metrics']['30']['hits'] == 1
    assert 'bm25_native_sources_hierarchy' in arms
    assert 'bm25_minilm_native_sources_hierarchy' in arms
    assert 'tree_2_8' in arms
    assert not any('e5' in name.lower() for name in arms)
    assert 'historical_perturbation' not in result and result['full_spec_satisfied'] is False
    assert not any('e5' in str(row).lower() for row in calls)
    assert (out/'prediction_lock.json').is_file()
    assert (out/'assets/construction_vectors.npz').is_file()
    with pytest.raises(FileExistsError):
        run_cpu(data, out, encoder=CPUEncoder(tmp_path/'cache', model_factory=factory(calls)))
    root = Path(__file__).resolve().parents[1]
    report = tmp_path/'article/report.md'
    pdf = report.with_suffix('.pdf')
    command = subprocess.run([sys.executable, str(root/'make_tables.py'), str(out),
        '--report', str(report), '--pdf', str(pdf)], cwd=root,
        env={**os.environ, 'PYTHONPATH': str(root/'src')}, capture_output=True, text=True, timeout=25)
    assert command.returncode == 0, command.stderr
    assert pdf.read_bytes().startswith(b'%PDF')
    assert (report.parent/'figures/cpu_retrieval.png').is_file()
    assert 'all-MiniLM-L6-v2' in report.read_text()
    # Exercise T6 on the same actual native construction, without any model call.
    from tkh.cpu_science import enrich_cpu
    science=enrich_cpu(out)
    assert len(science['perturbations']['rows'])==5
    assert len(science['temporal'])==4
    assert science['paired_retrieval']['bm25_native_sources']['targets']==1
    # Full scientific rendering uses this run's metrics and a three-to-five-page layout.
    from tkh.cpu_report import render_report
    full=json.loads((out/'metrics.json').read_text())
    assert 'Citation' in render_report(full) and '{{' not in render_report(full)
    command=subprocess.run([sys.executable,str(root/'make_tables.py'),str(out),
        '--report',str(report),'--pdf',str(pdf)],cwd=root,
        env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True,timeout=25)
    assert command.returncode==0,command.stderr
    import re
    pages=len(re.findall(rb'/Type\s*/Page\b',pdf.read_bytes()))
    assert 3<=pages<=5, f'Scientific article has {pages} pages'
    # T5 generation is mocked only at the API boundary; publication uses real provenance checks.
    from tkh.gemini_labels import run_labels
    from tkh.cpu_label_audit import prepare_audit,publish_labels
    from test_gemini_labels import reply
    monkeypatch.setenv('GEMINI_API_KEY','fake')
    def label_reply(model,prompt,key):
        if 'BATCH_EVIDENCE:\n' not in prompt:return reply()
        rows=json.loads(prompt.split('BATCH_EVIDENCE:\n')[1])
        return reply(json.dumps([{'group_id':g['group_id'],'label':'Graph model evaluation',
            'gloss':'Graph models connect supplied methods and evaluations.','status':'mixed'} for g in rows]))
    run_labels(out/'hierarchy',out/'labels',provider='gemini',evidence_policy='full_scope',send=label_reply)
    assert prepare_audit(out/'labels')['total_labelled_groups']==40
    from tkh.label_judging import run_judging
    from test_nvidia_labels import reply as nvidia_reply
    monkeypatch.setenv('NVIDIA_API_KEY','fake')
    check=run_judging(out/'labels',out/'labels/audit/api_check',controls_first=True,
        send=lambda *a,**kw:nvidia_reply(json.dumps({'rating':'accurate','coverage':'representative',
            'reason':'Synthetic test response.','evidence_ids':[]})))
    assert check['control_gate_passed'] is False  # An always-positive judge cannot certify faithfulness.
    publish_labels(out)
    assert validate_run(out/'labelled_hierarchy')==[]
    assert publish_labels(out)['generated']==40  # Safe resume without relabelling.
    # Reporting/illustration may canonicalize key order. Re-authenticating the
    # same labels must preserve metric bytes or invalidate a valid report hash.
    from tkh.submission_pipeline import _write
    _write(out/'metrics.json',json.loads((out/'metrics.json').read_text()))
    canonical=(out/'metrics.json').read_bytes()
    publish_labels(out)
    assert (out/'metrics.json').read_bytes()==canonical
    # Re-render after T5 so the report is bound to the latest metrics.
    command=subprocess.run([sys.executable,str(root/'make_tables.py'),str(out),
        '--report',str(report),'--pdf',str(pdf)],cwd=root,
        env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True,timeout=25)
    assert command.returncode==0,command.stderr
    from tkh.cpu_package import package_run
    import shutil,zipfile
    project=tmp_path/'project';project.mkdir()
    shutil.copytree(data,project/'data')
    for name in ('src','tests','schema','prompts','scripts','docs'):
        shutil.copytree(root/name,project/name,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    for name in ('README.md','AI_USAGE.md','Makefile','pyproject.toml','requirements-cpu.lock',
                 'requirements.lock','report.cpu.template.md','make_tables.py','validate.py'):
        shutil.copy2(root/name,project/name)
    import secrets
    private_secret=secrets.token_hex(24)
    (project/'.env').write_text('NVIDIA_API_KEY='+private_secret+'\n')
    (project/'.env.example').write_text('NVIDIA_API_KEY=\n')
    archive=tmp_path/'submission.zip'
    package_run(project,out,archive)
    with zipfile.ZipFile(archive) as z:
        assert 'results/metrics.json' in z.namelist() and 'report.pdf' in z.namelist()
        assert 'docs/REPRODUCIBILITY.md' in z.namelist()
        assert 'docs/REPAIR_RESULTS.md' not in z.namelist()
        assert '.env' not in z.namelist()
        assert private_secret.encode() not in b''.join(z.read(n) for n in z.namelist())
    # Real make all from the extracted ZIP: replace only the neural boundary.
    # Block networking so a regression cannot touch a real provider or download weights.
    cold=tmp_path/'cold';cold.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(cold)
    # Repackaging a saved result must use the files in its new project root.
    old_report=report.with_name('hidden-report.md')
    report.rename(old_report)
    try:
        package_run(cold,cold/'results',tmp_path/'relocated.zip')
    finally:
        old_report.rename(report)
    (cold/'src/sitecustomize.py').write_text('''
import sys, types, socket
import numpy as np
def no_network(*a,**kw):
    raise RuntimeError("Network forbidden in synthetic cold-start test")
socket.create_connection=no_network
class Model:
    def __init__(self,**kw):
        assert kw['device']=='cpu'
    def encode(self,texts,**kw):
        x=np.zeros((len(texts),384),dtype=np.float32)
        for i,text in enumerate(texts):
            x[i,0]=1; x[i,1]=sum(text.encode())%19+1
        return x/np.linalg.norm(x,axis=1,keepdims=True)
sys.modules['sentence_transformers']=types.SimpleNamespace(SentenceTransformer=Model)
''')
    cold_run=subprocess.run(['make','all',f'PYTHON={sys.executable}'],cwd=cold,
        env={**os.environ,'NVIDIA_API_KEY':'','GEMINI_API_KEY':''},
        capture_output=True,text=True,timeout=75)
    assert cold_run.returncode==0,cold_run.stdout+'\n'+cold_run.stderr
    assert (cold/'out/cpu-submission.zip').is_file()
    assert json.loads((cold/'out/cpu/labels/replayed_from.json').read_text())['new_labeller_calls']==0
    # Packaging must authenticate component evidence, not bless modified predictions.
    control=cold/'out/cpu/components/predictions.json'
    control.write_text(control.read_text()+' ')
    rejected=subprocess.run([sys.executable,'-m','tkh.cpu_package','--run','out/cpu',
        '--output','tampered.zip'],cwd=cold,env={**os.environ,'PYTHONPATH':str(cold/'src'),
        'NVIDIA_API_KEY':'','GEMINI_API_KEY':''},capture_output=True,text=True,timeout=25)
    assert rejected.returncode!=0 and 'Component artifact changed' in rejected.stderr
    assert not (cold/'tampered.zip').exists()
    (out/'metrics.json').write_text((out/'metrics.json').read_text()+' ')
    with pytest.raises(ValueError,match='report|metrics'):
        package_run(project,out,tmp_path/'stale.zip')
    damaged=out/'labels/snapshots/2022/hierarchy.json'
    value=json.loads(damaged.read_text());value['levels'][0]['groups'][0]['gloss']='Invented unsupported claim.'
    damaged.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='authenticated|response'):
        publish_labels(out)


def test_cpu_measurement_report_uses_new_model_and_actual_scores():
    from tkh.cpu_report import render_report
    metrics = {'encoder': {'model_id': 'sentence-transformers/all-MiniLM-L6-v2'},
        'construction': {'protocol': {'budgets': [12, 50]}, 'snapshots': [], 'transitions': []},
        'retrieval': {'arms': {'bm25': {'tracks': {'A': {'metrics': {'30':
            {'hits': 2, 'targets_all': 5, 'recall_all_micro': .4}}}}}}},
        'not_measured': ['Independent label faithfulness'], 'full_spec_satisfied': False}
    text = render_report(metrics)
    assert 'all-MiniLM-L6-v2' in text and '2/5' in text and '40.0%' in text
    assert 'Independent label faithfulness' in text and 'E5' not in text


def test_staged_build_and_assessment_only_encode_new_query_text(tmp_path):
    from tkh.cpu_embeddings import CPUEncoder
    from tkh.cpu_pipeline import run_cpu, assess_cpu
    data, _, _ = _fixture(tmp_path)
    (data/'questions.csv').write_text('question_id;question;type\nq;Name n0?;A\n')
    (data/'ground_truth.json').write_text(json.dumps({'q': {'type': 'A', 'expected_methods': ['Name n0']}}))
    cache = tmp_path/'cache'
    calls = []
    out = tmp_path/'cpu'
    run_cpu(data, out, encoder=CPUEncoder(cache, model_factory=factory(calls)),
            years=(2022, 2023), budgets=(2, 4), build_only=True)
    assert not (out/'metrics.json').exists()
    calls.clear()
    assess_cpu(data, out, CPUEncoder(cache, model_factory=factory(calls)))
    assert [r[1] for r in calls if r[0] == 'encode'] == [['Name n0?']]
