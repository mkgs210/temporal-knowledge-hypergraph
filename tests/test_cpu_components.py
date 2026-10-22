"""Component removal must run real solvers and a common locked evaluator."""
import json
from copy import deepcopy
from pathlib import Path
import shutil

import pytest


def test_offline_controls_preserve_baseline_and_lock_before_gold(tmp_path, monkeypatch):
    from tkh import cpu_components
    from tkh.cpu_embeddings import CPUEncoder
    from tkh.cpu_pipeline import run_cpu
    from tkh.cpu_science import enrich_cpu
    from tkh.validation import validate_run
    from test_cpu_embeddings import factory
    from test_submission_pipeline import _fixture
    data, _, _ = _fixture(tmp_path)
    (data/'questions.csv').write_text('question_id;question;type\nq;Name n0?;A\n')
    (data/'ground_truth.json').write_text(json.dumps({'q':{'type':'A','expected_methods':['Name n0']}}))
    graph_path = data/'tkh_collection10.json'
    graph = json.loads(graph_path.read_text())
    for n in graph['nodes']:
        if n['type'] == 'concept': n['type'] = 'method'
    graph_path.write_text(json.dumps(graph))
    out = tmp_path/'run'
    run_cpu(data,out,encoder=CPUEncoder(tmp_path/'cache',model_factory=factory([])),
            years=(2022,2023),budgets=(2,4))
    enrich_cpu(out)
    before = json.loads((out/'metrics.json').read_text())
    initial_bytes = (out/'hierarchy/snapshots/2022/hierarchy.json').read_bytes()
    original_read = Path.read_text
    def check_gold(path,*args,**kwargs):
        if path == data/'ground_truth.json':
            assert (out/'components/prediction_lock.json').is_file()
            assert (out/'components/predictions.json').is_file()
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',check_gold)
    monkeypatch.setattr(CPUEncoder,'encode',lambda *a,**k:pytest.fail('Components must reuse vectors'))
    result = cpu_components.run_components(data,out)
    assert set(result['variants']) == {'full','no_native_cost','unbounded_edits'}
    for key, gamma, mode in [('no_native_cost',0.,'bounded'),('unbounded_edits',.03,'unbounded')]:
        root = out/'components'/key/'hierarchy'
        protocol = json.loads((root/'protocol.json').read_text())
        assert protocol['gamma'] == gamma and protocol['method'] == mode
        assert protocol['budgets'] == [2,4] and validate_run(root) == []
        row = result['variants'][key]
        assert len(row['snapshots']) == 2 and len(row['temporal']) == 1
        assert 'tree_8_32' in row['retrieval'] and 'bm25_native_sources_hierarchy' in row['retrieval']
        assert row['label_faithfulness']['status'] == 'not_measured'
        assert row['perturbations']['summaries']['scientific']['1']['seeds'] == 5
    unbounded = out/'components/unbounded_edits/hierarchy/snapshots/2022/hierarchy.json'
    assert unbounded.read_bytes() == initial_bytes
    full = json.loads((out/'metrics.json').read_text())
    assert all(full[k] == before[k] for k in before)
    assert full['components'] == result
    assert cpu_components.run_components(data,out) == result
    relocated=tmp_path/'relocated'
    shutil.copytree(data,relocated/'data')
    shutil.copytree(out,relocated/'run')
    assert cpu_components.verify_components(relocated/'data',relocated/'run')==result
    reference=data/'ground_truth.json';reference_bytes=reference.read_bytes()
    reference.write_text(json.dumps({'q':{'type':'A','expected_methods':['Changed reference']}}))
    with pytest.raises(ValueError,match='reference|gold'):
        cpu_components.run_components(data,out)
    reference.write_bytes(reference_bytes)
    metric_path=out/'metrics.json';metric_bytes=metric_path.read_bytes()
    altered=deepcopy(full);altered['science']['snapshots'][0]['citation_coherence']['1']['observed']=.314
    metric_path.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='baseline|main|metrics'):
        cpu_components.run_components(data,out)
    metric_path.write_bytes(metric_bytes)
    damaged = out/'components/no_native_cost/hierarchy/snapshots/2023/hierarchy.json'
    damaged.write_text(damaged.read_text()+' ')
    with pytest.raises(ValueError,match='changed|digest|hash|artifact'):
        cpu_components.run_components(data,out)
    with pytest.raises(ValueError,match='artifact'):
        cpu_components.verify_components(data,out)


def test_vector_alignment_is_checked_before_comparison(tmp_path):
    from tkh.cpu_components import aligned_vectors
    import numpy as np
    metadata={'encoder':{'model_id':'x'},'documents':[{'id':'b','text':'B'},{'id':'a','text':'A'}],
              'queries':[{'id':'q','text':'Question'}]}
    view={'nodes':[{'id':'a','surface_form':'A'},{'id':'b','surface_form':'B'}]}
    questions=[{'question_id':'q','question':'Question'}]
    np.savez(tmp_path/'vectors.npz',passage=np.eye(2),query=np.array([[1.,0.]]))
    passage, query = aligned_vectors(view,questions,metadata,tmp_path/'vectors.npz',{'model_id':'x'})
    assert passage.tolist() == [[0.,1.],[1.,0.]] and query.tolist() == [[1.,0.]]
    invalid=deepcopy(metadata);invalid['documents'][0]['text']='Changed'
    with pytest.raises(ValueError,match='text|document'):
        aligned_vectors(view,questions,invalid,tmp_path/'vectors.npz',{'model_id':'x'})
