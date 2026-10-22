"""Offline package rebuilding must use actual saved-state construction."""
import builtins
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def _manifest(assets):
    _write(assets / 'manifest.json', {'sha256': {
        name: hashlib.sha256((assets / name).read_bytes()).hexdigest()
        for name in ('construction_vectors.npz', 'construction_metadata.json')}})


def _fixture(tmp_path):
    data, assets, evidence = (tmp_path / n for n in ('data', 'assets', 'evidence'))
    for path in (data, assets, evidence):
        path.mkdir()
    def node(nid, year=2022, source=1, kind='concept'):
        return {'id': nid, 'type': kind, 'surface_form': f'Name {nid}',
                'first_seen_year': year, 'provenance': {'articles': [source]}}
    def edge(eid, members, year=2022, source=1):
        return {'id': eid, 'members': members, 'relation_type': 'related', 'year': year,
                'provenance': {'article_id': source, 'article_year': year}}
    nodes = [node('n0'), node('n1'), node('n2'), node('n3'),
             node('p1', kind='article'), node('n4', 2023, 2), node('p2', 2023, 2, 'article')]
    _write(data / 'tkh_collection10.json', {'nodes': nodes, 'hyperedges': [
        edge('e0', ['n0', 'n1']), edge('e1', ['n2', 'n3']),
        edge('e2', ['n0', 'n4'], 2023, 2)]})
    # Shuffled asset order catches implementations relying on node row order.
    order = [5, 3, 6, 0, 4, 2, 1]
    x = np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.],
                  [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]], dtype=np.float32)
    np.savez_compressed(assets / 'construction_vectors.npz', vectors=x[order])
    _write(assets / 'construction_metadata.json', {
        'ids': [nodes[i]['id'] for i in order],
        'texts': [nodes[i]['surface_form'] for i in order],
        'encoder': {'model_id': 'test-fixture', 'role': 'matching', 'dimension': 3}})
    _manifest(assets)
    _write(evidence / 'historical_quality.json', {'status': 'historical', 'score': .123})
    return data, assets, evidence


def _build(*args, **kwargs):
    assert importlib.util.find_spec('tkh.submission_pipeline') is not None, (
        'Offline submission construction entry point is missing')
    return importlib.import_module('tkh.submission_pipeline').build_submission(*args, **kwargs)


def test_rebuilds_real_native_checkpoints_and_validates_without_model_or_gold_access(tmp_path, monkeypatch):
    data, assets, evidence = _fixture(tmp_path)
    graph_before = (data / 'tkh_collection10.json').read_bytes()
    original_import = builtins.__import__
    def offline_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'huggingface_hub'}:
            raise AssertionError(f'Offline construction tried to import {name}')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', offline_import)
    out = tmp_path / 'run'
    result = _build(data, assets, evidence, out, years=(2022, 2023), budgets=(2, 4))
    from tkh.divisive_hierarchy import DivisiveState
    from tkh.validation import validate_run
    assert result['status'] == 'completed'
    assert validate_run(out) == []
    assert _read(out / 'validation.json')['status'] == 'passed'
    assert _read(out / 'construction_metrics.json') == result
    first = DivisiveState.load(out / 'snapshots/2022/checkpoint')
    last = DivisiveState.load(out / 'snapshots/2023/checkpoint')
    assert set(first.assignment) == {'n0', 'n1', 'n2', 'n3', 'p1'}
    assert set(last.assignment) == {'n0', 'n1', 'n2', 'n3', 'n4', 'p1', 'p2'}
    assert first.trace['initial_builder_calls'] == 1
    assert last.trace['initial_builder_calls'] == 0
    assert last.trace['resumed_from_saved_state']
    assert any(c['solver'] == 'mtkahypar' for c in first.trace['solver_calls'])
    assert result['snapshots'][0]['snapshot']['nodes'] == 5
    assert result['snapshots'][1]['snapshot']['nodes'] == 7
    assert result['transitions'][0]['shared_old_nodes'] == 5
    assert result['transitions'][0]['levels'][1]['ari_shared_old_nodes'] == 1.
    assert result['historical_evidence']['status'] == 'archived_not_recomputed'
    assert 'historical_quality.json' in result['historical_evidence']['files_sha256']
    assert result['scientific_gaps']['scientific_labels']['status'] == 'not_established'
    np.testing.assert_array_equal(last.vectors[:4], first.vectors[:4])
    np.testing.assert_array_equal(first.vectors[0], [1., 0., 0.])
    assert (data / 'tkh_collection10.json').read_bytes() == graph_before
    with pytest.raises(FileExistsError):
        _build(data, assets, evidence, out, years=(2022, 2023), budgets=(2, 4))


def test_corrupt_assets_are_rejected_before_any_snapshot_is_published(tmp_path):
    data, assets, evidence = _fixture(tmp_path)
    with (assets / 'construction_vectors.npz').open('ab') as stream:
        stream.write(b'changed')
    out = tmp_path / 'run'
    with pytest.raises(ValueError, match='hash|digest|sha256'):
        _build(data, assets, evidence, out, years=(2022,), budgets=(2, 4))
    assert not (out / 'snapshots').exists()


def test_bounded_mode_uses_saved_state_without_inflating_frozen_budgets(tmp_path):
    data, assets, evidence = _fixture(tmp_path)
    out = tmp_path / 'bounded'
    result = _build(data,assets,evidence,out,years=(2022,2023),budgets=(2,4),method='bounded')
    trace = result['snapshots'][-1]['growth_and_cost_trace']
    assert trace['old_edit_limit'] == 0  # floor(10% of four original scientific records)
    assert trace['allowed_old_nodes'] == []
    assert trace['resumed_from_saved_state']
    assert result['protocol']['budgets'] == [2,4]
    assert result['protocol']['method'] == 'bounded'
    assert result['structural_validation'] == 'passed'


def test_balanced_initial_construction_is_explicit_and_keeps_the_same_trusted_caps(tmp_path):
    data,assets,evidence=_fixture(tmp_path)
    out=tmp_path/'balanced'
    result=_build(data,assets,evidence,out,years=(2022,2023),budgets=(2,4),
                  method='bounded',initial_strategy='balanced')
    assert result['protocol']['initial_strategy']=='balanced'
    assert 'balanced_hierarchy.py' in result['protocol']['source_sha256']
    assert result['protocol']['budgets']==[2,4]
    assert result['snapshots'][0]['growth_and_cost_trace']['solver_calls'][0]['stage']=='balanced_bisection'
    assert result['structural_validation']=='passed'


@pytest.mark.parametrize('mutation', ['text', 'missing', 'duplicate', 'nonunit'])
def test_authenticated_but_invalid_frozen_records_cannot_drive_construction(tmp_path, mutation):
    data, assets, evidence = _fixture(tmp_path)
    meta = _read(assets / 'construction_metadata.json')
    with np.load(assets / 'construction_vectors.npz', allow_pickle=False) as record:
        x = record['vectors']
    if mutation == 'text':
        meta['texts'][meta['ids'].index('n0')] = 'Changed name'
    elif mutation == 'missing':
        i = meta['ids'].index('n0')
        meta['ids'].pop(i); meta['texts'].pop(i); x = np.delete(x, i, axis=0)
    elif mutation == 'duplicate':
        meta['ids'][0] = meta['ids'][1]
    else:
        x[0] *= 2.
    _write(assets / 'construction_metadata.json', meta)
    np.savez_compressed(assets / 'construction_vectors.npz', vectors=x)
    _manifest(assets)
    out = tmp_path / 'run'
    with pytest.raises(ValueError, match='text|record|ID|unit|missing|Missing|vector'):
        _build(data, assets, evidence, out, years=(2022, 2023), budgets=(2, 4))
    assert not (out / 'snapshots/2022').exists()


def test_split_mode_records_real_implementation_and_rejects_missing_novelty(tmp_path):
    data,assets,evidence=_fixture(tmp_path)
    with pytest.raises(ValueError,match='novelty'):
        _build(data,assets,evidence,tmp_path/'invalid',enable_split=True)
    result=_build(data,assets,evidence,tmp_path/'split',years=(2022,2023),budgets=(2,4),
        method='bounded',initial_strategy='balanced',enable_novelty=True,
        novelty_policy='conservative',enable_split=True)
    assert result['protocol']['enable_split']
    assert 'topic_split.py' in result['protocol']['source_sha256']
    assert result['scientific_gaps']['automatic_structural_edits']['status']=='implemented_heuristic'
    assert result['structural_validation']=='passed'
