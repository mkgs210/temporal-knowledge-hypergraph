"""Rebuild the submitted divisive hierarchy offline from frozen own-model vectors.

This entry point performs real initial partitioning and persisted-state updates.
It loads no model, participant source, questions, or ground-truth answers. Saved
scientific evidence is identified separately from newly recomputed construction
and temporal diagnostics.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .divisive_hierarchy import DivisiveState, build_divisive, update_divisive
from .hierarchy import collapse
from .hierarchy_evaluation import compare_temporal, measure
from .temporal_input import POLICY, temporal_view
from .validation import validate_run


SOURCE_NAMES = (
    'submission_pipeline.py', 'divisive_hierarchy.py', 'divisive_solver.py',
    'hierarchy.py', 'hierarchy_evaluation.py', 'temporal_input.py', 'validation.py',
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_assets(assets):
    assets = Path(assets).resolve()
    manifest = json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))
    hashes = manifest.get('sha256')
    required = {'construction_vectors.npz', 'construction_metadata.json'}
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError('Asset sha256 manifest must authenticate construction vectors and metadata')
    frozen = {str(assets / 'manifest.json'): _sha(assets / 'manifest.json')}
    for name, expected in hashes.items():
        path = (assets / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(assets) or not path.is_file():
            raise ValueError(f'Invalid asset manifest path: {name}')
        if not isinstance(expected, str) or _sha(path) != expected:
            raise ValueError(f'Asset sha256 mismatch: {name}')
        frozen[str(path)] = expected
    metadata = json.loads((assets / 'construction_metadata.json').read_text(encoding='utf-8'))
    with np.load(assets / 'construction_vectors.npz', allow_pickle=False) as record:
        vectors = record['vectors']
    ids, texts = metadata.get('ids'), metadata.get('texts')
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(n, str) or not n for n in ids) or len(set(ids)) != len(ids) or
            not isinstance(texts, list) or len(texts) != len(ids) or
            any(not isinstance(t, str) for t in texts) or not isinstance(metadata.get('encoder'), dict)):
        raise ValueError('Frozen record IDs, texts and encoder metadata are invalid')
    if (vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] < 1 or
            not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all() or
            not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5, rtol=0)):
        raise ValueError('Frozen vectors must be finite, floating, unit and aligned to record IDs')
    cache = {nid: (text, vector) for nid, text, vector in zip(ids, texts, vectors, strict=True)}
    return cache, metadata['encoder'], frozen


def _view_vectors(view, cache):
    rows = []
    for node in view['nodes']:
        if node['id'] not in cache:
            raise ValueError(f"Missing frozen record ID: {node['id']}")
        text, vector = cache[node['id']]
        if text != node['surface_form']:
            raise ValueError(f"Frozen input text differs for record ID: {node['id']}")
        rows.append(vector)
    if not rows:
        raise ValueError('First view has no frozen vector records')
    return np.stack(rows)


def _publish(out, state, protocol, input_hash, encoder):
    """Independently validate a staged snapshot before making it public."""
    pending = Path(tempfile.mkdtemp(prefix='.pending-', dir=out))
    year = state.view['cutoff']
    try:
        snapshots = pending / 'snapshots'
        snapshots.mkdir()
        for previous in protocol['years'][:-1]:
            (snapshots / str(previous)).symlink_to(
                (out / 'snapshots' / str(previous)).resolve(), target_is_directory=True)
        directory = snapshots / str(year)
        directory.mkdir()
        hierarchy, label_inputs = state.export()
        artifacts = {
            'input.json': state.view, 'hierarchy.json': hierarchy,
            'label_inputs.json': label_inputs, 'temporal_events.json': state.events,
            'trace.json': state.trace,
            'coarsened_hyperedges.json': {'levels': [
                {'index': level['index'], 'hyperedges': collapse(state.view['hyperedges'], {
                    nid: group['id'] for group in level['groups'] for nid in group['members']})}
                for level in hierarchy['levels']]},
        }
        for name, value in artifacts.items():
            _write(directory / name, value)
        np.save(directory / 'raw_vectors.npy', state.vectors, allow_pickle=False)
        state.save(directory / 'checkpoint')
        _write(directory / 'manifest.json', {
            'cutoff': year, 'raw_input_sha256': input_hash, 'encoder': encoder,
            'time_policy': state.view['policy'], 'limitations': state.view['limitations'],
            'algorithm': state.config,
            'vector_policy': 'Frozen per-record exact ID and own-text; no model initialization.',
            'artifacts_sha256': {str(path.relative_to(directory)): _sha(path)
                                  for path in sorted(directory.rglob('*')) if path.is_file()},
        })
        _write(pending / 'protocol.json', protocol)
        errors = validate_run(pending)
        if errors:
            raise ValueError('Staged submission validation failed: ' + '; '.join(errors))
        _write(directory / 'validation.json', {
            'status': 'passed', 'violations': [], 'validated_years': protocol['years']})
        (out / 'snapshots').mkdir(exist_ok=True)
        directory.rename(out / 'snapshots' / str(year))
        _write(out / 'protocol.json', protocol)
    finally:
        shutil.rmtree(pending)


def build_submission(data, assets, evidence, out, *, years=(2022, 2023, 2024, 2025, 2026),
                     budgets=(12, 50), gamma=.03, method='legacy', max_old_fraction=.1,
                     movement_strength=.05, enable_novelty=False,initial_strategy='legacy',
                     novelty_policy='exploratory',enable_split=False):
    """Recompute the fixed selected method; leave broader assessment to the caller.

    The selected gamma is historical development configuration. This is not a
    new parameter study or a fresh independent scientific acceptance test.
    ``out`` must be new. Required vectors must already exist and authenticate
    exact own surface forms; a cache miss aborts rather than invoking a model.
    """
    data, assets, evidence, out = map(Path, (data, assets, evidence, out))
    if method not in {'legacy','bounded','append','unbounded'}:
        raise ValueError('Unknown construction method')
    if enable_novelty and method != 'bounded':
        raise ValueError('Novelty transactions require bounded mode')
    if enable_split and not enable_novelty:
        raise ValueError('Old topic split requires novelty calibration enabled')
    if initial_strategy not in {'legacy','balanced'} or novelty_policy not in {'exploratory','conservative'}:
        raise ValueError('Unknown initial strategy or novelty policy')
    if out.exists():
        raise FileExistsError(out)
    years, budgets = tuple(years), tuple(budgets)
    if (not years or any(type(y) is not int for y in years) or
            years != tuple(sorted(set(years)))):
        raise ValueError('years must be increasing unique integers')
    if (len(budgets) != 2 or any(type(b) is not int for b in budgets) or
            not 1 <= budgets[0] <= budgets[1] or not math.isfinite(gamma) or gamma < 0):
        raise ValueError('Expected two ordered positive budgets and finite nonnegative gamma')
    graph_path = data / 'tkh_collection10.json'
    graph = json.loads(graph_path.read_text(encoding='utf-8'))
    graph_hash = _sha(graph_path)
    cache, encoder, frozen = _load_assets(assets)
    # Preflight all requested rows without constructing a future-year partition.
    # A complete static cache is not a fitted representation on future inputs.
    for node in graph['nodes']:
        if node['id'] in cache and cache[node['id']][0] != node['surface_form']:
            raise ValueError(f"Frozen input text differs for record ID: {node['id']}")
    first = temporal_view(graph, years[0])
    first_vectors = _view_vectors(first, cache)
    source_root = Path(__file__).parent
    sources = {name: _sha(source_root / name) for name in SOURCE_NAMES}
    if method != 'legacy':
        sources['local_update.py'] = _sha(source_root/'local_update.py')
    if enable_novelty:
        sources['novelty_edits.py'] = _sha(source_root/'novelty_edits.py')
    if enable_split:
        sources['topic_split.py'] = _sha(source_root/'topic_split.py')
    if initial_strategy=='balanced':
        sources['balanced_hierarchy.py'] = _sha(source_root/'balanced_hierarchy.py')
    frozen.update({str((source_root / name).resolve()): digest for name, digest in sources.items()})
    frozen[str(graph_path.resolve())] = graph_hash
    evidence_hashes = ({str(p.relative_to(evidence)): _sha(p)
                        for p in sorted(evidence.rglob('*')) if p.is_file()}
                       if evidence.is_dir() else {})
    protocol = {
        'schema_version': 'submission-construction-protocol-v1',
        'planned_years': list(years), 'years': [], 'budgets': list(budgets),
        'singleton_budget_policy': 'visible_node_count', 'gamma': gamma, 'seed': 20261019,
        'method':method,'max_old_fraction':max_old_fraction,'movement_strength':movement_strength,
        'enable_novelty':enable_novelty,
        'enable_split':enable_split,
        'initial_strategy':initial_strategy,'novelty_policy':novelty_policy,
        'input_sha256': graph_hash, 'time_policy': POLICY, 'encoder': encoder,
        'source_sha256': sources,
        'asset_sha256': json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))['sha256'],
        'selection_policy': 'Fixed historical development choice; no reselection or new tuning.',
        'gold_policy': 'No questions or ground-truth answers read by construction.',
        'execution_policy': 'Offline frozen own-model cache; real initial solve and persisted-state updates.',
    }
    result = {
        'schema_version': 'submission-construction-v1', 'status': 'running',
        'protocol': protocol, 'snapshots': [], 'transitions': [],
        'historical_evidence': {
            'status': 'archived_not_recomputed' if evidence_hashes else 'unavailable',
            'directory': str(evidence), 'files_sha256': evidence_hashes,
            'interpretation': 'Saved scientific measurements are historical; this command does not rerun them.',
        },
        'scientific_gaps': {
            'scientific_labels': {'status': 'not_established',
                                  'reason': 'Extractive member names are not faithful scientific explanations.'},
            'untouched_test': {'status': 'not_available',
                              'reason': 'Corpus, evaluator and fixed gamma were already used in development.'},
            'automatic_structural_edits': {
                'status': 'implemented_heuristic' if enable_novelty else 'disabled',
                'reason': ('Cohesive birth/donor merge and optional genuine old-topic split use frozen first-view '
                           'construction heuristics; scientific novelty and downstream benefit are unverified.'
                           if enable_novelty else 'Persistent topic refinement only; event edits are an explicit experimental option.'),
                'old_topic_split_enabled':enable_split},
            'new_scientific_assessment': {'status': 'not_recomputed',
                                         'reason': 'Only construction, structural checks and temporal diagnostics run here.'},
        },
    }
    out.mkdir(parents=True)
    _write(out / 'construction_metrics.json', result)
    started = time.perf_counter()
    calibration = None
    if enable_novelty:
        from .novelty_edits import calibrate_novelty
        with threadpool_limits(limits=4):
            calibration = calibrate_novelty(first,first_vectors,
                cohesion_quantile=.75 if novelty_policy=='conservative' else .1,
                min_cohort_sources=2 if novelty_policy=='conservative' else 1)
        _write(out/'novelty_calibration.json',calibration)
    try:
        with threadpool_limits(limits=4):
            for index, year in enumerate(years):
                previous = None
                if index:
                    previous = DivisiveState.load(out / 'snapshots' / str(years[index - 1]) / 'checkpoint')
                    view = temporal_view(graph, year)
                    if method == 'legacy':
                        state = update_divisive(previous, view, _view_vectors(view, cache))
                    else:
                        from .local_update import update_local
                        state = update_local(previous,view,_view_vectors(view,cache),mode=method,
                            max_old_fraction=max_old_fraction,movement_strength=movement_strength,
                            novelty_calibration=calibration,allow_old_topic_split=enable_split)
                else:
                    builder = build_divisive
                    if initial_strategy=='balanced':
                        from .balanced_hierarchy import build_balanced
                        builder=build_balanced
                    state = builder(first, first_vectors, budgets=budgets, gamma=gamma, seed=20261019)
                protocol['years'] = list(years[:index + 1])
                _publish(out, state, protocol, graph_hash, encoder)
                if previous is not None:
                    result['transitions'].append(compare_temporal(previous, state))
                result['snapshots'].append(measure(state))
                _write(out / 'construction_metrics.json', result)
                print(f'Constructed and validated {year}: {len(state.view["nodes"])} nodes', flush=True)
        errors = validate_run(out)
        _write(out / 'validation.json', {'status': 'failed' if errors else 'passed', 'violations': errors})
        if errors:
            raise ValueError('Submission validation failed: ' + '; '.join(errors))
        if any(_sha(path) != digest for path, digest in frozen.items()):
            raise ValueError('Frozen source, input or asset sha256 changed during construction')
        result.update(status='completed', structural_validation='passed',
                      elapsed_seconds=time.perf_counter() - started)
        _write(out / 'construction_metrics.json', result)
        return result
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                      elapsed_seconds=time.perf_counter() - started)
        _write(out / 'construction_metrics.json', result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--evidence', type=Path, default=Path('evidence'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--method', choices=('legacy','bounded','append','unbounded'), default='legacy')
    parser.add_argument('--max-old-fraction',type=float,default=.1)
    parser.add_argument('--movement-strength',type=float,default=.05)
    parser.add_argument('--novelty',action='store_true')
    parser.add_argument('--old-topic-split',action='store_true')
    parser.add_argument('--initial-strategy',choices=('legacy','balanced'),default='legacy')
    parser.add_argument('--novelty-policy',choices=('exploratory','conservative'),default='exploratory')
    args = parser.parse_args()
    build_submission(args.data,args.assets,args.evidence,args.out,method=args.method,
        max_old_fraction=args.max_old_fraction,movement_strength=args.movement_strength,
        enable_novelty=args.novelty,initial_strategy=args.initial_strategy,novelty_policy=args.novelty_policy,
        enable_split=args.old_topic_split)


if __name__ == '__main__':
    main()
