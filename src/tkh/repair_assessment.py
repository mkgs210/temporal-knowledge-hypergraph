"""Fresh matched assessment of repaired chronologies, separate from frozen results.

No historical partition/prediction equality is required. No old blind judgement,
label-faithfulness number or edge-deletion result is attached to a changed method.
Questions and evaluators are development data, not an untouched final test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import t
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from .corpus import load_graph, load_questions
from .divisive_hierarchy import DivisiveState, build_divisive
from .hierarchy import collapse
from .hierarchy_retrieval import PairedHierarchySearch, BASELINES
from .submission_assessment import independent_coherence, _partition_digest
from .submission_pipeline import _load_assets, _view_vectors, _sha, _write
from .tag_graph_index import TagGraphIndex
from .temporal_uncertainty import snapshot_temporal_uncertainty
from .validation import validate_run


def _paths(hierarchy):
    levels = {level['index']: level for level in hierarchy['levels']}
    coarse = {g['id']: set(g['members']) for g in levels[0]['groups']}
    paths = {}
    for group in levels[1]['groups']:
        if group['parent_id'] not in coarse or not set(group['members']) <= coarse[group['parent_id']]:
            raise ValueError('Fine membership has an invalid coarse parent')
        for nid in group['members']:
            if nid in paths:
                raise ValueError('Overlapping fine membership')
            paths[nid] = (group['parent_id'], group['id'])
    return paths


def audit_frontier(previous_view, previous_hierarchy, current_view, current_hierarchy,
                   trace, *, mode, max_old_fraction=.1):
    """Recalculate complete old paths; successful trace booleans are not evidence."""
    old, new = _paths(previous_hierarchy), _paths(current_hierarchy)
    old_ids = {n['id'] for n in previous_view['nodes']}
    new_ids = {n['id'] for n in current_view['nodes']}
    if set(old) != old_ids or set(new) != new_ids or not old_ids <= new_ids:
        raise ValueError('Paths do not cover the monotone snapshot inputs')
    old_science = {n['id'] for n in previous_view['nodes'] if n['type'] not in ('article', 'author')}
    new_science = {n['id'] for n in current_view['nodes'] if n['type'] not in ('article', 'author')} - old_ids
    expected_limit = (0 if mode == 'append' else len(old_science) if mode == 'unbounded'
                      else math.floor(len(old_science) * max_old_fraction))
    if mode not in ('append', 'bounded', 'unbounded'):
        raise ValueError('Unknown update control')
    allowed = set(trace['allowed_old_nodes'])
    editable = set(trace['editable_ids'])
    moved = {n for n in old_ids if old[n] != new[n]}
    if trace['old_edit_limit'] != expected_limit or len(allowed) > expected_limit:
        raise ValueError('Declared old frontier exceeds the fixed protocol budget')
    permitted_metadata = (set(trace.get('novelty',{}).get('moved_old_ids',[])) &
                          (old_ids-old_science)) if trace.get('novelty',{}).get('accepted') else set()
    if not allowed <= old_science | permitted_metadata or editable != allowed | new_science:
        raise ValueError('Editable domain differs from declared old frontier plus new science')
    if not moved <= allowed or len(moved) > expected_limit:
        raise ValueError('An old path changed outside the permitted frontier')
    if set(trace['moved_ids']) != moved or trace['realized_old_path_edits'] != len(moved):
        raise ValueError('Trace conceals or miscounts old path edits')
    if trace['initial_builder_calls'] != 0 or not trace.get('resumed_from_saved_state'):
        raise ValueError('Update was not resumed without an initial-builder call')
    for call in trace.get('solver_calls', []):
        if call.get('accepted') and ('global_regularized_delta' in call):
            if (call['global_regularized_delta'] > 1e-10 or not math.isclose(
                    call['global_regularized_delta'], call['contracted_delta'], abs_tol=1e-10)):
                raise ValueError('Recorded local objective decrease/contraction is inconsistent')
    return {'old_nodes': len(old_ids), 'old_scientific_nodes': len(old_science),
            'new_scientific_nodes': len(new_science), 'old_edit_limit': expected_limit,
            'allowed_old_nodes': len(allowed), 'allowed_old_fraction': len(allowed) / len(old_science),
            'actual_moved_ids': sorted(moved), 'actual_old_path_edits': len(moved),
            'budgeted_metadata_edits':len(moved-old_science),
            'outside_frontier_old_nodes': len(old_ids - allowed),
            'outside_frontier_unchanged': True, 'no_initial_rebuild': True,
            'solver_delta_check_scope': 'Consistency of recorded stage deltas; intermediate assignments are not exported for an independent stage-by-stage replay.'}


def audit_quotient(view, hierarchy, artifact):
    """Recompute native endpoint multiplicities at every exported level."""
    saved = {level['index']: level['hyperedges'] for level in artifact['levels']}
    if set(saved) != {level['index'] for level in hierarchy['levels']}:
        raise ValueError('Missing or extra quotient level')
    raw_total = sum(len(e['members']) for e in view['hyperedges'])
    rows = []
    for level in hierarchy['levels']:
        assignment = {n: g['id'] for g in level['groups'] for n in g['members']}
        expected = collapse(view['hyperedges'], assignment)
        if expected != saved[level['index']]:
            raise ValueError('Exported quotient loses endpoints, counts or provenance')
        total = sum(sum(e['counts'].values()) for e in expected)
        if total != raw_total:
            raise ValueError('Quotient does not conserve native endpoints')
        rows.append({'level': level['index'], 'endpoint_total': total,
                     'native_edge_count': len(expected),
                     'internal_edges': sum(len(e['members']) == 1 for e in expected),
                     'multiway_edges': sum(len(e['members']) >= 3 for e in expected),
                     'exact_saved_artifact_match': True})
    return rows


def conditional_seed_interval(values):
    """Descriptive Student interval for mean ARI under the deletion seed design."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Need finite seed-level values')
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return {'values': values.tolist(), 'seeds': len(values), 'mean': mean,
            'low': None if half is None else mean - half, 'high': None if half is None else mean + half,
            'confidence': .95, 'method': 'Student t interval over deletion seeds, without clipping',
            'scope': 'conditional mean over the fixed initial view and edge-deletion randomization; not chronological-process uncertainty',
            'limitations': 'Five seeds give weak distributional evidence; the t approximation is not a coverage guarantee, and construction vectors and corpus stay fixed.'}


def prepare_channel_cache(index, questions, passages, queries):
    """Exactly the historical own channel formulas, computed once for all methods."""
    passages, queries = np.asarray(passages, dtype=float), np.asarray(queries, dtype=float)
    if (passages.ndim != 2 or queries.ndim != 2 or len(passages) != len(index.ids)
            or len(queries) != len(questions) or passages.shape[1] != queries.shape[1]
            or not np.isfinite(passages).all() or not np.isfinite(queries).all()):
        raise ValueError('Retrieval vector dimensions/alignment are invalid')
    non_author = np.array([index.nodes[n]['type'] != 'author' for n in index.ids])
    cache = {}
    for question, query in zip(questions, queries, strict=True):
        own = index.bm25_scores(question['question'], context=False)
        lexical = index.bm25_scores(question['question'], context=True)
        tags = index.tag_scores(question['question'])
        for channel in (own, lexical, tags):
            channel[~non_author] = 0.
        dense = np.zeros(len(index.ids))
        values = (passages @ query)[non_author] / .05
        dense[non_author] = np.exp(values - values.max())
        cache[question['question_id']] = {'bm25_own': own, 'lexical': lexical, 'dense': dense, 'tags': tags}
    return cache


def freeze_predictions(out, predictions, provenance):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'predictions.json').exists() or (out / 'prediction_lock.json').exists():
        raise FileExistsError('Refusing to replace frozen predictions')
    _write(out / 'predictions.json', predictions)
    _write(out / 'prediction_lock.json', {
        'predictions_sha256': _sha(out / 'predictions.json'), 'provenance': provenance,
        'gold_access_policy': 'Predictions serialized and hashed before this assessor opens the reference answers; development corpus and questions were already inspected historically.'})


def score_locked_predictions(out, graph, questions, gold_path):
    out = Path(out)
    lock = json.loads((out / 'prediction_lock.json').read_text())
    if _sha(out / 'predictions.json') != lock['predictions_sha256']:
        raise ValueError('Frozen prediction bytes changed before reference access')
    for path, digest in lock['provenance'].get('frozen_sha256', {}).items():
        if _sha(path) != digest:
            raise ValueError('Frozen assessor input/source changed before reference access')
    predictions = json.loads((out / 'predictions.json').read_text())
    # Deliberately import the gold-matching evaluator after prediction lock checks.
    from .evaluation import evaluate
    gold = json.loads(Path(gold_path).read_text())
    return {method: {arm: evaluate(graph, questions, ranking, gold, ks=[30])
                     for arm, ranking in arms.items()} for method, arms in predictions.items()}


def _checkpoint_persistence(previous, current):
    before, after = DivisiveState.load(previous / 'checkpoint'), DivisiveState.load(current / 'checkpoint')
    positions = {n['id']: i for i, n in enumerate(after.view['nodes'])}
    for i, n in enumerate(before.view['nodes']):
        if not np.array_equal(before.vectors[i], after.vectors[positions[n['id']]]):
            raise ValueError('An old embedding changed in the saved checkpoint')
    old_science = before.config['scientific_ids']
    if after.config['scientific_ids'][:len(old_science)] != old_science:
        raise ValueError('Saved affinity row alignment changed')
    if (before.affinity != after.affinity[:len(old_science), :len(old_science)]).nnz:
        raise ValueError('Saved old-old affinity block changed')
    return {'old_vectors_bitwise_identical': True, 'old_old_affinity_exactly_unchanged': True,
            'old_vector_rows': len(before.view['nodes']), 'old_affinity_rows': len(old_science)}


def _fresh_perturbations(view, hierarchy, vectors, protocol, out):
    builder=build_divisive
    strategy=protocol.get('initial_strategy','legacy')
    if strategy=='balanced':
        from .balanced_hierarchy import build_balanced
        builder=build_balanced
    elif strategy!='legacy':
        raise ValueError('Unknown perturbation construction strategy')
    initial_paths = _paths(hierarchy)
    scientific = {n['id'] for n in view['nodes'] if n['type'] not in ('article', 'author')}
    ids = sorted(initial_paths)
    rows = []
    for seed in range(5):
        removed = sorted(np.random.default_rng(seed).choice(len(view['hyperedges']),
                         size=math.floor(.1 * len(view['hyperedges'])), replace=False).tolist())
        changed = deepcopy(view)
        changed['hyperedges'] = [e for i, e in enumerate(view['hyperedges']) if i not in set(removed)]
        state = builder(changed, vectors, budgets=tuple(protocol['budgets']),
                              gamma=protocol['gamma'], seed=protocol['seed'])
        rebuilt, _ = state.export()
        paths = _paths(rebuilt)
        ari = {}
        for population, population_ids in [('all', ids), ('scientific', [n for n in ids if n in scientific])]:
            ari[population] = {str(level): float(adjusted_rand_score(
                [initial_paths[n][level] for n in population_ids], [paths[n][level] for n in population_ids]))
                for level in (0, 1)}
        row = {'seed': seed, 'removed_edge_ids': [view['hyperedges'][i]['id'] for i in removed],
               'removed_edges': len(removed), 'raw_edges': len(view['hyperedges']),
               'actual_removed_fraction': len(removed) / len(view['hyperedges']), 'ari': ari,
               'partition_digest': _partition_digest(rebuilt)}
        _write(out / f'seed-{seed}' / 'hierarchy.json', rebuilt)
        _write(out / f'seed-{seed}' / 'trace.json', state.trace)
        _write(out / f'seed-{seed}' / 'measurement.json', row)
        rows.append(row)
        print('Fresh initial edge-deletion rebuild', seed, flush=True)
    return {'cutoff': view['cutoff'], 'recomputed': True, 'rows': rows,'initial_strategy':strategy,
            'policy': 'Remove floor(10% of native hyperedges); five seed-specific masks; keep all nodes and vectors; rerun the unchanged initial builder with the same solver seed.',
            'summaries': {population: {str(level): conditional_seed_interval([r['ari'][population][str(level)] for r in rows])
                                     for level in (0, 1)} for population in ('all', 'scientific')},
            'scope': 'Initial-build robustness only. Annual repaired updaters have identical initial construction but their later-snapshot perturbation robustness is not measured here.'}


def assess_repair(runs, out, *, assets=Path('assets'), data=Path('data'), historical=Path('results')):
    runs = {name: Path(path).resolve() for name, path in runs.items()}
    out, assets, data = Path(out), Path(assets), Path(data)
    if out.exists():
        raise FileExistsError(out)
    if not runs:
        raise ValueError('At least one named chronology is required')
    out.mkdir(parents=True)
    cache, _, frozen = _load_assets(assets)
    source_root = Path(__file__).parent
    for name in ('repair_assessment.py', 'temporal_uncertainty.py', 'submission_assessment.py',
                 'submission_pipeline.py', 'divisive_hierarchy.py', 'divisive_solver.py', 'local_update.py',
                 'novelty_edits.py', 'hierarchy.py', 'hierarchy_evaluation.py', 'hierarchy_retrieval.py',
                 'balanced_hierarchy.py',
                 'hybrid_graph_search.py', 'tag_graph_index.py', 'corpus.py', 'evaluation.py', 'validation.py'):
        if (source_root / name).exists():
            frozen[str((source_root / name).resolve())] = _sha(source_root / name)
    for name in ('tkh_collection10.json', 'questions.csv'):
        frozen[str((data / name).resolve())] = _sha(data / name)
    metadata = json.loads((assets / 'independent_metadata.json').read_text())
    with np.load(assets / 'independent_vectors.npz', allow_pickle=False) as record:
        independent_vectors = record['vectors']
    if len(metadata['ids']) != len(set(metadata['ids'])) or len(metadata['ids']) != len(independent_vectors):
        raise ValueError('Independent-vector IDs are not aligned')
    independent = {n: (text, vector) for n, text, vector in zip(metadata['ids'], metadata['texts'], independent_vectors, strict=True)}
    methods, snapshots = {}, {}
    for method, directory in runs.items():
        errors = validate_run(directory)
        if errors:
            raise ValueError(f'{method} fails structural validation: {errors}')
        protocol = json.loads((directory / 'protocol.json').read_text())
        construction = json.loads((directory / 'construction_metrics.json').read_text())
        if construction.get('status') != 'completed':
            raise ValueError(f'{method} construction is unfinished')
        records, coherence, audits, temporal = [], [], [], []
        for year in protocol['years']:
            snapshot = directory / 'snapshots' / str(year)
            for filename in ('input.json', 'hierarchy.json', 'trace.json', 'coarsened_hyperedges.json'):
                frozen[str((snapshot / filename).resolve())] = _sha(snapshot / filename)
            view = json.loads((snapshot / 'input.json').read_text())
            hierarchy = json.loads((snapshot / 'hierarchy.json').read_text())
            trace = json.loads((snapshot / 'trace.json').read_text())
            quotient = audit_quotient(view, hierarchy, json.loads((snapshot / 'coarsened_hyperedges.json').read_text()))
            nodes = [n for n in view['nodes'] if n['type'] not in ('article', 'author')]
            if any(independent[n['id']][0] != f"{n['type']}: {n['surface_form']}" for n in nodes):
                raise ValueError('Independent evaluator text differs from snapshot own text')
            vectors = np.stack([independent[n['id']][1] for n in nodes])
            coherence.append({'cutoff': year, 'levels': {
                str(level): independent_coherence(hierarchy['levels'][level]['groups'], nodes, vectors)
                for level in (0, 1)}})
            check = {'cutoff': year, 'quotient': quotient, 'standalone_validator': 'passed'}
            if records:
                previous_view, previous_hierarchy = records[-1]
                check['frontier'] = audit_frontier(previous_view, previous_hierarchy, view, hierarchy, trace,
                    mode=protocol['method'], max_old_fraction=protocol['max_old_fraction'])
                check['checkpoint_persistence'] = _checkpoint_persistence(directory / 'snapshots' / str(previous_view['cutoff']), snapshot)
                temporal.append(snapshot_temporal_uncertainty(previous_view, previous_hierarchy, view, hierarchy, seed=1729))
            audits.append(check)
            records.append((view, hierarchy))
        methods[method] = {'protocol': protocol, 'coherence': coherence, 'audits': audits,
                           'temporal': temporal, 'construction_elapsed_seconds': construction['elapsed_seconds']}
        snapshots[method] = records
        print('Fresh coherence, invariants and temporal references', method, flush=True)
    first_name = next(iter(runs))
    first_view, first_hierarchy = snapshots[first_name][0]
    first_protocol = methods[first_name]['protocol']
    for method in runs:
        view, hierarchy = snapshots[method][0]
        if (view != first_view or _partition_digest(hierarchy) != _partition_digest(first_hierarchy)
                or any(methods[method]['protocol'].get(key) != first_protocol.get(key)
                       for key in ('budgets', 'gamma', 'seed','initial_strategy'))):
            raise ValueError('Initial partitions/protocol differ; shared perturbation evidence is invalid')
    perturbations = _fresh_perturbations(first_view, first_hierarchy, _view_vectors(first_view, cache),
                                       first_protocol, out / 'perturbations')
    graph = load_graph(data / 'tkh_collection10.json')
    questions = load_questions(data / 'questions.csv')
    index = TagGraphIndex(graph)
    roles = json.loads((assets / 'retrieval_metadata.json').read_text())
    if roles['documents'] != [{'id': n['id'], 'text': f"{n['type']}: {n['surface_form']}"} for n in graph['nodes']]:
        raise ValueError('Retrieval passage strings differ from authenticated cache')
    if roles['queries'] != [{'id': q['question_id'], 'text': q['question']} for q in questions]:
        raise ValueError('Retrieval questions differ from authenticated cache')
    with np.load(assets / 'retrieval_vectors.npz', allow_pickle=False) as record:
        passages, queries = record['passage'], record['query']
    positions = {n['id']: i for i, n in enumerate(graph['nodes'])}
    passages = passages[[positions[n] for n in index.ids]]
    for values in (passages, queries):
        if not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5):
            raise ValueError('Retrieval cache must contain unit vectors')
    channels = prepare_channel_cache(index, questions, passages, queries)
    np.savez_compressed(out / 'channel_cache.npz', **{
        qid + '__' + name: values for qid, row in channels.items() for name, values in row.items()})
    frozen[str((out / 'channel_cache.npz').resolve())] = _sha(out / 'channel_cache.npz')
    predictions, routing = {}, {}
    reference_flat = None
    for method, records in snapshots.items():
        search = PairedHierarchySearch(index.ids, index.nodes, index.layers, records[-1][1])
        predictions[method], routing[method] = {}, {}
        for question in questions:
            qid = question['question_id']
            rows = search.predict(channels[qid], question['type'])
            for name, row in rows.items():
                predictions[method].setdefault(name, {})[qid] = row['ids']
                if 'routing' in row['trace']:
                    route = row['trace']['routing']
                    routing[method].setdefault(name, {})[qid] = {k: v for k, v in route.items() if k != 'retained_ids'}
            print('Matched retrieval', method, qid, flush=True)
        flat = {name: rows for name, rows in predictions[method].items() if name.endswith('::baseline')}
        if reference_flat is not None and flat != reference_flat:
            raise AssertionError('Flat baseline changed across matched hierarchy methods')
        reference_flat = flat
    _write(out / 'routing_summary.json', routing)
    _write(out / 'intrinsic.json', {'methods': methods, 'fresh_initial_perturbations': perturbations})
    freeze_predictions(out, predictions, {'frozen_sha256': frozen, 'method_names': list(runs),
        'channel_policy': 'Common cached float64 BM25 own/context, tags, E5 exponential dot channels; same filters, restart and weights through own PairedHierarchySearch.predict.',
        'baseline_protocol': BASELINES, 'initial_partitions_identical': True,
        'historical_policy': 'No old partition or prediction equality gate. No old blind, perturbation, or faithfulness evidence attached.'})
    reports = score_locked_predictions(out, graph, questions, data / 'ground_truth.json')
    historical_summary = None
    if historical is not None and (Path(historical) / 'metrics.json').is_file():
        path = Path(historical) / 'metrics.json'
        old = json.loads(path.read_text())
        historical_summary = {'status': 'historical_pre_repair_reference_only', 'path': str(path), 'sha256': _sha(path),
                              'coherence': old.get('coherence'), 'retrieval_arms': old.get('retrieval', {}).get('arms'),
                              'policy': 'Historical results are separately labelled; they are not measurements of any repaired method.'}
    metrics = {'schema_version': 'repair-assessment-v1', 'methods': methods,
               'fresh_initial_perturbations': perturbations,
               'retrieval': {'arms': reports, 'primary': 'A micro Recall@30 over all 63 target occurrences; all 14 A questions',
                             'questions_total': len(questions), 'A_questions': sum(q['type'] == 'A' for q in questions),
                             'same_flat_predictions_across_methods': True, 'channels_computed_once': True,
                             'no_new_model_inference': True},
               'historical_baseline': historical_summary,
               'evaluator': {'construction_model': 'intfloat/e5-base-v2',
                             'coherence_model': metadata['encoder']['model_id'],
                             'independence': 'Different embedding family; same member own surface forms. Development evaluator already used, not a fresh held-out acceptance test.',
                             'null': 'Type-stratified membership shuffle preserving each group size and group-type composition; same fixed20 null permutations for comparability.'},
               'gaps': {'labels': 'Standalone label/NLI pilots do not establish final-hierarchy faithfulness; fresh blinded ratings unavailable.',
                        'novelty': 'Leave-source-out novelty trigger is heuristic evidence about node assignment, not ground truth for the birth of scientific themes.',
                        'temporal_ci': 'Shared source overlap leaves too few independent components; no process-level CI claimed. Random-correspondence significance is narrower.',
                        'perturbation': 'Five fresh masks assess common 2022 initial builder; later-snapshot updater perturbation robustness remains unmeasured.',
                        'retrieval': 'Coarse drill-down may lose targets. Results measure development utility with full initial channel computation, not an end-to-end search speedup.'},
               'full_scientific_spec_satisfied': False}
    _write(out / 'metrics.json', metrics)
    if _sha(out / 'predictions.json') != json.loads((out / 'prediction_lock.json').read_text())['predictions_sha256']:
        raise ValueError('Frozen predictions changed during evaluation')
    print('Repair assessment saved', out / 'metrics.json', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', action='append', required=True, help='name=chronology_directory; repeat per method')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=Path('assets'))
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--historical', type=Path, default=Path('results'))
    args = parser.parse_args()
    runs = {}
    for value in args.runs:
        if '=' not in value:
            parser.error('--runs must be name=directory')
        name, directory = value.split('=', 1)
        if not name or name in runs:
            parser.error('Method names must be nonempty and unique')
        runs[name] = directory
    with threadpool_limits(limits=4):
        assess_repair(runs, args.out, assets=args.assets, data=args.data, historical=args.historical)


if __name__ == '__main__':
    main()
