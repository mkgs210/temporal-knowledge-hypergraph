"""Construct and evaluate the bounded hierarchy from raw data on CPU."""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from threadpoolctl import threadpool_limits

from .corpus import load_graph, load_questions
from .cpu_embeddings import CPUEncoder
from .hybrid_graph_search import ANSWER_TYPES, combine_channels, diffuse
from .repair_assessment import freeze_predictions, score_locked_predictions, prepare_channel_cache
from .submission_pipeline import build_submission, _write, _sha
from .tag_graph_index import TagGraphIndex, IncidenceLayer
from .tree_search import TreeSearch
from .validation import validate_run


def _save_vectors(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rank(index, values, kind):
    types = ANSWER_TYPES if kind == 'A' else {'article'}
    rows = [i for i, n in enumerate(index.ids) if index.nodes[n]['type'] in types and values[i] > 0]
    rows.sort(key=lambda i: (-values[i], index.ids[i]))
    return [index.ids[i] for i in rows[:100]]


def assess_cpu(data, out, encoder):
    data, out = Path(data), Path(out)
    if (out/'metrics.json').exists():
        raise FileExistsError(out/'metrics.json')
    hierarchy_run = out/'hierarchy'
    errors = validate_run(hierarchy_run)
    if errors:
        raise ValueError('Hierarchy validation failed: ' + '; '.join(errors))
    construction = json.loads((hierarchy_run/'construction_metrics.json').read_text())
    if construction['protocol']['encoder'] != encoder.config:
        raise ValueError('Construction and retrieval encoder configurations differ')
    year = construction['protocol']['years'][-1]
    snapshot = hierarchy_run/'snapshots'/str(year)
    view = json.loads((snapshot/'input.json').read_text())
    hierarchy = json.loads((snapshot/'hierarchy.json').read_text())
    index = TagGraphIndex(view)
    questions = load_questions(data/'questions.csv')
    # Exactly the same own surface-form representation for construction/search.
    passages = encoder.encode([index.nodes[n]['surface_form'] for n in index.ids])
    queries = encoder.encode([q['question'] for q in questions])
    _save_vectors(out/'assets/retrieval_vectors.npz', passage=passages, query=queries)
    _write(out/'assets/retrieval_metadata.json', {
        'encoder': encoder.config, 'documents': [{'id': n, 'text': index.nodes[n]['surface_form']} for n in index.ids],
        'queries': [{'id': q['question_id'], 'text': q['question']} for q in questions]})
    channels = prepare_channel_cache(index, questions, passages, queries)
    tree = TreeSearch(index, passages, hierarchy)
    ordinary = [index.layers[n] for n in ('native', 'sources')
                if index.layers[n].matrix.nnz and np.any(index.layers[n].weights > 0)]
    identity = IncidenceLayer(index.ids, [[n] for n in index.ids], name='identity')
    group_layers = [IncidenceLayer(index.ids, [g['members'] for g in level['groups']], name=f'groups_{level["index"]}')
                    for level in hierarchy['levels'][:2]]
    predictions, traces = {}, {}
    for q, query in zip(questions, queries, strict=True):
        row = channels[q['question_id']]
        lexical = combine_channels({'lexical': row['lexical']}, {'lexical': 1.})
        hybrid = combine_channels({k: row[k] for k in ('lexical', 'dense')}, {'lexical': .5, 'dense': .25})
        scores = {'bm25': lexical, 'minilm': combine_channels({'dense': row['dense']}, {'dense': 1.}),
                  'bm25_minilm': hybrid}
        for name, seed in [('bm25', lexical), ('bm25_minilm', hybrid)]:
            base = name+'_native_sources'
            scores[base], _ = diffuse(seed, ordinary)
            layers = ordinary or [identity]
            weights = [.85/len(layers)]*len(layers) + [.075, .075]
            scores[base+'_hierarchy'], _ = diffuse(seed, layers+group_layers, layer_weights=weights)
        for name, values in scores.items():
            predictions.setdefault(name, {})[q['question_id']] = _rank(index, values, q['type'])
        for name, broad, fine in [('tree_2_8', 2, 8), ('tree_8_32', 8, 32), ('tree_all', None, None)]:
            result = tree.predict(q['question'], query, broad_width=broad, fine_width=fine,
                                  question_type=q['type'], top_k=100)
            predictions.setdefault(name, {})[q['question_id']] = result['ids']
            traces.setdefault(name, {})[q['question_id']] = result['trace']
        print('CPU retrieval:', q['question_id'], flush=True)
    _write(out/'routing.json', traces)
    frozen = {str(p.resolve()): _sha(p) for p in [
        data/'tkh_collection10.json', data/'questions.csv', snapshot/'hierarchy.json',
        out/'assets/retrieval_vectors.npz', out/'assets/retrieval_metadata.json']}
    frozen.update({str(p.resolve()): _sha(p) for p in Path(__file__).parent.glob('*.py')})
    freeze_predictions(out, {'minilm_cpu': predictions}, {'frozen_sha256': frozen,
        'encoder': encoder.config, 'historical_results_used': False,
        'policy': 'Fresh common BM25/MiniLM channels; matched graph baseline versus group overlay; separate coarse-first routes.'})
    if any(_sha(p) != digest for p, digest in frozen.items()):
        raise ValueError('Prediction inputs changed')
    # The reference is first opened inside this call, after predictions are frozen.
    graph = load_graph(data/'tkh_collection10.json')
    reports = score_locked_predictions(out, graph, questions, data/'ground_truth.json')['minilm_cpu']
    metrics = {'schema_version': 'cpu-run-v1', 'encoder': encoder.config,
        'construction': construction, 'retrieval': {'arms': reports,
            'primary': 'Type A micro Recall@30 over every reference target occurrence',
            'matching': 'Exact normalized surface form or explicit alias; unchanged shared evaluator',
            'questions': len(questions), 'development_data': True},
        'inference': {'new_unique_texts_this_process': encoder.encoded_texts,
                      'cached_unique_text_requests_this_process': encoder.cache_hits},
        'not_measured': ['Independent semantic coherence versus a null model',
            'Fresh edge-deletion robustness', 'Independent label faithfulness and overclaim rate',
            'An untouched final test; these questions have already been inspected'],
        'full_spec_satisfied': False}
    _write(out/'metrics.json', metrics)
    return metrics


def run_cpu(data, out, *, encoder=None, cache_dir='.cache/tkh/embeddings',
            years=(2022, 2023, 2024, 2025, 2026), budgets=(12, 50), build_only=False):
    data, out = Path(data), Path(out)
    if out.exists():
        raise FileExistsError(out)
    graph = load_graph(data/'tkh_collection10.json')
    encoder = encoder or CPUEncoder(cache_dir)
    vectors = encoder.encode([n['surface_form'] for n in graph['nodes']])
    assets = out/'assets'
    _save_vectors(assets/'construction_vectors.npz', vectors=vectors)
    _write(assets/'construction_metadata.json', {'ids': [n['id'] for n in graph['nodes']],
        'texts': [n['surface_form'] for n in graph['nodes']], 'encoder': encoder.config})
    _write(assets/'manifest.json', {'sha256': {name: _sha(assets/name) for name in
        ('construction_vectors.npz', 'construction_metadata.json')}})
    with threadpool_limits(limits=4):
        construction = build_submission(data, assets, out/'no_historical_evidence', out/'hierarchy',
            years=years, budgets=budgets, method='bounded', initial_strategy='balanced')
        if build_only:
            return construction
        return assess_cpu(data, out, encoder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('out/cpu'))
    parser.add_argument('--cache', type=Path, default=Path('.cache/tkh/embeddings'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--build-only', action='store_true')
    mode.add_argument('--assess-only', action='store_true')
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=4):
            if args.assess_only:
                assess_cpu(args.data, args.out, CPUEncoder(args.cache))
            else:
                run_cpu(args.data, args.out, cache_dir=args.cache, build_only=args.build_only)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
