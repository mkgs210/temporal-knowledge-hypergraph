"""Offline component controls on frozen CPU representations; no method selection."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .corpus import load_graph, load_questions
from .cpu_pipeline import _rank
from .cpu_science import citation_coherence, retrieval_comparison
from .hybrid_graph_search import combine_channels, diffuse
from .repair_assessment import (audit_frontier, audit_quotient, _checkpoint_persistence,
    _fresh_perturbations, freeze_predictions, score_locked_predictions, prepare_channel_cache)
from .submission_pipeline import build_submission, _load_assets, _view_vectors, _sha, _write
from .tag_graph_index import TagGraphIndex, IncidenceLayer
from .tree_search import TreeSearch
from .validation import validate_run

CONTROLS = {'no_native_cost': {'gamma': 0., 'method': 'bounded'},
            'unbounded_edits': {'method': 'unbounded', 'movement_strength': 0.}}
ARMS = ('bm25_native_sources_hierarchy', 'bm25_minilm_native_sources_hierarchy',
        'tree_2_8', 'tree_8_32', 'tree_all')


def aligned_vectors(view, questions, metadata, path, encoder):
    """Authenticate exact texts and align cached vectors with the sorted index."""
    if metadata['encoder'] != encoder: raise ValueError('Retrieval encoder changed')
    docs = metadata['documents']; queries = metadata['queries']
    by_id = {d['id']: i for i, d in enumerate(docs)}
    if len(by_id) != len(docs) or set(by_id) != {n['id'] for n in view['nodes']}:
        raise ValueError('Cached document IDs differ')
    nodes = {n['id']: n for n in view['nodes']}
    if any(d['text'] != nodes[d['id']]['surface_form'] for d in docs):
        raise ValueError('Cached document text changed')
    if queries != [{'id': q['question_id'], 'text': q['question']} for q in questions]:
        raise ValueError('Cached query text or ordering changed')
    with np.load(path, allow_pickle=False) as saved:
        passage = saved['passage']; query = saved['query']
    if (passage.ndim != 2 or query.ndim != 2 or len(passage) != len(docs)
            or len(query) != len(questions) or passage.shape[1] != query.shape[1]
            or passage.shape[1] != encoder.get('dimension', passage.shape[1])
            or not np.isfinite(passage).all() or not np.isfinite(query).all()
            or not np.allclose(np.linalg.norm(passage,axis=1),1.,atol=1e-5)
            or not np.allclose(np.linalg.norm(query,axis=1),1.,atol=1e-5)):
        raise ValueError('Cached vector dimensions or normalization changed')
    return passage[[by_id[n] for n in sorted(nodes)]], query


def _predict(index, hierarchy, passages, queries, questions, channels):
    ordinary = [index.layers[n] for n in ('native','sources')
                if index.layers[n].matrix.nnz and np.any(index.layers[n].weights > 0)]
    layers = ordinary or [IncidenceLayer(index.ids, [[n] for n in index.ids], name='identity')]
    groups = [IncidenceLayer(index.ids, [g['members'] for g in level['groups']],
                             name=f'groups_{level["index"]}') for level in hierarchy['levels'][:2]]
    weights = [.85/len(layers)]*len(layers)+[.075,.075]
    tree = TreeSearch(index, passages, hierarchy)
    predictions = {name: {} for name in ARMS}; traces = {}
    for q, query in zip(questions, queries, strict=True):
        row = channels[q['question_id']]
        seeds = {'bm25': combine_channels({'lexical':row['lexical']},{'lexical':1.}),
            'bm25_minilm': combine_channels({k:row[k] for k in ('lexical','dense')},
                                           {'lexical':.5,'dense':.25})}
        for name, seed in seeds.items():
            values, _ = diffuse(seed, layers+groups, layer_weights=weights)
            predictions[name+'_native_sources_hierarchy'][q['question_id']] = _rank(index,values,q['type'])
        for name, broad, fine in [('tree_2_8',2,8),('tree_8_32',8,32),('tree_all',None,None)]:
            result = tree.predict(q['question'],query,broad_width=broad,fine_width=fine,
                                  question_type=q['type'],top_k=100)
            predictions[name][q['question_id']] = result['ids']
            traces.setdefault(name,{})[q['question_id']] = result['trace']
    return predictions, traces


def _measure(root, construction, cache):
    snapshots=[]; temporal=[]; previous=None
    protocol=construction['protocol']
    for year in protocol['years']:
        folder=root/'snapshots'/str(year)
        view=json.loads((folder/'input.json').read_text())
        hierarchy=json.loads((folder/'hierarchy.json').read_text())
        snapshots.append({'cutoff':year,
            'citation_coherence':{str(l['index']):citation_coherence(view,l['groups'],seed=year+l['index'])
                                  for l in hierarchy['levels'][:2]},
            'quotient':audit_quotient(view,hierarchy,json.loads((folder/'coarsened_hyperedges.json').read_text()))})
        if previous:
            pv,ph,pd=previous
            temporal.append({'from_cutoff':pv['cutoff'],'to_cutoff':year,
                'frontier':audit_frontier(pv,ph,view,hierarchy,json.loads((folder/'trace.json').read_text()),
                    mode=protocol['method'],max_old_fraction=protocol['max_old_fraction']),
                'persistence':_checkpoint_persistence(pd,folder)})
        previous=(view,hierarchy,folder)
    first=root/'snapshots'/str(protocol['years'][0])
    view=json.loads((first/'input.json').read_text()); hierarchy=json.loads((first/'hierarchy.json').read_text())
    perturb=_fresh_perturbations(view,hierarchy,_view_vectors(view,cache),protocol,root.parent/'perturbations')
    return snapshots,temporal,perturb


def _row(construction, snapshots, temporal, perturbations, reports):
    return {'configuration':{k:construction['protocol'][k] for k in
                ('gamma','method','movement_strength','max_old_fraction','budgets','years','seed')},
        'snapshots':snapshots,'temporal':temporal,'transitions':construction['transitions'],
        'perturbations':perturbations,'retrieval':{name:reports[name] for name in ARMS},
        'label_faithfulness':{'status':'not_measured',
            'reason':'Component controls have different memberships; main-method label ratings do not transfer.'}}


def _main_digest(metrics):
    main={k:metrics[k] for k in ('encoder','construction','science','retrieval')}
    return hashlib.sha256(json.dumps(main,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _input_hashes(data,run,metrics,*,portable=True):
    protocol=metrics['construction']['protocol']
    paths=[data/'tkh_collection10.json',data/'questions.csv',run/'assets/retrieval_vectors.npz',
           run/'assets/retrieval_metadata.json',run/'assets/construction_vectors.npz',
           run/'assets/construction_metadata.json',run/'assets/manifest.json']
    paths += [p for p in (run/'hierarchy').rglob('*') if p.is_file()]
    source=Path(__file__).parent
    names=set(protocol['source_sha256'])|{'cpu_components.py','cpu_pipeline.py','cpu_science.py',
        'repair_assessment.py','tree_search.py','hybrid_graph_search.py','tag_graph_index.py','evaluation.py',
        'hierarchy_retrieval.py','corpus.py','submission_assessment.py','cpu_embeddings.py','temporal_uncertainty.py'}
    for name,digest in protocol['source_sha256'].items():
        if _sha(source/name)!=digest:raise ValueError('Main construction source changed: '+name)
    paths += [source/name for name in sorted(names)]
    if not portable:return {str(p.resolve()):_sha(p) for p in paths}
    roots=(('data',data.resolve()),('run',run.resolve()),('source',source.resolve()))
    result={}
    for path in paths:
        resolved=path.resolve()
        for prefix,base in roots:
            if resolved.is_relative_to(base):
                result[prefix+'/'+str(resolved.relative_to(base))]=_sha(path)
                break
        else:raise ValueError('Component input is outside declared roots: '+str(path))
    return result


def verify_components(data,run):
    """Read-only release/cache gate; never build, score or rerun a model."""
    data,run=Path(data),Path(run);out=run/'components'
    metrics=json.loads((run/'metrics.json').read_text())
    existing=json.loads((out/'summary.json').read_text())
    if not (out/'prediction_lock.json').is_file():raise ValueError('Component prediction lock is missing')
    if existing['main_metrics_sha256']!=_main_digest(metrics):raise ValueError('Main baseline metrics changed')
    if existing['inputs_sha256']!=_input_hashes(data,run,metrics):raise ValueError('Component inputs or sources changed')
    if existing['reference_sha256']!=_sha(data/'ground_truth.json'):raise ValueError('Component reference answers changed')
    for rel,digest in existing['artifacts_sha256'].items():
        path=out/rel
        if not path.is_file() or _sha(path)!=digest:raise ValueError('Component artifact changed: '+rel)
    if metrics.get('components')!=existing:raise ValueError('Component metrics summary changed')
    return existing


def run_components(data, run):
    """Build two predeclared controls, then assess all predictions under one lock."""
    data,run=Path(data),Path(run); out=run/'components'
    metrics=json.loads((run/'metrics.json').read_text()); protocol=metrics['construction']['protocol']
    if 'science' not in metrics: raise ValueError('Run the main scientific assessment first')
    if protocol['gamma'] <= 0 or protocol['method'] != 'bounded' or protocol['initial_strategy'] != 'balanced':
        raise ValueError('Expected the positive-native bounded balanced main method')
    frozen=_input_hashes(data,run,metrics,portable=False)
    if (out/'summary.json').exists():return verify_components(data,run)
    if out.exists():raise FileExistsError('Incomplete component directory; inspect its recorded failure before retrying')
    out.mkdir(); started=time.perf_counter()
    _write(out/'design.json',{'controls':CONTROLS,'held_fixed':{
        'years':protocol['years'],'budgets':protocol['budgets'],'encoder':metrics['encoder'],
        'seed':protocol['seed'],'retrieval_arms':ARMS},
        'scope':'Exploratory component removals, fixed before execution. No selection or model calls.',
        'limitations':['Zero native cost retains provenance metadata and retrieval relations.',
            'Unbounded edits retain warm starts, fixed group slots, metadata paths and stored affinities.',
            'Changed partitions have no transferred T5 labels or faithfulness ratings.']})
    cache,_,_=_load_assets(run/'assets')
    last=run/'hierarchy/snapshots'/str(protocol['years'][-1])
    view=json.loads((last/'input.json').read_text()); questions=load_questions(data/'questions.csv')
    passages,queries=aligned_vectors(view,questions,json.loads((run/'assets/retrieval_metadata.json').read_text()),
                                     run/'assets/retrieval_vectors.npz',metrics['encoder'])
    index=TagGraphIndex(view); channels=prepare_channel_cache(index,questions,passages,queries)
    predictions={}; measurements={}
    with threadpool_limits(limits=4):
        for name, override in CONTROLS.items():
            config={k:protocol[k] for k in ('years','budgets','gamma','method','movement_strength','max_old_fraction','initial_strategy')}
            config.update(override)
            root=out/name/'hierarchy'
            construction=build_submission(data,run/'assets',out/'no_historical_evidence',root,**config)
            if validate_run(root):raise ValueError('Component structure is invalid: '+name)
            h=json.loads((root/'snapshots'/str(protocol['years'][-1])/'hierarchy.json').read_text())
            pv=json.loads((root/'snapshots'/str(protocol['years'][-1])/'input.json').read_text())
            if pv!=view:raise ValueError('Component snapshot input differs from baseline')
            if name=='unbounded_edits':
                rel=Path('snapshots')/str(protocol['years'][0])/'hierarchy.json'
                if (root/rel).read_bytes()!=(run/'hierarchy'/rel).read_bytes():
                    raise ValueError('Temporal control does not share the exact initial hierarchy')
            predictions[name],traces=_predict(index,h,passages,queries,questions,channels)
            _write(out/name/'routing.json',traces)
            measured=_measure(root,construction,cache)
            measurements[name]=(construction,*measured)
            print('Component control measured:',name,flush=True)
    for p in out.rglob('*'):
        if p.is_file():frozen[str(p.resolve())]=_sha(p)
    freeze_predictions(out,predictions,{'frozen_sha256':frozen,'scope':'Fixed common retrieval; reference access follows this lock.'})
    reference_hash=_sha(data/'ground_truth.json')  # First reference access follows the prediction lock.
    reports=score_locked_predictions(out,load_graph(data/'tkh_collection10.json'),questions,data/'ground_truth.json')
    if _sha(data/'ground_truth.json')!=reference_hash:raise ValueError('Reference answers changed during scoring')
    base=metrics['retrieval']['arms']; science=metrics['science']
    variants={'full':_row(metrics['construction'],science['snapshots'],science['temporal'],science['perturbations'],base)}
    variants['full']['label_faithfulness']={'status':'see_main_method_audit','reason':'Only the main hierarchy receives the T5 audit.'}
    for name,(construction,snapshots,temporal,perturb) in measurements.items():
        variants[name]=_row(construction,snapshots,temporal,perturb,reports[name])
        variants[name]['paired_retrieval_from_full']={arm:retrieval_comparison(base[arm],reports[name][arm]) for arm in ARMS}
    if any(_sha(p)!=digest for p,digest in frozen.items()):raise ValueError('Component input or artifact changed during assessment')
    result={'schema_version':'cpu-components-v1','variants':variants,'elapsed_seconds':time.perf_counter()-started,
        'inputs_sha256':_input_hashes(data,run,metrics),
        'main_metrics_sha256':_main_digest(metrics),'reference_sha256':reference_hash,
        'artifacts_sha256':{str(p.relative_to(out)):_sha(p) for p in out.rglob('*') if p.is_file()},
        'interpretation':'Conditional comparisons on inspected development data; no parameter selection, independent run replicates or transferred label ratings.'}
    _write(out/'summary.json',result);metrics['components']=result;_write(run/'metrics.json',metrics)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--data',type=Path,default=Path('data'));args=parser.parse_args()
    run_components(args.data,args.run)
