"""Offline recomputation of independent coherence and matched retrieval.

Historical perturbation/blind evidence is retained, explicitly not rerun.
The output distinguishes new deterministic reproduction from new research.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .corpus import load_graph,load_questions
from .hierarchy_evaluation import _pair_coherence
from .hierarchy_retrieval import PairedHierarchySearch,BASELINES
from .submission_pipeline import _sha,_write
from .tag_graph_index import TagGraphIndex


def independent_coherence(groups,nodes,vectors):
    ids=[n['id'] for n in nodes];pool=set(ids);mapping={}
    for group in groups:
        members=set(group['members'])&pool
        if mapping.keys()&members:raise ValueError('Overlapping partition')
        mapping.update(dict.fromkeys(members,group['id']))
    if set(mapping)!=pool:raise ValueError('Incomplete partition')
    x=np.asarray(vectors,dtype=float)
    if x.ndim!=2 or len(x)!=len(ids) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1.,atol=1e-5):
        raise ValueError('Independent vectors must be aligned finite unit vectors')
    labels=np.array([mapping[n] for n in ids]);types=np.array([n['type'] for n in nodes])
    observed,pairs=_pair_coherence(x,labels);rng=np.random.default_rng(0);null=[]
    for _ in range(20):
        random_labels=labels.copy()
        for kind in sorted(set(types)):
            positions=np.flatnonzero(types==kind)
            random_labels[positions]=rng.permutation(labels[positions])
        null.append(_pair_coherence(x,random_labels)[0])
    finite=[v for v in null if v is not None];mean=float(np.mean(finite)) if finite else None
    return {'observed':observed,'within_group_pairs':pairs,'null_values':null,'null_mean':mean,
            'excess':observed-mean if observed is not None and mean is not None else None,
            'null_policy':'20 type-stratified size-preserving permutations; seed0; development evaluator, not percent correct groups.'}


def _partition_digest(hierarchy):
    rows=[{'index':l['index'],'groups':[{'id':g['id'],'parent_id':g['parent_id'],'members':g['members']} for g in l['groups']]} for l in hierarchy['levels']]
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def assess_submission(run,assets,evidence,data):
    run,assets,evidence,data=map(Path,(run,assets,evidence,data))
    if (run/'metrics.json').exists():raise FileExistsError(run/'metrics.json')
    frozen={}
    for directory in (assets,evidence):
        manifest=json.loads((directory/'manifest.json').read_text())
        frozen[str((directory/'manifest.json').resolve())]=_sha(directory/'manifest.json')
        for name,digest in manifest['sha256'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()) or _sha(path)!=digest:raise ValueError('Asset/evidence authentication failed')
            frozen[str(path)]=digest
    for path in Path(__file__).parent.glob('*.py'):frozen[str(path.resolve())]=_sha(path)
    for path in (data/'tkh_collection10.json',data/'questions.csv'):frozen[str(path.resolve())]=_sha(path)
    expected=json.loads((evidence/'expected_partitions.json').read_text())
    metadata=json.loads((assets/'independent_metadata.json').read_text())
    with np.load(assets/'independent_vectors.npz',allow_pickle=False) as a:z=a['vectors']
    if len(set(metadata['ids']))!=len(metadata['ids']) or len(z)!=len(metadata['ids']):raise ValueError('Independent ID alignment failed')
    independent={n:(t,v) for n,t,v in zip(metadata['ids'],metadata['texts'],z,strict=True)}
    construction=json.loads((run/'construction_metrics.json').read_text())
    coherence=[];replay={}
    for snapshot in construction['snapshots']:
        year=snapshot['snapshot']['cutoff'];directory=run/'snapshots'/str(year)
        view=json.loads((directory/'input.json').read_text());hierarchy=json.loads((directory/'hierarchy.json').read_text())
        replay[str(year)]=_partition_digest(hierarchy)==expected[str(year)]
        if not replay[str(year)]:raise ValueError(f'Rebuilt partition differs from historical evidence at {year}; do not attach old results')
        nodes=[n for n in view['nodes'] if n['type'] not in ('article','author')]
        if any(independent[n['id']][0]!=f"{n['type']}: {n['surface_form']}" for n in nodes):raise ValueError('Independent input text differs')
        vectors=np.stack([independent[n['id']][1] for n in nodes])
        coherence.append({'cutoff':year,'levels':{str(level):independent_coherence(hierarchy['levels'][level]['groups'],nodes,vectors) for level in (0,1)}})
    graph=load_graph(data/'tkh_collection10.json');questions=load_questions(data/'questions.csv')
    index=TagGraphIndex(graph)
    roles=json.loads((assets/'retrieval_metadata.json').read_text())
    if roles['documents']!=[{'id':n['id'],'text':f"{n['type']}: {n['surface_form']}"} for n in graph['nodes']]:raise ValueError('Retrieval passages differ')
    if roles['queries']!=[{'id':q['question_id'],'text':q['question']} for q in questions]:raise ValueError('Retrieval questions differ')
    with np.load(assets/'retrieval_vectors.npz',allow_pickle=False) as a:passages,queries=a['passage'],a['query']
    positions={n['id']:i for i,n in enumerate(graph['nodes'])}
    passages=passages[[positions[n] for n in index.ids]]
    for array,count in ((passages,len(index.ids)),(queries,len(questions))):
        if array.shape!=(count,768) or not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array,axis=1),1.,atol=1e-5):raise ValueError('Invalid retrieval vectors')
    hierarchy=json.loads((run/'snapshots'/str(construction['snapshots'][-1]['snapshot']['cutoff'])/'hierarchy.json').read_text())
    search=PairedHierarchySearch(index.ids,index.nodes,index.layers,hierarchy)
    non_author=np.array([index.nodes[n]['type']!='author' for n in index.ids])
    predictions={};routing={}
    # Same primary code's float64 dot product and channel normalization.
    passages=passages.astype(float)
    for q,query in zip(questions,queries,strict=True):
        own=index.bm25_scores(q['question'],context=False)
        lexical=index.bm25_scores(q['question'],context=True);tags=index.tag_scores(q['question'])
        for channel in (own,lexical,tags):channel[~non_author]=0.
        dense=np.zeros(len(index.ids));values=(passages@query.astype(float))[non_author]/.05
        dense[non_author]=np.exp(values-values.max())
        rows=search.predict({'bm25_own':own,'lexical':lexical,'dense':dense,'tags':tags},q['type'])
        for name,row in rows.items():
            predictions.setdefault(name,{})[q['question_id']]=row['ids']
            if 'routing' in row['trace']:
                r=row['trace']['routing']
                routing.setdefault(name,{})[q['question_id']]={k:v for k,v in r.items() if k!='retained_ids'}
        print('Reproduced retrieval',q['question_id'],flush=True)
    _write(run/'predictions.json',predictions);_write(run/'routing_summary.json',routing)
    _write(run/'replay.json',{'partitions_match':replay,'all_branch_checks_passed':True})
    def verify():
        if any(_sha(path)!=digest for path,digest in frozen.items()):raise ValueError('Frozen source/input/evidence changed')
    verify()
    locked={name:_sha(run/name) for name in ('predictions.json','routing_summary.json','replay.json')}
    _write(run/'pre_evaluation_manifest.json',{'inputs_sources_sha256':frozen,'predictions_sha256':locked,
        'gold_access':'All own predictions and checks saved before evaluator/reference access; reproduction of inspected development experiments.'})
    from .evaluation import evaluate
    gold_bytes=(data/'ground_truth.json').read_bytes();gold=json.loads(gold_bytes)
    (run/'reference.json').write_bytes(gold_bytes)
    reports={name:evaluate(graph,questions,prediction,gold,ks=[30]) for name,prediction in predictions.items()}
    old=json.loads((evidence/'retrieval_predictions.json').read_text())
    if predictions!=old:raise ValueError('Reproduced retrieval differs from original; preserve run, do not use historical uncertainty')
    historical=json.loads((evidence/'historical_science.json').read_text())
    old_metrics=json.loads((evidence/'retrieval_metrics.json').read_text())
    blind=json.loads((evidence/'blind/scores.json').read_text())
    metrics={'schema_version':'submission-report-v1','construction':construction,'coherence':coherence,
        'historical_perturbation':historical['perturbation'],'historical_labels':historical['labels'],
        'historical_transitions':historical['transitions'],'temporal_uncertainty':historical['temporal_uncertainty'],
        'blind':blind,'retrieval':{'arms':reports,'baseline_labels':old_metrics['baseline_labels'],
            'paired_question_uncertainty':old_metrics['paired_question_uncertainty']},
        'historical_calibration':json.loads((evidence/'calibration.json').read_text()),
        'answer_nodes':sum(n['type'] in {'method','technique','component','cited_work','dataset'} for n in graph['nodes']),
        'data_quality_errors':sum(1 for row in json.loads((evidence/'data_quality/issues.json').read_text()).get('issues',[]) if row.get('severity')=='error'),
        'reproduction':{'partition_match':replay,'all_280_retrieval_conditions_exact':True,'new_model_inference':False,
            'scientific_evidence_policy':'Current build/coherence/retrieval recomputed; perturbations/blind judgments/calibration retained with hashes, not rerun.'},
        'requirements':{'T1':'implemented_with_data_limitations','T2':'implemented_semantics_empirical',
            'T3':'partial_no_real_temporal_CI_or_adaptive_novelty','T4':'implemented_and_checked',
            'T5':'unmet_explanatory_labels_and_overclaim','T6':'partial_negative_hierarchy_utility',
            'T7':'report_generated'},'full_spec_satisfied':False}
    _write(run/'metrics.json',metrics)
    verify()
    if any(_sha(run/name)!=digest for name,digest in locked.items()):raise ValueError('Frozen prediction changed')
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--assets',type=Path,default=Path('assets'))
    parser.add_argument('--evidence',type=Path,default=Path('evidence'))
    parser.add_argument('--data',type=Path,default=Path('data'))
    args=parser.parse_args();assess_submission(args.run,args.assets,args.evidence,args.data)


if __name__=='__main__':main()
