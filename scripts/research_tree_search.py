"""Fixed-policy coarse-first retrieval experiment; freeze rankings before gold.

Run: PYTHONPATH=src python scripts/research_tree_search.py --out out/repair-tree
The experiment compares structures, never selects a policy using the answers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from tkh.corpus import load_graph,load_questions
from tkh.hybrid_graph_search import combine_channels,diffuse,ANSWER_TYPES
from tkh.submission_pipeline import _load_assets,_write,_sha
from tkh.tag_graph_index import TagGraphIndex
from tkh.tree_search import TreeSearch


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    _,_,frozen=_load_assets(Path('assets'))
    paths={'bounded':Path('out/repair-balanced-bounded-v2/snapshots/2026/hierarchy.json'),
           'novel':Path('out/repair-balanced-novel-v2/snapshots/2026/hierarchy.json'),
           'legacy':Path('results/snapshots/2026/hierarchy.json')}
    for path in [Path(__file__),Path('src/tkh/tree_search.py'),Path('data/questions.csv'),
                 Path('data/tkh_collection10.json'),*paths.values()]:
        frozen[str(path.resolve())]=_sha(path)
    protocol={'widths':[[2,8],[4,16],[8,32],['all','all']],
        'weights':{'lexical':.5,'dense':.25},'channels':['bm25','bm25_e5'],
        'leaf_context':'unchanged TagGraphIndex own + source-title + first8 native neighbors',
        'leaf_weights':'Global BM25S Lucene corpus weights, sliced before scoring; frozen E5 own text',
        'group_context':'Concatenate all scientific-member context documents, no truncation; BM25 length normalization',
        'group_vectors':'Unit normalized mean frozen own E5 passage vectors',
        'graph':'Only native and source incidence layers induced on routed leaf records',
        'gold_policy':'No gold read until predictions.json and lock serialized; development questions were previously inspected',
        'selection':'Fixed widths/weights before answers; no claim of successful utility; no policy selection',
        'sources':['https://github.com/xhluca/bm25s/discussions/69','https://arxiv.org/abs/2510.02539'],
        'frozen_sha256':frozen}
    _write(out/'protocol.json',protocol)
    graph=load_graph('data/tkh_collection10.json');questions=load_questions('data/questions.csv')
    index=TagGraphIndex(graph)
    roles=json.loads(Path('assets/retrieval_metadata.json').read_text())
    with np.load('assets/retrieval_vectors.npz',allow_pickle=False) as record:
        raw_passages=record['passage'];raw_queries=record['query']
    lookup={doc['id']:(doc['text'],v) for doc,v in zip(roles['documents'],raw_passages,strict=True)}
    passages=np.stack([lookup[n][1] for n in index.ids]).astype(float)
    for n in index.ids:
        if lookup[n][0]!=f"{index.nodes[n]['type']}: {index.nodes[n]['surface_form']}":
            raise ValueError('Frozen passage text mismatch')
    query_lookup={q['id']:(q['text'],v) for q,v in zip(roles['queries'],raw_queries,strict=True)}
    predictions={};traces={};all_width_errors=[]
    mask=np.array([index.nodes[n]['type']!='author' for n in index.ids])
    def flat(question,vector):
        lexical=index.bm25_scores(question,context=True);lexical[~mask]=0.
        channels={'lexical':lexical}
        if vector is not None:
            dense=np.zeros(len(index.ids));logits=passages[mask]@vector/.05
            dense[mask]=np.exp(logits-logits.max());channels['dense']=dense
        return diffuse(combine_channels(channels,{'lexical':.5,'dense':.25}),
                       [index.layers[n] for n in ('native','sources')])[0]
    # Full flat reference is a SEPARATE control/audit, never an input to routing.
    reference={};flat_times={}
    for question in questions:
        qid=question['question_id'];text,vector=query_lookup[qid]
        if text!=question['question']:raise ValueError('Frozen question text mismatch')
        for channel,q in [('bm25',None),('bm25_e5',np.asarray(vector,dtype=float))]:
            tick=time.perf_counter();values=flat(text,q)
            flat_times.setdefault(channel,{})[qid]=time.perf_counter()-tick
            reference[channel,qid]=values
    for name,path in paths.items():
        tree=TreeSearch(index,passages,json.loads(path.read_text()))
        predictions[name]={};traces[name]={'build_seconds':tree.build_seconds,'queries':{}}
        for question in questions:
            qid=question['question_id'];text,vector=query_lookup[qid]
            for channel,q in [('bm25',None),('bm25_e5',np.asarray(vector,dtype=float))]:
                for b,f in [(2,8),(4,16),(8,32),(None,None)]:
                    arm=f'{channel}::tree_{b}_{f}' if b is not None else f'{channel}::all'
                    result=tree.predict(text,q,broad_width=b,fine_width=f,question_type=question['type'])
                    predictions[name].setdefault(arm,{})[qid]=result['ids']
                    traces[name]['queries'].setdefault(arm,{})[qid]=result['trace']
                    if b is None:
                        error=float(np.abs(result['probabilities']-reference[channel,qid]).max())
                        all_width_errors.append(error)
                        if error>1e-12:raise AssertionError('All-width tree changed flat control probabilities')
            print('Coarse-first predictions',name,qid,flush=True)
    for path,digest in frozen.items():
        if _sha(path)!=digest:raise ValueError('Frozen input/source changed before prediction lock')
    _write(out/'predictions.json',predictions);_write(out/'traces.json',traces)
    lock={'predictions_sha256':_sha(out/'predictions.json'),'protocol_sha256':_sha(out/'protocol.json'),
          'all_width_max_probability_error':max(all_width_errors),'gold_read_after_this_file':True}
    _write(out/'prediction_lock.json',lock)
    from tkh.evaluation import evaluate
    gold=json.loads(Path('data/ground_truth.json').read_text())
    scores={name:{arm:evaluate(graph,questions,ranking,gold,ks=[30]) for arm,ranking in arms.items()}
            for name,arms in predictions.items()}
    summary={}
    for name,arms in scores.items():
        summary[name]={}
        for arm,score in arms.items():
            rows=traces[name]['queries'][arm]
            a=[rows[q['question_id']] for q in questions if q['type']=='A']
            metric=score['tracks']['A']['metrics']['30']
            summary[name][arm]={'hits':metric['hits'],'targets':metric['targets_all'],'recall':metric['recall_all_micro'],
                'mean_A_leaf_scores':float(np.mean([r['leaf_scores'] for r in a])),
                'mean_A_dense_dots':float(np.mean([r['leaf_dense_dot_products'] for r in a])),
                'mean_A_fine_scores':float(np.mean([r['fine_scores'] for r in a])),
                'mean_A_query_seconds':float(np.mean([r['query_seconds'] for r in a]))}
    if _sha(out/'predictions.json')!=lock['predictions_sha256']:raise ValueError('Rankings changed during scoring')
    _write(out/'metrics.json',{'protocol':protocol,'summary':summary,'scores':scores,
        'flat_reference_query_seconds':flat_times,'all_width_max_probability_error':max(all_width_errors),
        'primary':'A micro Recall@30 over all63 target occurrences, all14 A questions',
        'limitations':['Development evaluation; no untouched test.',
                       'Query timings are one pass per query with caching and no randomized execution order; not a latency significance test.',
                       'Leaf score counts exclude full offline group indexing and prototype aggregation.',
                       'Group text repeats overlapping context and can dilute a small subtheme.',
                       'A centroid does not model multimodal group content; a discarded group cannot recover through its excluded nodes.']})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',default='out/repair-tree')
    args=parser.parse_args()
    with threadpool_limits(limits=4):run(args.out)
