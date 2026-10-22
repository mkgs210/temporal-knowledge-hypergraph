"""Fresh structural/temporal evidence and held-out-citation coherence for CPU runs."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .divisive_hierarchy import PROVENANCE_RELATIONS
from .repair_assessment import (audit_frontier, audit_quotient, _checkpoint_persistence,
                                _fresh_perturbations)
from .submission_pipeline import _load_assets, _view_vectors, _write, _sha
from .temporal_uncertainty import snapshot_temporal_uncertainty


def citation_coherence(view, groups, *, n_permutations=20, seed=0):
    """Within-citation endpoint-pair retention, equal source-paper weight.

    Citation incidences have zero construction cost and are not consulted by
    semantic neighbour building or metadata attachment. This is a held-out
    relational proxy, not an estimate of the fraction of scientifically correct
    topics. Cited-work names remain visible to the encoder; only links are held out.
    """
    if 'cites' not in PROVENANCE_RELATIONS:
        raise ValueError('Citation links are no longer excluded from construction')
    ids=[n['id'] for n in view['nodes']]
    membership=[n for g in groups for n in g['members']]
    if Counter(membership)!=Counter(ids):raise ValueError('Expected a complete nonoverlapping partition')
    mapping={n:i for i,g in enumerate(groups) for n in g['members']}
    positions={n:i for i,n in enumerate(ids)}
    labels=np.array([mapping[n] for n in ids]);types=np.array([n['type'] for n in view['nodes']])
    edges=[e for e in view['hyperedges'] if e['relation_type']=='cites' and len(set(e['members']))>=2]
    def score(assignment):
        papers=defaultdict(list)
        for e in edges:
            counts=Counter(assignment[positions[n]] for n in set(e['members']))
            arity=sum(counts.values())
            fraction=sum(c*(c-1) for c in counts.values())/(arity*(arity-1))
            papers[e.get('provenance',{}).get('article_id',e['id'])].append(fraction)
        return float(np.mean([np.mean(v) for v in papers.values()])) if papers else None
    observed=score(labels);null=[];rng=np.random.default_rng(seed)
    if observed is not None:
        for _ in range(n_permutations):
            shuffled=labels.copy()
            for kind in sorted(set(types)):
                rows=np.flatnonzero(types==kind);shuffled[rows]=rng.permutation(labels[rows])
            null.append(score(shuffled))
    mean=float(np.mean(null)) if null else None
    return {'observed':observed,'citation_edges':len(edges),'null_values':null,'null_mean':mean,
        'excess':observed-mean if mean is not None else None,
        'permutation_pvalue':(1+sum(v>=observed for v in null))/(1+len(null)) if null else None,
        'construction_signal_excluded':'cites',
        'null_policy':'Type-stratified membership shuffle preserves every group/type count; fixed seed.',
        'limitations':'Held-out citation topology, not independent full-text semantic truth; names were construction inputs.'}


def retrieval_comparison(baseline, hierarchy, *, n_bootstrap=1999, seed=0):
    a={r['question_id']:r for r in baseline['questions'] if r['type']=='A'}
    b={r['question_id']:r for r in hierarchy['questions'] if r['type']=='A'}
    if set(a)!=set(b) or not a:raise ValueError('Paired Type A questions differ or are empty')
    gains=[];losses=[];rows=[]
    for qid in sorted(a):
        x,y=a[qid]['targets'],b[qid]['targets']
        if [t['label'] for t in x]!=[t['label'] for t in y]:raise ValueError('Paired reference targets differ')
        ah=bh=0
        for old,new in zip(x,y,strict=True):
            before=old['first_rank'] is not None and old['first_rank']<=30
            after=new['first_rank'] is not None and new['first_rank']<=30
            ah+=before;bh+=after
            if after and not before:gains.append({'question_id':qid,'target':old['label']})
            if before and not after:losses.append({'question_id':qid,'target':old['label']})
        rows.append([ah,bh,len(x)])
    values=np.asarray(rows);totals=values.sum(axis=0)
    sampled=values[np.random.default_rng(seed).integers(len(rows),size=(n_bootstrap,len(rows)))].sum(axis=1)
    difference=(sampled[:,1]-sampled[:,0])/sampled[:,2]
    return {'baseline_hits':int(totals[0]),'hierarchy_hits':int(totals[1]),'targets':int(totals[2]),
        'delta_recall_at_30':float((totals[1]-totals[0])/totals[2]),
        'paired_question_bootstrap_95':np.quantile(difference,[.025,.975]).tolist(),
        'gained_targets':gains,'lost_targets':losses,
        'uncertainty_scope':'Conditional paired question bootstrap on inspected development questions; not unseen-test uncertainty.'}


def enrich_cpu(out):
    out=Path(out);metrics=json.loads((out/'metrics.json').read_text())
    if 'science' in metrics:raise FileExistsError('Science measurements already exist')
    root=out/'hierarchy';protocol=metrics['construction']['protocol']
    cache,_,frozen=_load_assets(out/'assets')
    for name,expected in protocol['source_sha256'].items():
        file=Path(__file__).parent/name
        if _sha(file)!=expected:raise ValueError('Construction method changed; run fresh before perturbation tests: '+name)
        frozen[str(file.resolve())]=expected
    for file in Path(__file__).parent.glob('*.py'):frozen[str(file.resolve())]=_sha(file)
    snapshots=[];temporal=[];previous=None
    for year in protocol['years']:
        d=root/'snapshots'/str(year)
        for name in ('input.json','hierarchy.json','coarsened_hyperedges.json','trace.json'):
            frozen[str((d/name).resolve())]=_sha(d/name)
        view=json.loads((d/'input.json').read_text());hierarchy=json.loads((d/'hierarchy.json').read_text())
        quotient=audit_quotient(view,hierarchy,json.loads((d/'coarsened_hyperedges.json').read_text()))
        snapshots.append({'cutoff':year,'quotient':quotient,'citation_coherence':{
            str(level['index']):citation_coherence(view,level['groups'],seed=year+level['index'])
            for level in hierarchy['levels'][:2]}})
        if previous:
            pv,ph,pd=previous
            frontier=audit_frontier(pv,ph,view,hierarchy,json.loads((d/'trace.json').read_text()),
                mode=protocol['method'],max_old_fraction=protocol['max_old_fraction'])
            temporal.append({'frontier':frontier,'persistence':_checkpoint_persistence(pd,d),
                'uncertainty':snapshot_temporal_uncertainty(pv,ph,view,hierarchy,seed=1729)})
        previous=(view,hierarchy,d)
    d=root/'snapshots'/str(protocol['years'][0])
    first=json.loads((d/'input.json').read_text());h=json.loads((d/'hierarchy.json').read_text())
    perturbations=_fresh_perturbations(first,h,_view_vectors(first,cache),protocol,out/'science/perturbations')
    arms=metrics['retrieval']['arms'];paired={}
    for base in ('bm25_native_sources','bm25_minilm_native_sources'):
        paired[base]=retrieval_comparison(arms[base],arms[base+'_hierarchy'])
    paired['coarse_first_2_8']=retrieval_comparison(arms['tree_all'],arms['tree_2_8'])
    paired['coarse_first_8_32']=retrieval_comparison(arms['tree_all'],arms['tree_8_32'])
    metrics['science']={'snapshots':snapshots,'temporal':temporal,'perturbations':perturbations,
                        'paired_retrieval':paired}
    if any(_sha(path)!=expected for path,expected in frozen.items()):raise ValueError('Scientific assessment input/source changed during computation')
    _write(out/'science/protocol.json',{'inputs_sources_sha256':frozen,
        'coherence':'Held-out cites incidences; source-paper weighting; type/size-preserving membership null.',
        'perturbations':'Five initial-view masks; construction source bytes match the baseline builder.',
        'temporal':'Conditional source-component diagnostics and random-correspondence significance; no fabricated process CI.'})
    metrics['not_measured']=[s for s in metrics['not_measured'] if s not in
        {'Independent semantic coherence versus a null model','Fresh edge-deletion robustness'}]
    metrics['not_measured'] += ['Citation coherence is a relational proxy; full-text semantic ground truth is unavailable.',
                              'Perturbation robustness covers the initial view; process-level temporal confidence remains unavailable.']
    _write(out/'metrics.json',metrics)
    return metrics['science']


if __name__=='__main__':
    import argparse
    from threadpoolctl import threadpool_limits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=4):enrich_cpu(args.run)
