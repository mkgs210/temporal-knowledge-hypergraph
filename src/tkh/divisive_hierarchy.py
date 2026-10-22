"""Saved-state top-down semantic/native hypergraph hierarchy.

The initial two stages and quotient refinement are heuristic. Local updates
retain all original semantic affinities and use explicit fixed-part terminals.
No calibrated birth/split/merge rule is claimed by this first implementation.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy import sparse

from .hierarchy import State, _validated, canonical, collapse
from .temporal_input import delta
from .divisive_solver import SCALE, objective, quotient, boundary_quotient, solve_partition

METADATA_TYPES = {'article', 'author'}
PROVENANCE_RELATIONS = {'presents', 'authored_by', 'cites', 'membership', 'member_of'}


def _input(view, vectors):
    ids, x = _validated(view, vectors)
    if not ids or not np.allclose(np.linalg.norm(x,axis=1), 1., atol=1e-6, rtol=1e-6):
        raise ValueError('Vectors must be finite nonzero unit vectors')
    cutoff = view['cutoff']
    for n in view['nodes']:
        if any(type(n.get(key)) is int and n[key] > cutoff for key in ('observed_at','available_year','first_seen_year')):
            raise ValueError('Future node in current view')
    for e in view['hyperedges']:
        if any(type(y) is int and y > cutoff for y in (e.get('year'),e.get('available_year'),e.get('provenance',{}).get('article_year'))):
            raise ValueError('Future edge in current view')
        if len(set(e['members'])) != len(e['members']):
            raise ValueError('Duplicate hyperedge endpoints')
    science = [n['id'] for n in view['nodes'] if n['type'] not in METADATA_TYPES]
    if not science:
        raise ValueError('No scientific nodes available')
    return ids, x, science


def scientific_edges(view, scientific_ids, gamma=.1, native_papers=None):
    """Equal positive mass per source paper; membership edges have zero cost.

    All restricted original edges are retained, including zero/one-endpoint rows.
    The frozen denominator is the initial number of eligible source papers.
    """
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError('Gamma must be finite and nonnegative')
    science = set(scientific_ids)
    sums = defaultdict(float)
    rows = []
    for e in view['hyperedges']:
        members = [n for n in e['members'] if n in science]
        weight = float(e.get('weight',1.))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Invalid native weight')
        eligible = len(members) >= 2 and e['relation_type'] not in PROVENANCE_RELATIONS and weight > 0
        paper = str(e.get('provenance',{}).get('article_id','missing-source:'+e['id']))
        if eligible:
            sums[paper] += weight
        rows.append((e, members, paper, weight if eligible else 0.))
    denominator = max(len(sums),1) if native_papers is None else native_papers
    if type(denominator) is not int or denominator < 1:
        raise ValueError('Invalid frozen native normalization')
    edges = []
    for e,members,paper,weight in rows:
        edges.append({**deepcopy(e), 'members':members, 'counts':dict.fromkeys(members,1),
                      'original_arity':len(e['members']), 'kind':'native',
                      'weight':gamma * weight / sums[paper] / denominator if weight else 0.})
    return edges, {'native_papers':denominator, 'visible_positive_papers':len(sums),
                   'paper_raw_weights':dict(sums), 'positive_edges':sum(e['weight'] > 0 for e in edges),
                   'scientific_endpoints':sum(len(e['members']) for e in edges),
                   'native_total_weight':math.fsum(e['weight'] for e in edges),
                   'excluded_relations':sorted(PROVENANCE_RELATIONS)}


def _affinities(science, ids, x, previous=None, k=15):
    index = {n:i for i,n in enumerate(ids)}
    sx = x[[index[n] for n in science]]
    n = len(science)
    rows, cols, values = [], [], []
    start = 0 if previous is None else len(previous.config['scientific_ids'])
    # New-to-all exact neighbor queries; old-old entries never recomputed.
    for offset in range(start,n,256):
        sims = np.clip(sx[offset:offset+256] @ sx.T,0.,1.)
        for i, row in enumerate(sims,offset):
            row[i] = -np.inf
            order = np.argsort(-row,kind='stable')[:min(k,n-1)]
            for j in order:
                if row[j] > 0:
                    rows.append(i);cols.append(int(j));values.append(float(row[j]))
    raw = sparse.csr_matrix((values,(rows,cols)),shape=(n,n))
    raw = raw.maximum(raw.T)
    if previous is None:
        norm = float(sparse.triu(raw,k=1).sum()) or 1.
        return raw / norm, norm
    norm = previous.config['normalizations']['semantic']
    old = previous.affinity.tocoo()
    existing = sparse.csr_matrix((old.data,(old.row,old.col)),shape=(n,n))
    return existing + raw / norm, norm


def _semantic_edges(science, affinity):
    upper = sparse.triu(affinity,k=1,format='coo')
    return [{'id':f'semantic:{i}:{j}', 'kind':'semantic', 'members':[science[i],science[j]],
             'weight':float(w)} for i,j,w in zip(upper.row,upper.col,upper.data,strict=True)]


def _restrict(edges, domain):
    domain = set(domain)
    result = []
    for e in edges:
        members = [n for n in e['members'] if n in domain]
        if len(members) >= 2:
            result.append({**e, 'members':members, 'counts':{n:e.get('counts',{}).get(n,1) for n in members}})
    return result


def _metadata(view, x, science_assignment, previous=None, declared_dependencies=None):
    nodes = {n['id']:n for n in view['nodes']}
    index = {n['id']:i for i,n in enumerate(view['nodes'])}
    result = dict(science_assignment)
    article_by_source = {}
    for n in view['nodes']:
        if n['type'] == 'article':
            for source in n.get('provenance',{}).get('articles',[]):
                article_by_source[source] = n['id']
    presented = defaultdict(list)
    for e in view['hyperedges']:
        if e['relation_type'] == 'presents':
            ns = [n for n in e['members'] if n in science_assignment]
            for a in e['members']:
                if nodes[a]['type'] == 'article':
                    presented[a].extend(ns)
    fallback, no_anchor, forced = [], [], []
    science = sorted(science_assignment)
    previous_flags = {} if previous is None else previous.trace.get('metadata_attachment',{})
    def winner(votes):
        counts = Counter(votes)
        return min(counts,key=lambda g:(-counts[g],g))
    for a in sorted(n for n in nodes if nodes[n]['type'] == 'article'):
        if previous is not None and a in previous.assignment and a not in declared_dependencies:
            result[a] = previous.assignment[a]
            if a in previous_flags.get('article_title_fallback',[]):
                fallback.append(a)
            if a in previous_flags.get('cross_topic_vote',[]):
                forced.append(a)
            continue
        votes = [science_assignment[n] for n in sorted(set(presented[a]))]
        if votes:
            result[a] = winner(votes)
            if len(set(votes)) > 1:
                forced.append(a)
        else:
            nearest = min(science,key=lambda n:(-float(x[index[a]] @ x[index[n]]),n))
            result[a] = science_assignment[nearest]
            fallback.append(a)
    for a in sorted(n for n in nodes if nodes[n]['type'] == 'author'):
        if previous is not None and a in previous.assignment and a not in declared_dependencies:
            result[a] = previous.assignment[a]
            if a in previous_flags.get('no_scientific_anchor',[]):
                no_anchor.append(a)
            if a in previous_flags.get('cross_topic_vote',[]):
                forced.append(a)
            continue
        votes = [result[article_by_source[source]] for source in nodes[a].get('provenance',{}).get('articles',[])
                 if source in article_by_source]
        if votes:
            result[a] = winner(votes)
            if len(set(votes)) > 1:
                forced.append(a)
        else:
            result[a] = min(science_assignment.values())
            no_anchor.append(a)
    # Publish the dependent metadata closure. Unaffected old metadata is frozen.
    dependent = set(declared_dependencies or [])
    if previous is not None:
        old = {n['id']:n for n in previous.view['nodes']}
        for n in result:
            if n in science_assignment:
                continue
            if n not in old or old[n] != nodes[n] or result[n] != previous.assignment.get(n):
                dependent.add(n)
            else:
                result[n] = previous.assignment[n]
    return result, {'article_title_fallback':fallback, 'no_scientific_anchor':no_anchor,
                    'cross_topic_vote':forced, 'dependent_metadata_ids':sorted(dependent),
                    'policy':'presented-entity majority; title-nearest science fallback; author visible-source article majority; stable ID ties'}


def _metadata_dependencies(previous, view, editable, groups, new_science):
    """Close declared scientific region over metadata BEFORE any refinement."""
    old={n['id']:n for n in previous.view['nodes']}
    nodes={n['id']:n for n in view['nodes']}
    dependent={n for n in nodes if nodes[n]['type'] in METADATA_TYPES and
               (n not in old or old[n]!=nodes[n] or previous.assignment[n] in groups)}
    presented=defaultdict(set)
    for e in view['hyperedges']:
        if e['relation_type']=='presents':
            science={n for n in e['members'] if nodes[n]['type'] not in METADATA_TYPES}
            for n in e['members']:
                if nodes[n]['type']=='article':
                    presented[n].update(science)
    article_sources=set()
    for n,record in nodes.items():
        if record['type']=='article':
            if presented[n]&editable or (not presented[n] and new_science):
                dependent.add(n)
            if n in dependent:
                article_sources.update(record.get('provenance',{}).get('articles',[]))
    for n,record in nodes.items():
        if record['type']=='author' and article_sources&set(record.get('provenance',{}).get('articles',[])):
            dependent.add(n)
    return dependent


def _events(previous, view, assignment, parents):
    before = [{},{},{}] if previous is None else [
        {n:previous.parents[g] for n,g in previous.assignment.items()}, previous.assignment,
        {n:'node:'+n for n in previous.assignment}]
    after = [{n:parents[g] for n,g in assignment.items()},assignment,{n['id']:'node:'+n['id'] for n in view['nodes']}]
    events = []
    for level,(old,new) in enumerate(zip(before,after,strict=True)):
        for group in sorted(set(old.values()) | set(new.values())):
            a = {n for n,g in old.items() if g == group}
            b = {n for n,g in new.items() if g == group}
            if a == b:
                continue
            kind = 'birth' if not a else 'death' if not b else 'growth' if a <= b else 'move'
            events.append({'kind':kind,'level':level,'group_id':group,'node_ids':sorted(b-a),
                           'removed_node_ids':sorted(a-b),'previous_group_ids':[group] if a else []})
    return events


@dataclass
class DivisiveState(State):
    affinity: sparse.csr_matrix | None = None

    def to_dict(self):
        data = super().to_dict()
        a = self.affinity.tocsr()
        data['affinity_sha256'] = hashlib.sha256(a.data.tobytes()+a.indices.tobytes()+a.indptr.tobytes()).hexdigest()
        return data

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.divisive-checkpoint-',dir=path.parent))
        try:
            np.savez_compressed(staging/'vectors.npz',vectors=self.vectors)
            sparse.save_npz(staging/'affinity.npz',self.affinity)
            (staging/'state.json').write_text(canonical(self.to_dict())+'\n',encoding='utf-8')
            staging.rename(path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @classmethod
    def load(cls,path):
        path=Path(path)
        data=json.loads((path/'state.json').read_text(encoding='utf-8'))
        x=np.load(path/'vectors.npz',allow_pickle=False)['vectors']
        if hashlib.sha256(x.tobytes()).hexdigest() != data.pop('vectors_sha256'):
            raise ValueError('Corrupt checkpoint vectors')
        a=sparse.load_npz(path/'affinity.npz').tocsr()
        digest=hashlib.sha256(a.data.tobytes()+a.indices.tobytes()+a.indptr.tobytes()).hexdigest()
        if digest != data.pop('affinity_sha256'):
            raise ValueError('Corrupt checkpoint semantic affinity')
        state=cls(vectors=x,affinity=a,**data)
        ids,_,science=_input(state.view,x)
        _check(state,ids,science)
        if (a.shape != (len(science),len(science)) or (a != a.T).nnz or
                not np.isfinite(a.data).all() or np.any(a.data<0) or np.any(a.diagonal()!=0)):
            raise ValueError('Invalid checkpoint affinity')
        state._loaded_from = str(path.resolve())
        return state


def _check(state, ids, science):
    if set(state.assignment) != set(ids) or set(state.assignment.values()) != set(state.parents):
        raise ValueError('Assignment coverage or fine-group parent mismatch')
    if len(state.parents)>state.config['budgets'][1] or len(set(state.parents.values()))>state.config['budgets'][0]:
        raise ValueError('Hierarchy exceeds frozen budgets')
    if set(state.config['scientific_ids']) != set(science):
        raise ValueError('Scientific ID manifest mismatch')
    if set(state.assignment[n] for n in science) != set(state.parents):
        raise ValueError('Metadata-only scientific group')


def _coarse_refine(science, edges, assignment, parents, seed, editable=None):
    q=quotient(edges,assignment)
    groups=sorted(parents)
    broad=sorted(set(parents.values()))
    codes={g:i for i,g in enumerate(broad)}
    initial={g:codes[parents[g]] for g in groups}
    masses=dict(Counter(assignment.values()))
    fixed={} if editable is None else {g:initial[g] for g in groups if g not in editable}
    result,trace=solve_partition(groups,q,len(broad),initial=initial,fixed=fixed,masses=masses,seed=seed)
    new={g:broad[result[g]] for g in groups}
    original_before=objective(edges,{n:parents[assignment[n]] for n in science})
    original_after=objective(edges,{n:new[assignment[n]] for n in science})
    if not math.isclose(original_after,objective(q,result),abs_tol=1e-10):
        raise AssertionError('Quotient failed original-node objective identity')
    trace.update(quotient_consumed=True, original_before=original_before, original_after=original_after,
                 quotient_edges=len(q), endpoint_counts=sum(sum(e['counts'].values()) for e in q))
    return new,trace,q


def _trace_finish(state, science, edges, native, q):
    coarse={n:state.parents[state.assignment[n]] for n in science}
    scientific_assignment={n:state.assignment[n] for n in science}
    fine_cost=objective(edges,scientific_assignment)
    broad_cost=objective(edges,coarse)
    mapped=objective(q,state.parents)
    state.trace.update(objectives={'broad':broad_cost,'fine':fine_cost},
        numeric_checks={'quotient_objective_equal':math.isclose(mapped,broad_cost,abs_tol=1e-10),
                        'semantic_total':float(sparse.triu(state.affinity,k=1).sum()),
                        'native_total':math.fsum(e['weight'] for e in native)},
        endpoint_accounting={'full':sum(len(e['members']) for e in state.view['hyperedges']),
                             'scientific':sum(len(e['members']) for e in native),
                             'full_quotient':sum(sum(e['counts'].values()) for e in collapse(state.view['hyperedges'],state.assignment)),
                             'scientific_quotient':sum(sum(e.get('counts',{}).values()) for e in quotient(native,scientific_assignment)),
                             'semantic_edges':state.affinity.nnz//2},
        scientific_quotient=quotient(native,scientific_assignment))
    _check(state,[n['id'] for n in state.view['nodes']],science)


def build_divisive(view, vectors, budgets=(12,50), gamma=.1, seed=20261019):
    ids,x,science=_input(view,vectors)
    if len(budgets)!=2 or any(type(b)is not int for b in budgets) or not 1<=budgets[0]<=budgets[1]:
        raise ValueError('Expected ordered positive integer budgets')
    if type(seed)is not int:
        raise ValueError('Seed must be an integer')
    affinity,semantic_norm=_affinities(science,ids,x)
    native,normal=scientific_edges(view,science,gamma)
    edges=_semantic_edges(science,affinity)+native
    top=min(budgets[0],len(science)); fine=min(budgets[1],len(science))
    broad,broad_trace=solve_partition(science,edges,top,seed=seed)
    parents={f'L1-div{i:06d}':f'L0-div{i:06d}' for i in range(top)}
    assignment={n:f'L1-div{broad[n]:06d}' for n in science}
    index={n:i for i,n in enumerate(ids)}
    calls=[broad_trace];lineage=[];next_id=top
    while len(parents)<fine:
        members={g:[n for n in science if assignment[n]==g] for g in parents}
        candidates=[g for g,ns in members.items() if len(ns)>1]
        def dispersion(group):
            xx=x[[index[n] for n in members[group]]]
            return float(len(xx)-np.dot(xx.sum(axis=0),xx.sum(axis=0))/len(xx))
        group=min(candidates,key=lambda g:(-dispersion(g),g))
        ns=members[group]
        split,trace=solve_partition(ns,_restrict(edges,ns),2,seed=seed+next_id)
        trace.update(stage='within_parent_split',group=group)
        calls.append(trace)
        counts=Counter(split.values())
        retaining=min(counts,key=lambda g:(-counts[g],min(n for n in ns if split[n]==g)))
        born=f'L1-div{next_id:06d}';next_id+=1
        parents[born]=parents[group]
        for n in ns:
            if split[n]!=retaining:
                assignment[n]=born
        lineage.append({'kind':'split','level':1,'retained_id':group,'new_id':born,
                        'parent_id':parents[group], 'criterion':'largest construction-vector dispersion; native cut'})
    parents,coarse_trace,q=_coarse_refine(science,edges,assignment,parents,seed)
    full,metadata=_metadata(view,x,assignment)
    config={'method':'divisive-native-km1-v1','budgets':list(budgets),'gamma':gamma,'seed':seed,
            'scientific_ids':science, 'knn':15,'epsilon':.5,'integer_scale':SCALE,
            'normalizations':{'semantic':semantic_norm,'native_papers':normal['native_papers']},
            'native_weight_policy':'equal total mass per eligible paper; frozen initial paper-count divisor',
            'semantic_affinity':'positive cosine; symmetric maximum kNN15; frozen initial upper-triangle total; append new-node queries only',
            'feature_policy':'own unit sentence vectors; unchanged old rows',
            'refinement_scope':'whole-touched-subtopics with fixed global-part terminals',
            'automatic_edit_triggers':'not_calibrated; proposals rejected; no birth/split/merge during updates',
            'label_status':'extractive representative names only; faithful scientific labels pending'}
    trace={'initial_builder_calls':1,'new_nodes':ids,'new_edges':[e['id'] for e in view['hyperedges']],
           'effective_budgets':[top,fine],'budget_shortfall':top<budgets[0] or fine<budgets[1],
           'editable_ids':science,'moved_ids':[],'old_node_moves':[], 'solver_calls':calls,
           'coarse_refinement':coarse_trace,'coarsened_edges_used':len(q),
           'metadata_attachment':metadata,'lineage':lineage,'native_normalization':normal,
           'limitations':['Initial staged solves are heuristic; floating optimization is approximate after rounding.',
                          'Automatic update birth/split/merge thresholds remain uncalibrated and are not executed.']}
    state=DivisiveState(deepcopy(view),x.copy(),full,parents,config,_events(None,view,full,parents),trace,next_id,affinity)
    _trace_finish(state,science,edges,native,q)
    return state


def update_divisive(previous, view, vectors):
    ids,x,current_science=_input(view,vectors)
    change=delta(previous.view,view)
    old={n['id']:n for n in previous.view['nodes']}
    nodes={n['id']:n for n in view['nodes']}
    for n,record in old.items():
        if any(nodes[n].get(k)!=v for k,v in record.items() if k!='provenance'):
            raise ValueError('Existing node changed')
        a=record.get('provenance',{});b=nodes[n].get('provenance',{})
        if not set(a.get('articles',[]))<=set(b.get('articles',[])) or any(b.get(k)!=v for k,v in a.items() if k!='articles'):
            raise ValueError('Existing node provenance removed or rewritten')
    index={n:i for i,n in enumerate(ids)}
    if x.shape[1]!=previous.vectors.shape[1] or any(not np.array_equal(x[index[n['id']]],previous.vectors[i]) for i,n in enumerate(previous.view['nodes'])):
        raise ValueError('Existing embedding changed')
    if view==previous.view:
        state=deepcopy(previous)
        state.trace={**state.trace,'initial_builder_calls':0,'new_nodes':[],'new_edges':[],
                     'editable_ids':[],'moved_ids':[],'old_node_moves':[],
                     'dependent_metadata_ids':[], 'noop':True,
                     'resumed_from_saved_state':hasattr(previous,'_loaded_from')}
        return state
    cfg=deepcopy(previous.config)
    old_science=list(cfg['scientific_ids'])
    science=old_science+[n for n in current_science if n not in set(old_science)]
    new=set(science)-set(old_science)
    cfg['scientific_ids']=science
    affinity,_=_affinities(science,ids,x,previous,k=cfg['knn'])
    native,normal=scientific_edges(view,science,cfg['gamma'],cfg['normalizations']['native_papers'])
    edges=_semantic_edges(science,affinity)+native
    old_native,old_normal=scientific_edges(previous.view,old_science,cfg['gamma'],cfg['normalizations']['native_papers'])
    old_weights={e['id']:e['weight'] for e in old_native}
    reweighted=[e['id'] for e in native if e['id'] in old_weights and e['weight']!=old_weights[e['id']]]
    fresh={e['id'] for e in change['hyperedges_added']} | set(reweighted)
    touched={n for e in native if e['id'] in fresh for n in e['members'] if n in old}
    for i in range(len(old_science),len(science)):
        touched.update(science[j] for j in affinity.getrow(i).indices if j<len(old_science))
    groups={previous.assignment[n] for n in touched}
    editable={n for n in old_science if previous.assignment[n] in groups} | new
    assignment={n:previous.assignment[n] for n in old_science}
    parents=dict(previous.parents)
    # New-node starting placement uses individual old science, never topic means.
    sx=previous.vectors[[i for i,n in enumerate(previous.view['nodes']) if n['id'] in set(old_science)]]
    aligned_old=[n['id'] for n in previous.view['nodes'] if n['id'] in set(old_science)]
    novelty=[]
    for n in sorted(new):
        similarities=sx @ x[index[n]]
        j=int(np.argmax(similarities))
        assignment[n]=previous.assignment[aligned_old[j]]
        groups.add(assignment[n])
        # No calibrated threshold: retain proposal evidence without claiming birth.
        novelty.append({'node_id':n,'max_old_cosine':float(similarities[j]),'status':'unresolved',
                        'reason':'automatic novelty/edit triggers not calibrated; existing legal path retained'})
    # Close over complete newly touched subtopics before any optimization.
    editable.update(n for n in old_science if previous.assignment[n] in groups)
    declared=sorted(editable)
    metadata_dependencies=_metadata_dependencies(previous,view,editable,groups,new)
    calls=[];boundary_deltas=[]
    for parent in sorted(set(parents.values())):
        domain=[n for n in science if parents[assignment[n]]==parent]
        active=set(domain)&editable
        if not active:
            continue
        labels=sorted({assignment[n] for n in domain})
        codes={g:i for i,g in enumerate(labels)}
        start={n:codes[assignment[n]] for n in domain}
        local_edges=_restrict(edges,domain)
        contracted,fixed=boundary_quotient(local_edges,active,start)
        local_ids=sorted(active)+sorted(fixed,key=str)
        initial={n:start[n] for n in active}|fixed
        masses={n:1 for n in active}
        for terminal,part in fixed.items():
            masses[terminal]=sum(start[n]==part for n in domain if n not in active)
        before=objective(edges,assignment)
        result,trace=solve_partition(local_ids,contracted,len(labels),initial=initial,fixed=fixed,
                                     masses=masses,seed=cfg['seed'])
        proposed=dict(assignment)
        for n in active:
            proposed[n]=labels[result[n]]
        after=objective(edges,proposed)
        reduced=objective(contracted,result)-objective(contracted,initial)
        if not math.isclose(after-before,reduced,abs_tol=1e-10):
            raise AssertionError('Fixed terminal quotient failed original objective delta')
        if after>before+1e-12:
            raise AssertionError('Floating objective increased')
        assignment=proposed
        trace.update(parent=parent,global_delta=after-before,terminal_delta=reduced)
        calls.append(trace)
        boundary_deltas.append({'parent':parent,'original_delta':after-before,'contracted_delta':reduced})
    parents,coarse_trace,q=_coarse_refine(science,edges,assignment,parents,cfg['seed'],editable=groups)
    full,metadata=_metadata(view,x,assignment,previous,metadata_dependencies)
    moved=[n for n in old_science if full[n]!=previous.assignment[n] or parents[full[n]]!=previous.parents[previous.assignment[n]]]
    untouched=set(old_science)-editable
    invariant=all(full[n]==previous.assignment[n] and parents[full[n]]==previous.parents[previous.assignment[n]] for n in untouched)
    if not invariant:
        raise AssertionError('Outside-region assignment changed')
    trace={'initial_builder_calls':0,'new_nodes':[n['id'] for n in change['nodes_added']],
           'new_edges':[e['id'] for e in change['hyperedges_added']],
           'editable_ids':declared,'editable_groups':sorted(groups),'allowed_old_nodes':sorted(editable-new),
           'region_declared_before_optimization':True,
           'moved_ids':sorted(moved),'old_node_moves':sorted(n for n in moved if full[n]!=previous.assignment[n]),
           'coarse_group_moves':sorted(g for g in parents if parents[g]!=previous.parents[g]),
           'metadata_attachment':metadata,'dependent_metadata_ids':metadata['dependent_metadata_ids'],
           'solver_calls':calls,'coarse_refinement':coarse_trace,
           'boundary_checks':{'outside_region_unchanged':invariant,'exact_deltas':boundary_deltas},
           'coarsened_edges_used':len(q),'native_normalization':normal,'reweighted_native_edges':reweighted,
           'unresolved_novelty':novelty,'edit_proposals':[{'operation':'birth/split/merge','accepted':False,
           'reason':'2022 leave-one-paper-out edit thresholds are not calibrated; hard budget preserved'}] if new else [],
           'old_old_affinities_unchanged':True,
           'resumed_from_saved_state':hasattr(previous,'_loaded_from'),
           'previous_checkpoint':getattr(previous,'_loaded_from',None),
           'global_input_scan':{'nodes':len(ids),'hyperedges':len(view['hyperedges']),
                                'reason':'input validation and exact quotient reconstruction scan current snapshot'},
           'scanned_scientific_nodes':len(science),'scanned_semantic_edges':affinity.nnz//2,
           'limitations':deepcopy(previous.trace.get('limitations',[]))}
    state=DivisiveState(deepcopy(view),x.copy(),full,parents,cfg,_events(previous,view,full,parents),trace,previous.next_id,affinity)
    _trace_finish(state,science,edges,native,q)
    return state
