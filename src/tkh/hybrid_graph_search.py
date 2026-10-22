"""Gold-free BM25/tag/E5 seeds and probability diffusion over native incidences."""
import hashlib
import numpy as np

from .tag_graph_index import IncidenceLayer

ANSWER_TYPES={'method','technique','component','cited_work','dataset'}
BASE_ARMS=('bm25_own','bm25_context','e5_entities','bm25_tags','hybrid_no_graph',
           'hybrid_graph','hybrid_no_native','hybrid_no_tags')
ARMS=BASE_ARMS+('hybrid_graph_hierarchy','hybrid_graph_routed')+tuple(
    f'hybrid_shuffled_native_{s}' for s in range(5))


def combine_channels(channels,weights):
    if not channels:
        raise ValueError('At least one channel required')
    arrays={k:np.asarray(v,dtype=np.float64) for k,v in channels.items()}
    shapes={x.shape for x in arrays.values()}
    if len(shapes)!=1 or len(next(iter(shapes)))!=1:
        raise ValueError('Aligned one-dimensional channel scores required')
    result=np.zeros_like(next(iter(arrays.values())))
    for name,x in arrays.items():
        if not np.isfinite(x).all() or (x<0).any():
            raise ValueError('Channel scores must be finite and nonnegative')
        w=weights.get(name,0.)
        if not np.isfinite(w) or w<0:
            raise ValueError('Channel weight must be finite and nonnegative')
        if x.sum()>0:
            result+=w*x/x.sum()
    return result/result.sum() if result.sum()>0 else result


def diffuse(seed,layers,*,alpha=.65,max_steps=40,tol=1e-9,layer_weights=None):
    seed=np.asarray(seed,dtype=np.float64)
    if (seed.ndim!=1 or not np.isfinite(seed).all() or (seed<0).any()
            or not 0<=alpha<1 or max_steps<1 or tol<=0):
        raise ValueError('Invalid probability diffusion parameters')
    seed=seed/seed.sum() if seed.sum() else seed.copy()
    weights=np.ones(len(layers)) if layer_weights is None else np.asarray(layer_weights,dtype=float)
    if weights.shape!=(len(layers),) or not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Invalid layer weights')
    active=[(layer,float(w)) for layer,w in zip(layers,weights,strict=True)
            if w>0 and layer.matrix.nnz and np.any(layer.weights>0)]
    if not seed.any() or not active or alpha==0:
        return seed,{'iterations':0,'converged':True,'residual_l1':0.,'mass':float(seed.sum()),
                     'layers':[],'alpha':alpha,'sparse_incidence_visits':0}
    total=sum(w for _,w in active)
    x=seed.copy();residual=float('inf')
    for iteration in range(1,max_steps+1):
        walked=sum((w/total)*layer.step(x,seed) for layer,w in active)
        updated=(1-alpha)*seed+alpha*walked
        if not np.isfinite(updated).all() or (updated<-1e-14).any() or not np.isclose(updated.sum(),1.,atol=1e-10):
            raise ValueError('Diffusion failed mass/finite invariant')
        residual=float(np.abs(updated-x).sum());x=updated
        if residual<=tol:break
    return x,{'iterations':iteration,'converged':residual<=tol,'residual_l1':residual,
        'mass':float(x.sum()),'alpha':alpha,'layers':[layer.name for layer,_ in active],
        'layer_weights':[w/total for _,w in active],
        'sparse_incidence_visits':2*iteration*sum(layer.matrix.nnz for layer,_ in active)}


def shuffle_native_layer(layer,seed=0,swaps_per_incidence=5):
    """Bipartite endpoint switches preserve each original degree and edge arity."""
    matrix=layer.matrix.tocsc()
    edges=[matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]].tolist() for j in range(matrix.shape[1])]
    members=[set(e) for e in edges]
    slots=[(j,i) for j,e in enumerate(edges) for i in range(len(e))]
    rng=np.random.default_rng(seed);success=0
    attempts=len(slots)*swaps_per_incidence if len(slots)>1 else 0
    for _ in range(attempts):
        a,b=rng.integers(len(slots),size=2)
        e,i=slots[int(a)];f,j=slots[int(b)]
        u,v=edges[e][i],edges[f][j]
        if e==f or u==v or u in members[f] or v in members[e]:continue
        members[e].remove(u);members[e].add(v);members[f].remove(v);members[f].add(u)
        edges[e][i]=v;edges[f][j]=u;success+=1
    result=IncidenceLayer(layer.ids,[[layer.ids[i] for i in e] for e in edges],
                          weights=layer.weights.copy(),name=f'shuffled_native_{seed}')
    if not np.array_equal(result.matrix.sum(axis=0),layer.matrix.sum(axis=0)) or not np.array_equal(result.matrix.sum(axis=1),layer.matrix.sum(axis=1)):
        raise AssertionError('Shuffle changed native degrees/arities')
    return result,{'seed':seed,'attempts':attempts,'successful_swaps':success,
        'degree_and_arity_preserved':True,'changed_incidence_entries':int((result.matrix!=layer.matrix).nnz),
        'scope':'Native propagation only; original context/tag/lexical inputs held fixed.'}


class HybridGraphSearch:
    def __init__(self,index,passage_vectors=None,hierarchy=None):
        self.index=index;self.ids=list(index.ids);self.position={n:i for i,n in enumerate(self.ids)}
        self.layers=dict(index.layers);self.hierarchy=hierarchy
        self.non_author=np.array([index.nodes[n]['type']!='author' for n in self.ids])
        self.passages=None if passage_vectors is None else np.asarray(passage_vectors,dtype=np.float64)
        if self.passages is not None and (self.passages.ndim!=2 or len(self.passages)!=len(self.ids)
                or not np.isfinite(self.passages).all() or not np.allclose(np.linalg.norm(self.passages,axis=1),1.,atol=1e-5)):
            raise ValueError('Passage vectors must be aligned finite unit vectors')
        self.hierarchy_layers={};self.groups={}
        if hierarchy is not None:
            levels={l['index']:l['groups'] for l in hierarchy['levels'] if l['index'] in (0,1)}
            if set(levels)!={0,1}:raise ValueError('Two hierarchy levels required')
            for level,groups in levels.items():
                members=[n for g in groups for n in g['members']]
                if len(members)!=len(self.ids) or set(members)!=set(self.ids):
                    raise ValueError('Hierarchy must partition exactly the graph IDs')
                if len({g['id'] for g in groups})!=len(groups):raise ValueError('Duplicate groups')
                self.groups[level]=groups
                self.hierarchy_layers[str(level)]=IncidenceLayer(self.ids,[g['members'] for g in groups],name=f'hierarchy_{level}')
            parents={g['id']:set(g['members']) for g in levels[0]}
            if any(g['parent_id'] not in parents or not set(g['members'])<=parents[g['parent_id']] for g in levels[1]):
                raise ValueError('Hierarchy is not laminar')
        self.shuffle_layers={};self.shuffle_traces={}
        for seed in range(5):
            self.shuffle_layers[seed],self.shuffle_traces[seed]=shuffle_native_layer(self.layers['native'],seed)
        self.config={'alpha':.65,'max_steps':40,'tolerance':1e-9,'dense_temperature':.05,
            'weights':{'lexical':.5,'dense':.25,'tags':.25},'hierarchy_weight':.15,
            'candidate_types_A':sorted(ANSWER_TYPES),'candidate_types_B':['article'],
            'primary_arm':'hybrid_graph','primary_metric':'Recall@30(all A targets)',
            'arms':list(ARMS if hierarchy is not None else BASE_ARMS+ARMS[10:]),
            'tie_policy':'descending score then ascending original node ID; no zero-score padding',
            'routing':'post-diffusion top2 broad/top8 fine; mean top5 science-member scores; upstream full work retained'}
        self.last_channels={};self.last_scores={}

    def describe(self):
        return {**self.config,'nodes':len(self.ids),'native_shuffle':self.shuffle_traces,
                'layers':{n:l.describe() for n,l in self.layers.items()}}

    def _routed(self,scores):
        def group_score(g):
            x=[scores[self.position[n]] for n in g['members'] if self.index.nodes[n]['type'] not in {'article','author'}]
            return float(np.mean(sorted(x,reverse=True)[:5])) if x else 0.
        tops=sorted(self.groups[0],key=lambda g:(-group_score(g),g['id']))[:2]
        topids={g['id'] for g in tops}
        fine=sorted([g for g in self.groups[1] if g['parent_id'] in topids],key=lambda g:(-group_score(g),g['id']))[:8]
        allowed={n for g in fine for n in g['members']}
        return allowed,{'top_ids':sorted(topids),'fine_ids':[g['id'] for g in fine],
                       'candidate_region_size':len(allowed),'upstream_full_graph_work':True}

    def predict(self,question,query_vector=None,question_type='A',top_k=100):
        if not isinstance(question,str) or not question.strip() or question_type not in ('A','B') or type(top_k)is not int or top_k<1:
            raise ValueError('Nonempty question, A/B type and positive output budget required')
        own=np.asarray(self.index.bm25_scores(question,context=False),dtype=float)
        lexical=np.asarray(self.index.bm25_scores(question,context=True),dtype=float)
        tags=np.asarray(self.index.tag_scores(question),dtype=float)
        for x in (own,lexical,tags):
            if x.shape!=(len(self.ids),) or not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid lexical/tag scores')
            x[~self.non_author]=0.
        dense=np.zeros(len(self.ids));cosines=np.zeros(len(self.ids))
        if query_vector is not None:
            q=np.asarray(query_vector,dtype=float)
            if self.passages is None or q.shape!=(self.passages.shape[1],) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q),1.,atol=1e-5):
                raise ValueError('Query vector must match cached unit passage vectors')
            cosines=self.passages@q
            values=cosines[self.non_author]/self.config['dense_temperature']
            dense[self.non_author]=np.exp(values-values.max())
        self.last_channels={'bm25_own':own,'lexical':lexical,'dense':dense,'tags':tags,'cosine':cosines}
        channels={'lexical':lexical,'dense':dense,'tags':tags}
        seed=combine_channels(channels,self.config['weights'])
        notags=combine_channels(channels,{'lexical':.5,'dense':.25})
        no_graph={'iterations':0,'converged':True,'residual_l1':0.,'layers':[],'sparse_incidence_visits':0}
        scores={'bm25_own':combine_channels({'x':own},{'x':1}),
                'bm25_context':combine_channels({'x':lexical},{'x':1}),
                'e5_entities':combine_channels({'x':dense},{'x':1}),
                'bm25_tags':combine_channels(channels,{'lexical':.5,'tags':.5}),
                'hybrid_no_graph':seed}
        traces={n:dict(no_graph) for n in scores}
        arm_seeds=dict(scores)
        normal=[self.layers[n] for n in ('native','sources','tags')]
        for name,start,layers in [('hybrid_graph',seed,normal),
                ('hybrid_no_native',seed,[self.layers[n] for n in ('sources','tags')]),
                ('hybrid_no_tags',notags,[self.layers[n] for n in ('native','sources')])]:
            scores[name],traces[name]=diffuse(start,layers)
            arm_seeds[name]=start
        if self.hierarchy_layers:
            layers=normal+list(self.hierarchy_layers.values())
            # Keep .15 hierarchy mass even when some ordinary layers are empty.
            active=sum(bool(l.matrix.nnz and np.any(l.weights>0)) for l in normal)
            weights=[.85/active if active else 0. for _ in normal]+[.075,.075]
            scores['hybrid_graph_hierarchy'],traces['hybrid_graph_hierarchy']=diffuse(seed,layers,layer_weights=weights)
            allowed,routing=self._routed(scores['hybrid_graph'])
            scores['hybrid_graph_routed']=scores['hybrid_graph'].copy()
            traces['hybrid_graph_routed']={**traces['hybrid_graph'],'routing':routing}
            arm_seeds['hybrid_graph_hierarchy']=seed
            arm_seeds['hybrid_graph_routed']=seed
        for n,layer in self.shuffle_layers.items():
            name=f'hybrid_shuffled_native_{n}'
            scores[name],traces[name]=diffuse(seed,[layer,self.layers['sources'],self.layers['tags']])
            traces[name]['shuffle']=self.shuffle_traces[n]
            arm_seeds[name]=seed
        self.last_scores=scores
        candidate=[i for i,n in enumerate(self.ids) if self.index.nodes[n]['type'] in (ANSWER_TYPES if question_type=='A' else {'article'})]
        result={}
        digest=hashlib.sha256(np.stack([own,lexical,tags,dense]).tobytes()).hexdigest()
        for name,values in scores.items():
            selected=[i for i in candidate if values[i]>0 and (name!='hybrid_graph_routed' or self.ids[i] in allowed)]
            selected.sort(key=lambda i:(-values[i],self.ids[i]))
            ranked=[{'node_id':self.ids[i],'score':float(values[i]),'seed_probability':float(arm_seeds[name][i]),
                'channel_scores':{'bm25':float((own if name=='bm25_own' else lexical)[i]),
                                  'tags':float(tags[i]),'cosine':float(cosines[i])}}
                for i in selected[:top_k]]
            result[name]={'ranked':ranked,'trace':{**traces[name],'eligible_candidates':len(candidate),
                'positive_candidates':len(selected),'returned':len(ranked),'channel_sha256':digest,
                'query_tags':self.index.query_tags(question),
                'scoring_scope':'All graph nodes; output restricted to fixed original A/B types.',
                'primary_arm':name=='hybrid_graph'}}
        return result
