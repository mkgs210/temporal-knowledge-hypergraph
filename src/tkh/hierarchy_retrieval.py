"""Matched hierarchy overlays and seed-first traversal of a frozen hierarchy."""
import copy
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

from .hybrid_graph_search import ANSWER_TYPES, combine_channels, diffuse
from .tag_graph_index import IncidenceLayer


BASELINES={
    'bm25_own':({'bm25_own':1.},()),
    'bm25_context':({'lexical':1.},()),
    'e5_entities':({'dense':1.},()),
    'bm25_tags':({'lexical':.5,'tags':.5},()),
    'hybrid_no_graph':({'lexical':.5,'dense':.25,'tags':.25},()),
    'hybrid_graph':({'lexical':.5,'dense':.25,'tags':.25},('native','sources','tags')),
    'hybrid_no_native':({'lexical':.5,'dense':.25,'tags':.25},('sources','tags')),
    'hybrid_no_tags':({'lexical':.5,'dense':.25},('native','sources')),
    'bm25__native':({'lexical':1.},('native',)),
    'bm25__sources':({'lexical':1.},('sources',)),
    'bm25__native_sources':({'lexical':1.},('native','sources')),
    'lexical_dense__none':({'lexical':.5,'dense':.25},()),
    'lexical_dense__native':({'lexical':.5,'dense':.25},('native',)),
    'lexical_dense__sources':({'lexical':.5,'dense':.25},('sources',)),
}


class _FixedMixture:
    """A transition operator retaining empty induced layers' restart share."""
    def __init__(self,layers):
        self.layers=layers
        self.name='fixed_induced_mixture'
        self.weights=np.array([1.])
        # diffuse uses matrix.nnz only for activity and sparse-visit accounting.
        self.matrix=SimpleNamespace(nnz=sum(l.matrix.nnz for l in layers))

    def step(self,mass,restart):
        return sum((1/len(self.layers))*l.step(mass,restart) for l in self.layers)


def fixed_mixture_walk(seed,layers):
    operator=[_FixedMixture(layers)] if layers else []
    values,trace=diffuse(seed,operator)
    trace['component_layers']=[l.name for l in layers]
    trace['empty_induced_layers']=[l.name for l in layers if not l.matrix.nnz]
    trace['component_weights']=[1/len(layers)]*len(layers) if layers else []
    return values,trace


class PairedHierarchySearch:
    def __init__(self,ids,nodes,layers,hierarchy):
        self.ids=list(ids);self.nodes=nodes;self.layers=layers
        if len(set(ids))!=len(ids) or set(ids)!=set(nodes):raise ValueError('Original IDs must align')
        if any(layer.ids!=self.ids for layer in layers.values()):raise ValueError('Layer IDs differ')
        self.positions={n:i for i,n in enumerate(ids)}
        self.hierarchy=copy.deepcopy(hierarchy)
        self.groups,self.hierarchy_layers=self._prepare_hierarchy(self.hierarchy,'hierarchy')
        self.identity=IncidenceLayer(ids,[[n] for n in ids],name='identity')
        self.null_hierarchies=[];self.null_groups=[];self.null_layers=[]
        by_type=defaultdict(list)
        for n in ids:by_type[nodes[n]['type']].append(n)
        for seed in range(5):
            rng=np.random.default_rng(seed);mapping={}
            for kind in sorted(by_type):
                members=sorted(by_type[kind])
                mapping.update(zip(members,rng.permutation(members).tolist(),strict=True))
            null=copy.deepcopy(hierarchy)
            for level in null['levels']:
                for group in level['groups']:group['members']=sorted(mapping[n] for n in group['members'])
            groups,incidences=self._prepare_hierarchy(null,f'null_{seed}')
            self.null_hierarchies.append(null);self.null_groups.append(groups);self.null_layers.append(incidences)

    def _prepare_hierarchy(self,hierarchy,prefix):
        levels=[level for level in hierarchy['levels'] if level['index'] in (0,1)]
        if sorted(l['index'] for l in levels)!=[0,1]:raise ValueError('Exactly two hierarchy levels required')
        groups={l['index']:l['groups'] for l in levels}
        for current in groups.values():
            members=[n for g in current for n in g['members']]
            if len(members)!=len(self.ids) or set(members)!=set(self.ids):raise ValueError('Hierarchy must partition IDs')
            if len({g['id'] for g in current})!=len(current):raise ValueError('Duplicate group IDs')
        parents={g['id']:set(g['members']) for g in groups[0]}
        if any(g['parent_id'] not in parents or not set(g['members'])<=parents[g['parent_id']] for g in groups[1]):
            raise ValueError('Hierarchy must be laminar')
        layers=[IncidenceLayer(self.ids,[g['members'] for g in groups[level]],name=f'{prefix}_{level}') for level in (0,1)]
        return groups,layers

    def _rank(self,values,question_type,top_k):
        types=ANSWER_TYPES if question_type=='A' else {'article'}
        candidates=[i for i,n in enumerate(self.ids) if self.nodes[n]['type'] in types and values[i]>0]
        candidates.sort(key=lambda i:(-values[i],self.ids[i]))
        return {'ids':[self.ids[i] for i in candidates[:top_k]],
                'scores':[float(values[i]) for i in candidates[:top_k]]}

    def _route(self,seed,groups,broad_budget,fine_budget):
        def score(group):
            values=[seed[self.positions[n]] for n in group['members'] if self.nodes[n]['type'] not in {'article','author'}]
            return float(np.mean(sorted(values,reverse=True)[:5])) if values else 0.
        broad=sorted(groups[0],key=lambda g:(-score(g),g['id']))[:broad_budget]
        parents={g['id'] for g in broad}
        fine=sorted([g for g in groups[1] if g['parent_id'] in parents],key=lambda g:(-score(g),g['id']))[:fine_budget]
        allowed={n for g in fine for n in g['members']}
        retained=[i for i,n in enumerate(self.ids) if n in allowed]
        return retained,{'broad_ids':[g['id'] for g in broad],'fine_ids':[g['id'] for g in fine],
            'retained_ids':[self.ids[i] for i in retained], 'retained_nodes':len(retained),
            'seed_mass_retained':float(seed[retained].sum()),
            'full_graph_diffusion_before_routing':False,'initial_node_channels_already_evaluated':len(self.ids)}

    def _induce(self,layer,retained):
        ids=[self.ids[i] for i in retained]
        matrix=layer.matrix[retained].tocsc()
        columns=np.flatnonzero(np.asarray(matrix.sum(axis=0)).ravel()>0)
        edges=[[ids[i] for i in matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]]] for j in columns]
        return IncidenceLayer(ids,edges,weights=layer.weights[columns],name=layer.name)

    def predict(self,channels,question_type='A',top_k=100,routing_widths=((2,8),(4,16))):
        if question_type not in ('A','B') or top_k<1:raise ValueError('Invalid retrieval task')
        if any(type(b)is not int or type(f)is not int or b<1 or f<1 for b,f in routing_widths):raise ValueError('Invalid route budget')
        channels={k:np.asarray(v,dtype=float).copy() for k,v in channels.items()}
        for values in channels.values():
            if values.shape!=(len(self.ids),) or not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid aligned channel')
            for i,n in enumerate(self.ids):
                if self.nodes[n]['type']=='author':values[i]=0.
        rows={};induced_cache={};route_cache={}
        def save(base,condition,values,trace):
            rows[f'{base}::{condition}']={**self._rank(values,question_type,top_k),
                'probabilities':values,'trace':{**trace,'all_branch_replay_checked':True}}
        def routed(seed,active,groups,cache_key,broad,fine):
            key=(cache_key,broad,fine)
            if key not in route_cache:route_cache[key]=self._route(seed,groups,broad,fine)
            retained,routing=route_cache[key]
            local_seed=seed[retained]
            if local_seed.sum():local_seed=local_seed/local_seed.sum()
            local=[]
            for layer in active:
                induced_key=(layer.name,tuple(retained))
                if induced_key not in induced_cache:induced_cache[induced_key]=self._induce(layer,retained)
                local.append(induced_cache[induced_key])
            local_values,trace=fixed_mixture_walk(local_seed,local)
            values=np.zeros(len(self.ids));values[retained]=local_values
            trace['routing']=routing
            trace['induced_layers']={l.name:{'nodes':len(l.ids),'edges':len(l.edges),'incidences':int(l.matrix.nnz)} for l in local}
            return values,trace
        for base,(weights,names) in BASELINES.items():
            # Dict order matches the original primary controller (lexical/dense/tags).
            seed=combine_channels(channels,weights)
            active=[self.layers[n] for n in names if self.layers[n].matrix.nnz and np.any(self.layers[n].weights>0)]
            values,trace=diffuse(seed,active)
            baseline=self._rank(values,question_type,top_k)['ids']
            save(base,'baseline',values,trace)
            full,_=routed(seed,active,self.groups,(tuple(weights.items()),'true'),len(self.groups[0]),len(self.groups[1]))
            if self._rank(full,question_type,top_k)['ids']!=baseline:raise AssertionError(f'All-branch routing changed {base}')
            ordinary=active or [self.identity]
            ordinary_weights=[.85/len(ordinary)]*len(ordinary)
            for condition,hierarchy_layers in [('hierarchy',self.hierarchy_layers),
                    ('identity_overlay',[self.identity,self.identity])]+[(f'null_{n}',l) for n,l in enumerate(self.null_layers)]:
                mixed,trace=diffuse(seed,ordinary+hierarchy_layers,layer_weights=ordinary_weights+[.075,.075])
                save(base,condition,mixed,trace)
            for broad,fine in routing_widths:
                for label,groups in [('true',self.groups)]+[(str(n),g) for n,g in enumerate(self.null_groups)]:
                    values,trace=routed(seed,active,groups,(tuple(weights.items()),label),broad,fine)
                    condition=f'route_{broad}_{fine}' if label=='true' else f'null_{label}_route_{broad}_{fine}'
                    save(base,condition,values,trace)
        return rows
