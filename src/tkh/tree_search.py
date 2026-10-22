"""Experimental coarse-first search with no pre-routing leaf scoring.

Group documents and unit-mean sentence prototypes are built once from the fixed
hierarchy. Queries score all coarse groups, only their candidate fine children,
then only selected leaves. BM25 keeps GLOBAL corpus weights: a prebuilt CSR
document matrix is sliced before query-token accumulation. This is not an
incremental BM25 index or a claim of end-to-end speedup.

Grounding: BM25S maintainer discussion69 documents the weighted CSC arrays;
Gupta et al., Hierarchical Semantic Retrieval with Cobweb (arXiv:2510.02539),
motivates hierarchy traversal. No participant implementation is used.
"""
from copy import deepcopy
import time

import numpy as np
from scipy import sparse

from .hybrid_graph_search import ANSWER_TYPES, combine_channels
from .hierarchy_retrieval import fixed_mixture_walk
from .tag_graph_index import IncidenceLayer


class SelectedBM25:
    """Exact selected-document scores for the repository's BM25S Lucene model."""
    def __init__(self, model):
        self.model=model
        self.matrix=None
        if model is not None:
            if model.method!='lucene' or model.nonoccurrence_array is not None:
                raise ValueError('Selected scoring currently requires Lucene occurrence-only weights')
            record=model.scores
            self.matrix=sparse.csc_matrix((record['data'],record['indices'],record['indptr']),
                shape=(record['num_docs'],len(record['indptr'])-1)).tocsr()

    def score(self,tokens,rows):
        rows=np.asarray(rows,dtype=int)
        if rows.ndim!=1 or len(np.unique(rows))!=len(rows) or (rows<0).any():
            raise ValueError('Selected rows must be distinct nonnegative positions')
        trace={'documents_scored':len(rows),'global_posting_scores_evaluated':False,
               'selected_row_postings':0,'query_posting_contributions':0,
               'postings_scope':'CSR selected-document rows before query columns; no full-corpus query score vector'}
        if self.model is None:return np.zeros(len(rows)),trace
        if (rows>=self.matrix.shape[0]).any():raise ValueError('Selected row outside corpus')
        local=self.matrix[rows]
        trace['selected_row_postings']=int(local.nnz)
        token_ids=self.model.get_tokens_ids(tokens)
        # The library accumulates in corpus dtype and counts duplicate tokens.
        result=np.zeros(len(rows),dtype=self.model.dtype)
        for token in token_ids:
            column=local[:,token]
            result+=column.toarray().ravel()
            trace['query_posting_contributions']+=int(column.nnz)
        return result.astype(float),trace


class TreeSearch:
    """Fixed lexical/dense group routing followed by an induced native/source walk."""
    def __init__(self,index,passages,hierarchy):
        started=time.perf_counter()
        self.index=index;self.ids=list(index.ids);self.positions={n:i for i,n in enumerate(self.ids)}
        self.passages=np.asarray(passages,dtype=float)
        if (self.passages.ndim!=2 or len(self.passages)!=len(self.ids) or
                not np.isfinite(self.passages).all() or
                not np.allclose(np.linalg.norm(self.passages,axis=1),1.,atol=1e-5)):
            raise ValueError('Expected aligned finite unit passage vectors')
        self.groups={l['index']:deepcopy(l['groups']) for l in hierarchy['levels'] if l['index'] in (0,1)}
        if set(self.groups)!={0,1}:raise ValueError('Exactly two group levels required')
        for groups in self.groups.values():
            members=[n for g in groups for n in g['members']]
            if len(members)!=len(self.ids) or set(members)!=set(self.ids) or len({g['id'] for g in groups})!=len(groups):
                raise ValueError('Group levels must partition the original corpus')
        parents={g['id']:set(g['members']) for g in self.groups[0]}
        if any(g['parent_id'] not in parents or not set(g['members'])<=parents[g['parent_id']] for g in self.groups[1]):
            raise ValueError('Hierarchy must be laminar')
        self.non_author=np.array([index.nodes[n]['type']!='author' for n in self.ids])
        self.lexical=SelectedBM25(index._bm25[True])
        documents={doc['id']:doc['text'] for doc in index.documents}
        self.group_lexical={};self.prototypes={};self.group_documents={}
        for level,groups in self.groups.items():
            texts=[];prototypes=[]
            for group in groups:
                science=sorted(n for n in group['members'] if index.nodes[n]['type'] not in {'article','author'})
                texts.append('\n'.join(documents[n] for n in science))
                vector=self.passages[[self.positions[n] for n in science]].sum(axis=0) if science else np.zeros(self.passages.shape[1])
                norm=np.linalg.norm(vector)
                prototypes.append(vector/norm if norm else vector)
            self.group_documents[level]=texts
            self.group_lexical[level]=SelectedBM25(index._make_bm25(texts))
            self.prototypes[level]=np.asarray(prototypes)
        self.layers=[index.layers[n] for n in ('native','sources')
                     if index.layers[n].matrix.nnz and np.any(index.layers[n].weights>0)]
        self.induced={}
        self.build_seconds=time.perf_counter()-started

    def _group_scores(self,level,rows,tokens,query):
        lexical,trace=self.group_lexical[level].score(tokens,rows)
        channels={'lexical':lexical}
        if query is not None:
            logits=self.prototypes[level][rows]@query/.05
            dense=np.exp(logits-logits.max()) if len(rows) else np.zeros(0)
            # Metadata-only regions have no scientific prototype.
            dense[np.linalg.norm(self.prototypes[level][rows],axis=1)==0]=0.
            channels['dense']=dense
        return combine_channels(channels,{'lexical':.5,'dense':.25}),trace

    def _layers(self,rows):
        key=tuple(rows)
        if key not in self.induced:
            ids=[self.ids[i] for i in rows];layers=[]
            for layer in self.layers:
                matrix=layer.matrix[rows].tocsc()
                columns=np.flatnonzero(np.asarray(matrix.sum(axis=0)).ravel()>0)
                edges=[[ids[i] for i in matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]]] for j in columns]
                layers.append(IncidenceLayer(ids,edges,layer.weights[columns],layer.name))
            self.induced[key]=layers
        return self.induced[key]

    def predict(self,question,query_vector,*,broad_width=2,fine_width=8,question_type='A',top_k=100):
        if not isinstance(question,str) or not question.strip() or question_type not in {'A','B'} or type(top_k)is not int or top_k<1:
            raise ValueError('Expected a nonempty question, A/B type and positive top_k')
        if any(w is not None and (type(w)is not int or w<1) for w in (broad_width,fine_width)):
            raise ValueError('Widths must be positive integers or None for all')
        query=None if query_vector is None else np.asarray(query_vector,dtype=float)
        if query is not None and (query.shape!=(self.passages.shape[1],) or not np.isfinite(query).all() or not np.isclose(np.linalg.norm(query),1.,atol=1e-5)):
            raise ValueError('Expected an aligned finite unit query vector')
        started=time.perf_counter();tokens=self.index._tokens(question)
        coarse_rows=list(range(len(self.groups[0])))
        coarse_scores,coarse_trace=self._group_scores(0,coarse_rows,tokens,query)
        ranked=sorted(coarse_rows,key=lambda i:(-coarse_scores[i],self.groups[0][i]['id']))
        broad=ranked[:broad_width] if broad_width is not None else ranked
        parents={self.groups[0][i]['id'] for i in broad}
        fine_rows=[i for i,g in enumerate(self.groups[1]) if g['parent_id'] in parents]
        fine_scores,fine_trace=self._group_scores(1,fine_rows,tokens,query)
        ranked=sorted(zip(fine_rows,fine_scores,strict=True),key=lambda pair:(-pair[1],self.groups[1][pair[0]]['id']))
        fine=ranked[:fine_width] if fine_width is not None else ranked
        selected={n for i,_ in fine for n in self.groups[1][i]['members']}
        rows=[i for i,n in enumerate(self.ids) if n in selected]
        lexical,leaf_trace=self.lexical.score(tokens,rows)
        mask=self.non_author[rows];lexical[~mask]=0.
        channels={'lexical':lexical};dense_count=0
        if query is not None:
            dense=np.zeros(len(rows));eligible=np.asarray(rows)[mask]
            values=(self.passages[eligible]@query)/.05
            if len(values):dense[mask]=np.exp(values-values.max())
            channels['dense']=dense;dense_count=len(eligible)
        seed=combine_channels(channels,{'lexical':.5,'dense':.25})
        local,walk=fixed_mixture_walk(seed,self._layers(rows))
        probabilities=np.zeros(len(self.ids));probabilities[rows]=local
        types=ANSWER_TYPES if question_type=='A' else {'article'}
        candidates=[i for i in rows if self.index.nodes[self.ids[i]]['type'] in types and probabilities[i]>0]
        candidates.sort(key=lambda i:(-probabilities[i],self.ids[i]))
        chosen=candidates[:top_k]
        trace={'coarse_scores':len(coarse_rows),'fine_scores':len(fine_rows),'leaf_scores':len(rows),
               'leaf_dense_dot_products':dense_count,'group_dense_dot_products':len(coarse_rows)+len(fine_rows) if query is not None else 0,
               'full_leaf_scores_before_routing':0,'selected_coarse_ids':[self.groups[0][i]['id'] for i in broad],
               'selected_fine_ids':[self.groups[1][i]['id'] for i,_ in fine],
               'retained_ids':[self.ids[i] for i in rows],'bm25':{'coarse':coarse_trace,'fine':fine_trace,'leaf':leaf_trace},
               'walk':walk,'query_seconds':time.perf_counter()-started,
               'routing':'BM25 concatenated scientific-member context + unit mean sentence prototype; weights .5/.25',
               'limitations':['Group indexing scans full member text at build time; no truncation.',
                              'CSR row slicing reads selected-row vocabulary entries, not only query matches.',
                              'Induced walk cannot reach nodes outside the routed region.',
                              'Operation counts are not a latency or semantic-quality guarantee.']}
        return {'ids':[self.ids[i] for i in chosen],'scores':[float(probabilities[i]) for i in chosen],
                'probabilities':probabilities,'trace':trace}
