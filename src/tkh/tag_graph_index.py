"""Gold-free lexical/tag indexing and sparse native hypergraph transitions.

Derived source/tag memberships describe retrieval context and co-occurrence;
none assert entailment or modify the original knowledge graph.
"""
from __future__ import annotations

import copy
import math
import re
from collections import defaultdict

import bm25s
import numpy as np
import Stemmer
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize

from .corpus import validate_graph


class IncidenceLayer:
    """Binary node/edge incidence with weighted, column-stochastic diffusion."""

    def __init__(self, ids, edges, weights=None, name="layer"):
        self.ids = list(ids)
        if (not all(isinstance(nid, str) and nid for nid in self.ids)
                or len(set(self.ids)) != len(self.ids)):
            raise ValueError("Incidence node IDs must be nonempty unique strings")
        positions = {nid: i for i, nid in enumerate(self.ids)}
        self.edges = [list(edge) for edge in edges]
        self.weights = np.asarray(weights if weights is not None else np.ones(len(self.edges)), dtype=float)
        if (self.weights.shape != (len(self.edges),) or not np.isfinite(self.weights).all()
                or np.any(self.weights < 0)):
            raise ValueError("Incidence weights must be finite nonnegative and match edges")
        rows, columns = [], []
        for column, members in enumerate(self.edges):
            if not members or len(set(members)) != len(members):
                raise ValueError("Incidence edges must have nonempty unique members")
            if any(member not in positions for member in members):
                raise ValueError("Incidence edge has an unknown member")
            self.edges[column] = sorted(members)
            rows.extend(positions[member] for member in members)
            columns.extend([column] * len(members))
        self.name = str(name)
        self.matrix = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)),
                                        shape=(len(self.ids), len(self.edges)))
        self.edge_degrees = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.node_degrees = np.asarray(self.matrix @ self.weights).ravel()
        if not np.isfinite(self.node_degrees).all():
            raise ValueError("Weighted node degrees overflowed")
        self.edge_ids = [f"{self.name}:{i}" for i in range(len(self.edges))]
        self.records = [{"id": eid, "members": members, "weight": float(weight), "kind": self.name}
                        for eid, members, weight in zip(self.edge_ids, self.edges, self.weights)]

    def step(self, mass, restart):
        """Apply H W De^-1 H.T Dv^-1, returning dangling mass to restart."""
        mass = np.asarray(mass, dtype=float)
        restart = np.asarray(restart, dtype=float)
        shape = (len(self.ids),)
        if (mass.shape != shape or restart.shape != shape
                or not np.isfinite(mass).all() or not np.isfinite(restart).all()
                or np.any(mass < 0) or np.any(restart < 0)):
            raise ValueError("Mass and restart must be aligned finite nonnegative vectors")
        restart_total = float(restart.sum())
        if not math.isfinite(float(mass.sum())) or not math.isfinite(restart_total):
            raise ValueError("Mass totals must be finite")
        dangling = self.node_degrees == 0
        dangling_mass = float(mass[dangling].sum())
        if dangling_mass and restart_total <= 0:
            raise ValueError("Positive dangling mass requires a positive restart distribution")
        scaled = np.divide(mass, self.node_degrees, out=np.zeros_like(mass), where=~dangling)
        edge_mass = self.matrix.T @ scaled
        edge_mass *= np.divide(self.weights, self.edge_degrees,
                               out=np.zeros_like(self.weights), where=self.edge_degrees > 0)
        result = np.asarray(self.matrix @ edge_mass).ravel()
        if dangling_mass:
            result += dangling_mass * (restart / restart_total)
        return result

    def describe(self):
        return {"name": self.name, "nodes": len(self.ids), "edges": len(self.edges),
                "incidences": int(self.matrix.nnz), "dangling_nodes": int(np.count_nonzero(self.node_degrees == 0)),
                "weight_sum": float(self.weights.sum()),
                "operator": "H W De^-1 H.T Dv^-1; dangling mass returns to restart",
                "records": self.records}


class TagGraphIndex:
    """BM25S own/context documents, extractive TF-IDF tags and incidence layers."""

    def __init__(self, graph):
        validate_graph(graph)
        self.nodes = {node["id"]: copy.deepcopy(node) for node in graph["nodes"]}
        self.ids = sorted(self.nodes)
        self._stemmer = Stemmer.Stemmer("english")
        native_edges = graph["hyperedges"]
        self.layers = {"native": IncidenceLayer(self.ids, [edge["members"] for edge in native_edges],
                                              [edge.get("weight", 1) for edge in native_edges], "native")}
        self._set_records(self.layers["native"], [edge["id"] for edge in native_edges])
        incident = defaultdict(list)
        for edge in native_edges:
            for nid in edge["members"]:
                incident[nid].append(edge)
        sources, article_nodes = defaultdict(set), defaultdict(set)
        node_sources = {}
        for nid in self.ids:
            source_ids = self.nodes[nid].get("provenance", {}).get("articles", [])
            if (not isinstance(source_ids, list)
                    or any(not isinstance(s, (str, int)) or isinstance(s, bool) or s == "" for s in source_ids)):
                raise ValueError("Provenance articles must be a list of source identifiers")
            if len(set(source_ids)) != len(source_ids):
                raise ValueError("Provenance article identifiers must be unique")
            node_sources[nid] = source_ids
            for source in source_ids:
                sources[source].add(nid)
                if self.nodes[nid]["type"] == "article":
                    article_nodes[source].add(nid)
        if any(len(owners) > 1 for owners in article_nodes.values()):
            raise ValueError("Ambiguous source article ownership")
        source_ids = sorted(sources, key=lambda value: (type(value).__name__, str(value)))
        self.layers["sources"] = IncidenceLayer(self.ids, [sources[s] for s in source_ids], name="sources")
        self._set_records(self.layers["sources"], [f"source:{type(s).__name__}:{s}" for s in source_ids])
        for record, source in zip(self.layers["sources"].records, source_ids):
            record["source_article_id"] = source
        self.documents = []
        for nid in self.ids:
            node = self.nodes[nid]
            own = self._own_text(node)
            added_nodes, added_edges = set(), set()
            texts = [own]
            if node["type"] not in {"author", "article"}:
                for source in node_sources[nid]:
                    for article in sorted(article_nodes.get(source, ())):
                        if article != nid and article not in added_nodes:
                            texts.append(self.nodes[article]["surface_form"])
                            added_nodes.add(article)
                # Gather context directly from incidence lists, without a clique projection.
                neighbor_edges = defaultdict(set)
                for edge in incident[nid]:
                    for other in edge["members"]:
                        if other != nid and self.nodes[other]["type"] not in {"author", "article"}:
                            neighbor_edges[other].add(edge["id"])
                for other in sorted(neighbor_edges)[:8]:
                    texts.append(self._own_text(self.nodes[other]))
                    added_nodes.add(other)
                    added_edges.update(neighbor_edges[other])
            self.documents.append({"id": nid, "type": node["type"], "surface_form": node["surface_form"],
                                   "own_text": own, "text": "\n".join(texts), "tag_ids": [],
                                   "context_node_ids": sorted(added_nodes), "context_edge_ids": sorted(added_edges),
                                   "source_article_ids": sorted(node_sources[nid], key=lambda value: (type(value).__name__, str(value))),
                                   "unresolved_source_article_ids": sorted(
                                       (source for source in node_sources[nid] if source not in article_nodes),
                                       key=lambda value: (type(value).__name__, str(value)))})
        self._documents_by_id = {doc["id"]: doc for doc in self.documents}
        self._bm25 = {False: self._make_bm25([doc["own_text"] for doc in self.documents]),
                      True: self._make_bm25([doc["text"] for doc in self.documents])}
        self._build_tags()

    @staticmethod
    def _set_records(layer, ids):
        layer.edge_ids = list(ids)
        for record, eid in zip(layer.records, ids):
            record["id"] = eid

    @staticmethod
    def _own_text(node):
        return "; ".join([node["type"] + ": " + node["surface_form"], *node.get("aliases", [])])

    def _tokens(self, text):
        return bm25s.tokenize([text], stopwords="english", stemmer=self._stemmer,
                              return_ids=False, show_progress=False)[0]

    def _make_bm25(self, texts):
        tokens = bm25s.tokenize(texts, stopwords="english", stemmer=self._stemmer,
                               return_ids=False, show_progress=False)
        if not any(tokens):
            return None
        model = bm25s.BM25(method="lucene", k1=1.5, b=.75)
        model.index(tokens, show_progress=False)
        return model

    def bm25_scores(self, query, context=True):
        tokens = self._tokens(query)
        model = self._bm25[bool(context)]
        if model is None or not tokens:
            return np.zeros(len(self.ids))
        return np.asarray(model.get_scores(tokens), dtype=float)

    def _analyze(self, text):
        words = [word for word in re.findall(r"(?u)\b\w\w+\b", text.lower()) if word not in ENGLISH_STOP_WORDS]
        stems = self._stemmer.stemWords(words)
        return stems + [left + " " + right for left, right in zip(stems, stems[1:])]

    def _build_tags(self):
        eligible = [i for i, nid in enumerate(self.ids) if self.nodes[nid]["type"] != "author"]
        self.tag_records = []
        self._tag_vectorizer = None
        self._tag_matrix = sparse.csr_matrix((len(self.ids), 0))
        self._tag_columns = {}
        self._tag_min_df = 1 if len(eligible) < 4 else 2
        self._tag_max_df = max(self._tag_min_df, math.floor(.6 * len(eligible)))
        if eligible:
            texts = [self.nodes[self.ids[i]]["surface_form"] if self.nodes[self.ids[i]]["type"] == "article"
                     else self.documents[i]["text"] for i in eligible]
            vectorizer = TfidfVectorizer(analyzer=self._analyze, min_df=self._tag_min_df,
                                         max_df=self._tag_max_df, max_features=6000, norm="l2")
            try:
                fitted = vectorizer.fit_transform(texts).tocsr()
            except ValueError as exc:
                if "empty vocabulary" not in str(exc) and "no terms remain" not in str(exc).lower():
                    raise
            else:
                features = vectorizer.get_feature_names_out()
                evidence_tokens = {}
                for global_row, text in zip(eligible, texts):
                    spans = [(match.group(), match.start(), match.end())
                             for match in re.finditer(r"(?u)\b\w\w+\b", text)
                             if match.group().lower() not in ENGLISH_STOP_WORDS]
                    stems = self._stemmer.stemWords([word.lower() for word, _, _ in spans])
                    evidence_tokens[self.ids[global_row]] = (text, spans, stems)
                rows, columns, values = [], [], []
                for local_row, global_row in enumerate(eligible):
                    row = fitted.getrow(local_row)
                    top = sorted(zip(row.indices, row.data), key=lambda pair: (-pair[1], features[pair[0]]))[:12]
                    for column, value in top:
                        rows.append(global_row)
                        columns.append(column)
                        values.append(value)
                self._tag_matrix = normalize(sparse.csr_matrix((values, (rows, columns)),
                                                              shape=(len(self.ids), len(features))), norm="l2").tocsr()
                self._tag_vectorizer = vectorizer
                by_column = self._tag_matrix.tocsc()
                for column, text in enumerate(features):
                    members = [self.ids[row] for row in by_column.indices[by_column.indptr[column]:by_column.indptr[column + 1]]]
                    if not members:
                        continue
                    tid = f"tag:{column:05d}"
                    self._tag_columns[column] = tid
                    support = []
                    for nid in members:
                        doc = self._documents_by_id[nid]
                        doc["tag_ids"].append(tid)
                        original, spans, stems = evidence_tokens[nid]
                        keys = str(text).split()
                        offset = next(i for i in range(len(stems) - len(keys) + 1)
                                      if stems[i:i + len(keys)] == keys)
                        phrase = original[spans[offset][1]:spans[offset + len(keys) - 1][2]]
                        support.append({"node_id": nid, "phrase": phrase, "context_node_ids": doc["context_node_ids"],
                                        "context_edge_ids": doc["context_edge_ids"],
                                        "source_article_ids": doc["source_article_ids"]})
                    self.tag_records.append({"id": tid, "key": str(text), "text": support[0]["phrase"], "idf": float(vectorizer.idf_[column]),
                                             "members": members, "support": support,
                                             "semantics": "extractive retrieval context; co-occurrence does not imply entailment"})
        self.layers["tags"] = IncidenceLayer(self.ids, [record["members"] for record in self.tag_records],
                                            [record["idf"] for record in self.tag_records], "tags")
        self._set_records(self.layers["tags"], [record["id"] for record in self.tag_records])

    def tag_scores(self, query):
        if self._tag_vectorizer is None:
            return np.zeros(len(self.ids))
        query_vector = self._tag_vectorizer.transform([query])
        return np.asarray((self._tag_matrix @ query_vector.T).toarray()).ravel()

    def query_tags(self, query):
        if self._tag_vectorizer is None:
            return []
        row = self._tag_vectorizer.transform([query]).tocsr()
        return [self._tag_columns[column] for column in sorted(row.indices) if column in self._tag_columns]

    def describe(self):
        return {"nodes": len(self.ids), "bm25": {"library": "bm25s", "version": bm25s.__version__,
                                                "method": "lucene", "k1": 1.5, "b": .75,
                                                "stemmer": "PyStemmer English Snowball"},
                "tags": {"count": len(self.tag_records), "features": self._tag_matrix.shape[1],
                         "top_per_node": 12, "min_df": self._tag_min_df, "max_df_fraction": .6,
                         "effective_max_df_count": self._tag_max_df, "max_features": 6000,
                         "scoring": "cosine of selected document TF-IDF and fitted-analyzer query TF-IDF",
                         "semantics": "retrieval context and co-occurrence, not entailment"},
                "context": {"scientific_native_neighbor_limit": 8, "neighbor_order": "original node ID",
                            "own_text_truncated": False, "authors_receive_context": False},
                "layers": {name: layer.describe() for name, layer in self.layers.items()}}
