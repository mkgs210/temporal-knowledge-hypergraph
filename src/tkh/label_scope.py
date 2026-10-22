"""Full topic scope with separately withheld claim text, without neural inference.

Member names are deliberately visible: an unseen material cannot constrain a
label's scope. Only claim-type sentences on the support side remain withheld.
This is NOT a source-disjoint or out-of-sample assessment of the visible names.
"""
from collections import Counter, defaultdict, deque


def topic_representation(evidence):
    """Keywords and four representative records, with TF-IDF/cosine and MMR.

    This follows the notebook's keywords + representative documents prompt
    pattern, not BERTopic's clustering or its distinct c-TF-IDF weighting.
    Keep the complete member scope alongside these examples. Adjacent claims
    never become representative members or determine the keywords.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    members = evidence['members'] + evidence.get('generation_claims', [])
    unique = {}
    for node in sorted(members, key=lambda n: n['id']):
        text = ' '.join(node['surface_form'].split()).casefold()
        unique.setdefault(text, node)
    nodes = list(unique.values())
    keywords = []
    ranked = list(range(len(nodes)))
    if nodes:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
        try:
            matrix = vectorizer.fit_transform([n['surface_form'] for n in nodes])
        except ValueError as exc:
            if 'empty vocabulary' not in str(exc):
                raise
        else:
            center = np.asarray(matrix.mean(axis=0))
            vocabulary = vectorizer.get_feature_names_out()
            weights = center.ravel()
            terms = sorted(range(len(vocabulary)), key=lambda i: (-weights[i], vocabulary[i]))
            keywords = [str(vocabulary[i]) for i in terms[:10]]
            scores = cosine_similarity(matrix, center).ravel()
            # Standard MMR with equal relevance/diversity weights: centrality
            # alone can return four near duplicates.
            remaining = set(ranked)
            ranked = []
            redundancy = np.zeros(len(nodes))
            for _ in range(min(4, len(nodes))):
                best = min(remaining, key=lambda i: (
                    -(.5 * scores[i] - .5 * redundancy[i]), nodes[i]['id']))
                ranked.append(best)
                remaining.remove(best)
                redundancy = np.maximum(redundancy, cosine_similarity(matrix, matrix[best]).ravel())
    return {'keywords': keywords, 'documents': [nodes[i] for i in ranked[:4]],
            'members_used': len(members), 'unique_texts': len(nodes),
            'method': 'Within-group TF-IDF (1–3 grams), mean term weights; cosine/MMR diversity=0.5; not c-TF-IDF.',
            'document_role': 'Representative extracted member records, not full articles; complete scope follows.'}


def _compact(node):
    return {k: node[k] for k in ('id', 'type', 'surface_form')} | {
        'source_ids': sorted(node['provenance']['articles'])}


def _interleave(nodes):
    """Keep every record; alternate source/type blocks to avoid source-first order."""
    blocks = defaultdict(list)
    for node in sorted(nodes, key=lambda n: n['id']):
        blocks[(min(node['provenance']['articles']), node['type'])].append(node)
    queues = [deque(blocks[key]) for key in sorted(blocks)]
    result = []
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.popleft())
    return result


def build_scope_evidence(graph, label_input, *, level, group_id):
    cutoff = label_input['cutoff']
    if graph['cutoff'] != cutoff:
        raise ValueError('Scope cutoff does not match temporal view')
    lookup = {n['id']: n for n in graph['nodes']}
    nodes = [lookup[mid] for mid in label_input['member_ids']]
    for node in nodes:
        if node.get('observed_at', cutoff + 1) > cutoff:
            raise ValueError('future scope member evidence')
        if not node.get('provenance', {}).get('articles'):
            raise ValueError('Scope member without source provenance')
    scientific = [n for n in nodes if n['type'] not in {'article', 'author'}]
    names = [n for n in scientific if n['type'] != 'claim']
    gen = set(label_input['generation']['source_ids'])
    claims = [n for n in scientific if n['type'] == 'claim'
              and set(n['provenance']['articles']) <= gen]
    member_claim_ids = {n['id'] for n in claims}
    visible = {n['id'] for n in names + claims}
    unique = {}
    for item in label_input['generation']['claim_context']:
        node = item['node']
        linked = set(item['supporting_member_ids']) & visible
        if not linked or node['id'] in member_claim_ids:
            continue
        if node.get('observed_at', cutoff + 1) > cutoff:
            raise ValueError('future scope context evidence')
        sources = set(node.get('provenance', {}).get('articles', []))
        if not sources or not sources <= gen:
            raise ValueError('Context claim is not generation-side evidence')
        row = unique.setdefault(node['id'], {'node': node, 'edges': set(), 'linked': set()})
        row['edges'].add(item['edge_id']); row['linked'].update(linked)
    # Adjacency supplies context, not extra members. Cap it explicitly and record
    # omissions, so a table attached to one node cannot dominate the explanation.
    contexts = sorted(unique.values(), key=lambda c: (-len(c['linked']), c['node']['id']))
    per_source = Counter(); chosen = []
    for item in contexts:
        sources = item['node']['provenance']['articles']
        if any(per_source[a] >= 2 for a in sources):
            continue
        chosen.append({'node': _compact(item['node']), 'edge_ids': sorted(item['edges']),
                       'supporting_member_ids': sorted(item['linked'])})
        per_source.update(sources)
        if len(chosen) == 12:
            break
    counts = Counter(a for n in scientific for a in n['provenance']['articles'])
    return {'cutoff': cutoff, 'level': 'coarse' if level == 0 else 'fine',
        'group_id': group_id, 'total_member_count': len(nodes),
        'scope_summary': {'scientific_member_count': len(scientific),
            'visible_nonclaim_members': len(names), 'visible_claim_members': len(claims),
            'withheld_claim_members': sum(n['type'] == 'claim' for n in scientific) - len(claims),
            'ignored_article_author_members': len(nodes) - len(scientific),
            'type_counts': dict(sorted(Counter(n['type'] for n in scientific).items()))},
        'source_profile': [{'source_id': a, 'scientific_member_count': counts[a]} for a in sorted(counts)],
        'members': [_compact(n) for n in _interleave(names)],
        'generation_claims': [_compact(n) for n in _interleave(claims)],
        'claim_context': chosen,
        'context_summary': {'available_unique_claims': len(unique), 'supplied_claims': len(chosen),
                            'omitted_unique_claims': len(unique) - len(chosen)},
        'evidence_roles': {
            'members': 'All nonclaim scientific member names, including support-source names; no sampling.',
            'generation_claims': 'Actual member claim sentences from generation sources only.',
            'claim_context': 'Adjacent claims, NOT extra members; do not let one example define the group.',
            'evaluation': 'Scope names are visible; only support claim texts remain withheld. '
                          'Coverage ratings are an audit, not unseen-source validation.'}}
