"""Evidence-separated explanatory labels; model scores are proxies, not truth.

Inputs are derived temporal views, never raw exports or evaluation questions.
Generation and support use disjoint source blocks. Shared-source members are
excluded so their names cannot silently bridge the split. This does not make
support independent of clustering: construction saw all member names, and both
checkers have general pretraining. Report that limitation explicitly.
"""
from collections import Counter
import hashlib
import json
import math
import re


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_label_input(graph, members, cutoff, seed=0, max_generation_members=24,
                      selection_policy='hashed_members'):
    """Make an exact reproducible source-block split; never use future evidence."""
    if graph.get('cutoff') != cutoff:
        raise ValueError('cutoff does not match temporal view')
    nodes = {n['id']: n for n in graph['nodes']}
    mids = sorted(set(members))
    if not mids or set(mids) - nodes.keys():
        raise ValueError('empty group or unknown member')
    for mid in mids:
        n = nodes[mid]
        if n.get('observed_at', cutoff + 1) > cutoff:
            raise ValueError('future member evidence')
        if not n.get('provenance', {}).get('articles'):
            raise ValueError('member without source provenance')
    sources = sorted({a for mid in mids for a in nodes[mid]['provenance']['articles']},
                     key=lambda a: (_digest([seed, a]), a))
    # Single-source groups can be described, but cannot pass an independent check.
    split = max(1, len(sources) // 2)
    gs, ss = set(sources[:split]), set(sources[split:])
    excluded = []
    sides = []
    for source_set in (gs, ss):
        ids = [mid for mid in mids if set(nodes[mid]['provenance']['articles']) <= source_set]
        edges = []
        claim_context = []
        for edge in graph['hyperedges']:
            if not (set(edge['members']) & set(ids)):
                continue
            p = edge.get('provenance', {})
            if edge.get('year', cutoff + 1) > cutoff or p.get('article_year', cutoff + 1) > cutoff:
                raise ValueError('future hyperedge evidence')
            # A claim adjacent to an actual group member is admissible evidence
            # even if that claim is assigned to a different topic. Its provenance
            # must remain wholly inside this evidence half, never merely share a name.
            if edge['relation_type'] == 'claims' and p.get('article_id') in source_set:
                linked = sorted(set(edge['members']) & set(ids))
                for nid in sorted(set(edge['members']) - set(ids)):
                    node = nodes[nid]
                    if node['type'] != 'claim':
                        continue
                    if node.get('observed_at', cutoff + 1) > cutoff:
                        raise ValueError('future incident claim evidence')
                    if set(node.get('provenance', {}).get('articles', [])) <= source_set:
                        claim_context.append({'node': node, 'edge_id': edge['id'],
                                              'supporting_member_ids': linked})
            # Facts supplied to either side must stay wholly in its evidence set.
            if p.get('article_id') in source_set and set(edge['members']) <= set(ids):
                edges.append(edge)
        sides.append({'source_ids': sorted(source_set), 'member_ids': ids,
                      'nodes': [nodes[mid] for mid in ids], 'hyperedges': sorted(edges, key=lambda e: e['id']),
                      'claim_context': claim_context})
    for mid in mids:
        if mid not in sides[0]['member_ids'] + sides[1]['member_ids']:
            excluded.append(mid)
    instructions = (
        'Describe the scientific theme of this group using ONLY the supplied evidence. '
        'Return one JSON object with label (2-8 words), gloss (one explanatory sentence), '
        'and status (coherent, mixed, or insufficient). Explain what connects the methods, '
        'tasks or findings; do not just list member names. Do not assert performance, '
        'causation or universal coverage unless stated in the evidence. A group can be mixed. '
        'If there is no defensible theme, use status insufficient and state that limitation. '
        'Do not import facts from your background knowledge.\nEVIDENCE:\n'
    )
    if type(max_generation_members) is not int or max_generation_members < 1:
        raise ValueError('max_generation_members must be positive')
    if selection_policy not in {'hashed_members','science_evidence'}:
        raise ValueError('Unknown generation selection policy')
    selected = sorted(sides[0]['nodes'], key=lambda n: (_digest([seed, n['id']]), n['id']))[:max_generation_members]
    if selection_policy=='science_evidence':
        # Keep authors/record containers from consuming a scientific explanation
        # prompt. Type round-robin prevents a large claim/method type swallowing
        # every role. Within-role TF-IDF centroid relevance has deterministic ties.
        from collections import defaultdict
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        candidates=[n for n in sides[0]['nodes'] if n['type'] not in ('author','article')]
        selected=[]
        if candidates:
            try:
                matrix=TfidfVectorizer(stop_words='english',ngram_range=(1,2)).fit_transform(
                    [n['surface_form'] for n in candidates])
                relevance=np.asarray(matrix @ matrix.mean(axis=0).T).ravel()
            except ValueError:
                relevance=np.zeros(len(candidates))
            by_type=defaultdict(list)
            for i,node in enumerate(candidates):by_type[node['type']].append((float(relevance[i]),node))
            for rows in by_type.values():rows.sort(key=lambda pair:(-pair[0],pair[1]['id']))
            kinds=sorted(by_type,key=lambda kind:(-len(by_type[kind]),kind))
            while len(selected)<max_generation_members and any(by_type.values()):
                for kind in kinds:
                    if by_type[kind] and len(selected)<max_generation_members:
                        selected.append(by_type[kind].pop(0)[1])
    selected_ids = {n['id'] for n in selected}
    payload = {'nodes': selected, 'hyperedges': [e for e in sides[0]['hyperedges']
                                               if set(e['members']) <= selected_ids]}
    if selection_policy=='science_evidence':
        context=[c for c in sides[0]['claim_context'] if set(c['supporting_member_ids']) & selected_ids]
        context.sort(key=lambda c:(-len(set(c['supporting_member_ids']) & selected_ids),c['node']['id'],c['edge_id']))
        unique={}
        for claim in context:
            unique.setdefault(claim['node']['id'],{**claim,'supporting_member_ids':sorted(
                set(claim['supporting_member_ids']) & selected_ids)})
        payload['claim_context']=list(unique.values())[:12]
    prompt = instructions + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {'cutoff': cutoff, 'seed': seed, 'member_ids': mids,
            'generation': sides[0], 'support': sides[1],
            'excluded_shared_member_ids': excluded,
            'support_available': bool(sides[1]['member_ids']),
            'prompt_member_ids': sorted(selected_ids),
            'generation_members_omitted_from_prompt': len(sides[0]['member_ids']) - len(selected),
            'selection_policy':selection_policy,
            'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'limitations': ['Source-disjoint support is not independent of construction inputs.',
                            'No full article text; only supplied extracted nodes and relations.',
                            'Shared-source members excluded; assessable subset is reported.']}


TEMPLATE_GLOSS_PHRASES=('representative member names','various scientific','related topics','collection of methods')


def parse_label_output(raw, *, enforce_length=True,enforce_template=True,enforce_sentence=True):
    """Validate output without repairing malformed/vacuous text into a success."""
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)[:-3].strip()
    try:
        x = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('not a JSON label') from exc
    if not isinstance(x, dict) or set(x) != {'label', 'gloss', 'status'}:
        raise ValueError('label requires exactly label, gloss and status')
    if not all(isinstance(x[k], str) and x[k].strip() for k in x):
        raise ValueError('empty or nontext label field')
    if x['status'] not in {'coherent', 'mixed', 'insufficient'}:
        raise ValueError('unknown label status')
    if enforce_length and (not 2 <= len(x['label'].split()) <= 8 or not 6 <= len(x['gloss'].split()) <= 65):
        raise ValueError('label/gloss length does not explain a theme')
    if enforce_sentence and len(re.findall(r'[.!?](?:\s|$)', x['gloss'])) != 1:
        raise ValueError('gloss must be one sentence')
    if enforce_template and any(p in x['gloss'].lower() for p in TEMPLATE_GLOSS_PHRASES):
        raise ValueError('vacuous list or template gloss')
    return {k: v.strip() for k, v in x.items()}


def calibrate_checker(positive_scores, negative_scores):
    """Reject unusable checker controls; thresholds fixed before label assessment.

    Synthetic paraphrase positives and incompatible negatives are a wiring/domain
    sensitivity check, not a human-labeled calibration set. They gate usability,
    never estimate thresholds from four artificial examples. Conservative separated
    bands avoid counting an undecidable claim as correct.
    """
    pos, neg = list(positive_scores), list(negative_scores)
    for p in pos + neg:
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('checker score must be a finite probability')
    usable = bool(pos and neg and min(pos) > max(neg) and min(pos) >= .8 and max(neg) <= .2)
    return {'usable': usable, 'positive_scores': pos, 'negative_scores': neg,
            'support_threshold': .8,
            'unsupported_threshold': .2,
            'control_type': 'synthetic sensitivity controls; not human calibration'}


def assess_support(gloss, scores, calibration, *, generation_status='coherent'):
    """Classify support on source-block documents using conservative bands.

    A maximum across documents tests existence of a supporting held-out source,
    not universal group coverage. Mixed/insufficient generation cannot be called
    supported. Unsupported means low model support, not proven factual falsehood.
    """
    scores = list(scores)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError('invalid support score')
    best = max(scores) if scores else None
    status = 'insufficient_evidence'
    if calibration['usable'] and best is not None and generation_status == 'coherent':
        if best >= calibration['support_threshold']:
            status = 'supported'
        elif best <= calibration['unsupported_threshold']:
            status = 'unsupported'
    return {'gloss': gloss, 'status': status, 'scores': scores, 'best_score': best,
            'generation_status': generation_status,
            'estimand': 'existence of supporting held-out source; not coverage or factual truth'}


def summarize_support(records):
    counts = Counter(x['status'] for x in records)
    total = len(records)
    assessable = counts['supported'] + counts['unsupported']
    return {'total': total, 'counts': dict(counts), 'assessable': assessable,
            'unsupported_fraction_all': counts['unsupported'] / total if total else None,
            'unsupported_fraction_assessable': counts['unsupported'] / assessable if assessable else None,
            'insufficient_fraction': counts['insufficient_evidence'] / total if total else None,
            'is_human_verified_overclaim_rate': False}


def support_documents(label_input):
    """Return individual held-out claim sentences with source and node provenance.

    No name-only list is an entailment premise. Each sentence is independently
    checked, never silently truncated or treated as full article evidence.
    """
    result = []
    for node in label_input['support']['nodes']:
        if node['type'] != 'claim':
            continue
        result.append({'text': node['surface_form'], 'claim_node_ids': [node['id']],
                       'source_ids': node['provenance']['articles']})
    for context in label_input['support'].get('claim_context', []):
        node = context['node']
        if any(node['id'] in d['claim_node_ids'] for d in result):
            continue
        result.append({'text': node['surface_form'], 'claim_node_ids': [node['id']],
                       'source_ids': node['provenance']['articles'],
                       'supporting_member_ids': context['supporting_member_ids'],
                       'edge_ids': [context['edge_id']]})
    return result


def cross_group_controls(records):
    """Pair each label with another group's evidence, excluding shared sources.

    A deranged label is a specificity control, not automatically a false claim:
    broad themes can legitimately cover two groups. Never score these as gold.
    """
    result = []
    for rec in sorted(records, key=lambda x: x['id']):
        own = set(rec['input']['generation']['source_ids'] + rec['input']['support']['source_ids'])
        for donor in sorted(records, key=lambda x: x['id']):
            other = set(donor['input']['generation']['source_ids'] + donor['input']['support']['source_ids'])
            if donor['level'] == rec['level'] and donor['id'] != rec['id'] and donor.get('label') and not own & other:
                result.append({'recipient_id': rec['id'], 'donor_id': donor['id'],
                               'gloss': donor['label']['gloss'], 'is_known_false': False})
                break
    return result


def extract_model_label(raw):
    """Extract a complete fenced object, explicitly recording extra model prose.

    This is output-format tolerance, never semantic repair. Multiple objects,
    incomplete JSON, vague templates and extra fields remain errors.
    """
    try:
        return {'label': parse_label_output(raw), 'format_compliant': True,
                'ignored_trailing_text': ''}
    except ValueError:
        match = re.fullmatch(r'\s*```(?:json)?\s*\n(.*?)\n```\s*(.+)', raw, flags=re.S)
        if match is None or any(c in match[2] for c in ('{', '}', '```')):
            raise ValueError('malformed or ambiguous structured output')
        return {'label': parse_label_output(match[1]), 'format_compliant': False,
                'ignored_trailing_text': match[2].strip()}


def attach_generated_labels(expected, generated):
    """Attach support inputs only after generation over identical safe prompts."""
    from copy import deepcopy
    def mapping(rows):
        keys=[(r['id'],r['level']) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate generated group')
        return dict(zip(keys,rows,strict=True))
    targets,outputs=mapping(expected),mapping(generated)
    if set(targets)!=set(outputs):
        raise ValueError('Generated group set differs from requested groups')
    result=[]
    for key,record in targets.items():
        output=outputs[key]
        if output['prompt_sha256']!=record['input']['prompt_sha256']:
            raise ValueError('Generation prompt hash differs before support attachment')
        item=deepcopy(record);item['raw_output']=output['raw_output']
        item.update(extract_model_label(output['raw_output']))
        result.append(item)
    return result
