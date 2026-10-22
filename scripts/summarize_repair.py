"""Generate the repair research summary from frozen measurements, never rerun/tune.

Run from any directory: python scripts/summarize_repair.py
Add another compatible assessment with --study v3=out/repair-study-v3/metrics.json.
This supplemental audit is not the unfinished submission article.
"""
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES=('accurate','vague','wrong','insufficient_evidence')
ARMS=(('BM25 N+S','bm25__native_sources'),
      ('BM25+E5 N+S','hybrid_no_tags'),
      ('BM25+E5+tags N+S+T','hybrid_graph'))


def join_blind(ratings,key):
    """Join blinded ratings by task ID, without converting insufficiency to wrong."""
    rows=ratings['tasks'];lookup={row['task_id']:row for row in rows}
    keys={row['task_id']:row for row in key}
    if len(lookup)!=len(rows) or len(keys)!=len(key) or set(lookup)!=set(keys):
        raise ValueError('Blind ratings and key must have identical unique task IDs')
    counts={condition:Counter({c:0 for c in CATEGORIES}) for condition in ('own','cross_group')}
    for tid,row in lookup.items():
        rating=row['rating'];condition=keys[tid]['condition']
        if rating not in CATEGORIES or condition not in counts:
            raise ValueError('Unknown blind rating or condition')
        counts[condition][rating]+=1
    return {condition:{**counter,'total':sum(counter.values())} for condition,counter in counts.items()}


def retrieval_cell(arms,name):
    track=arms[name]['tracks']['A'];score=track['metrics']['30']
    hits,total,recall=score['hits'],score['targets_all'],score['recall_all_micro']
    if (track['questions']!=14 or total!=63 or type(hits)is not int or not 0<=hits<=total
            or not math.isclose(recall,hits/total,abs_tol=1e-14)):
        raise ValueError('Primary retrieval must use all14 A questions and all63 targets consistently')
    return f'{hits}/{total} ({100*recall:.1f}%)'


def temporal_cell(row):
    p=row['alignment_significance']['p_greater_equal']
    bounds=row['source_delete_one']['range']
    ci='CI unavailable' if row['interval'] is None else f"conditional CI {row['interval']}"
    sensitivity='unavailable' if bounds is None else f'{bounds[0]:.4f}–{bounds[1]:.4f}'
    return f"{row['point_ari']:.4f}; p={p:g}; {ci}; source-deletion {sensitivity}"


def path_change_count(previous,current):
    """Count complete old paths, including metadata and changed coarse parents."""
    def paths(hierarchy):
        rows={}
        for level in hierarchy['levels']:
            if level['index']==1:
                for group in level['groups']:
                    for node in group['members']:
                        if node in rows:raise ValueError('Overlapping fine memberships')
                        rows[node]=(group['parent_id'],group['id'])
        return rows
    old,new=paths(previous),paths(current)
    if not old or not old.keys()<=new.keys():raise ValueError('Snapshot lost old paths')
    return sum(path!=new[node] for node,path in old.items())


def _table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                       *['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows]])


def summarize(root=ROOT,extra_studies=()):
    root=Path(root);sources={}
    def load(relative):
        path=root/relative;raw=path.read_bytes()
        sources[str(Path(relative))]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    historical=load('results/metrics.json')
    historical_hierarchies=[load(f'results/snapshots/{year}/hierarchy.json') for year in (2022,2023,2024,2025,2026)]
    studies=[('v1',load('out/repair-study-v1/metrics.json')),
             ('v2',load('out/repair-study-v2/metrics.json'))]
    names={name for name,_ in studies}
    for name,path in extra_studies:
        if not name or name in names:raise ValueError('Study names must be nonempty and unique')
        names.add(name);studies.append((name,load(path)))
    tree=load('out/repair-tree/run1/metrics.json')
    label_names=('balanced-evidence-2022','balanced-strong-2022-checked')
    labels=[]
    for name in label_names:
        prefix=f'out/repair-labels/{name}'
        rating=load(f'{prefix}/blind_ratings_codex.json');key=load(f'{prefix}/blind_rating_key.private.json')
        labels.append((name,join_blind(rating,key),rating.get('protocol',{})))
    methods=[('historical',None,historical['coherence'],historical['retrieval']['arms'])]
    for version,study in studies:
        for method,data in study['methods'].items():
            methods.append((f'{version}/{method}',data,data['coherence'],study['retrieval']['arms'][method]))
    lines=['# Repair results — generated evidence summary','',
        'This is a supplemental research audit, not a completed submission article. '
        'It preserves the earlier result as a baseline and compares independently measured repairs. '
        '**The current evidence does not establish full scientific completion or a downstream improvement.**','',
        'The hierarchy remains two aggregate levels with trusted display caps of **12 coarse / 50 fine groups**, '
        'followed by individual records. Construction uses own frozen E5-base-v2 vectors and native multiway incidence; '
        'no model fine-tuning or participant implementation is used. v1 retains the original initial builder. '
        'v2 starts with recursive native balanced bisections, then applies append-only, bounded, or bounded+novelty updates. '
        'Bounded updates permit at most 10% of old scientific-node count as complete old-path edits, including charged metadata donor paths. '
        'This limits editable old records, not global input validation or new-to-old similarity scans.','',
        'All construction settings are development decisions. No improvement is inferred from a newer model alone; '
        'evaluation questions, evaluator and corpus were already used during development. '
        'Predictions were frozen before this run read gold, which does not turn reused development data into an untouched test.','',
        '## Matched downstream comparison','',
        'Primary metric: micro Recall@30 across **all 63 target occurrences in all 14 A questions**, '
        'including unreachable references. Each pair changes only the hierarchy overlay; '
        'the original contexts, filters, channel weights and native/source graph inputs stay fixed. '
        'N = native hyperedges, S = shared source article, T = extractive tag incidence. '
        'The overlay combines ordinary walking with both hierarchy incidence levels; it is not coarse-first traversal.','']
    rows=[]
    for name,_,_,arms in methods:
        for label,key in ARMS:
            rows.append((name,label,retrieval_cell(arms,key+'::baseline'),retrieval_cell(arms,key+'::hierarchy')))
    lines += [_table(['Structure','Retrieval channels/layers','Without hierarchy','With hierarchy'],rows),'',
        'Exact arm keys are `bm25__native_sources`, `hybrid_no_tags`, and `hybrid_graph`, each suffixed '
        '`::baseline` / `::hierarchy`. BM25+E5 N+S uses lexical/dense weights .5/.25; '
        'the tagged variant additionally uses tags .25 and a different walking layer. '
        'These are different baselines and must not be conflated. Negative or unchanged pairs remain negative findings.','',
        '## Independent semantic grouping assessment','',
        'Construction uses E5; coherence uses the different Nemotron embedding family on the same own member strings. '
        'This removes same-encoder circularity, but the reused evaluator is not a held-out acceptance signal. '
        'The null has 20 type-stratified member permutations preserving group sizes and type composition. '
        'Each cell is observed cosine / excess over its own matched null. These values are not percentages of correct groups.','']
    rows=[]
    for name,_,coherence,_ in methods:
        first,last=coherence[0],coherence[-1]
        def cell(snapshot,level):
            x=snapshot['levels'][str(level)]
            if not math.isclose(x['observed']-x['null_mean'],x['excess'],abs_tol=1e-12):
                raise ValueError('Coherence excess differs from observed minus null')
            return f"{x['observed']:.4f} / {x['excess']:+.4f}"
        rows.append((name,cell(first,0),cell(first,1),cell(last,0),cell(last,1)))
    lines += [_table(['Structure','2022 coarse','2022 fine','2026 coarse','2026 fine'],rows),'',
        'Balanced construction removes tiny coarse partitions through declared size guards, but '
        'size balance alone does not establish a recognizable scientific idea. Both observed coherence and null excess '
        'must be read together; small changes in a reused evaluator are development evidence, not a significant superiority claim.','',
        '## Locality, identity and chronological testing','',
        'Entries below are ordered 2022→2023 / 2023→2024 / 2024→2025 / 2025→2026. '
        'Old paths include both the fine ID and coarse-parent ID. The repair assessor independently reconstructs paths from saved memberships, '
        'checks coverage/laminarity/trusted caps, recomputes endpoint counts and provenance, and compares saved old vectors and affinity blocks. '
        'It does not treat a trace flag as proof of unchanged membership.','']
    rows=[]
    old_trans=historical['construction']['transitions']
    old_traces=[r['growth_and_cost_trace'] for r in historical['construction']['snapshots'][1:]]
    rows.append(('historical',' / '.join(str(len(t['allowed_old_nodes'])) for t in old_traces),
        ' / '.join(str(path_change_count(a,b)) for a,b in zip(historical_hierarchies,historical_hierarchies[1:])),
        ' / '.join(f"{t['levels'][0]['ari_shared_old_nodes']:.4f}" for t in old_trans),
        ' / '.join(f"{t['levels'][1]['ari_shared_old_nodes']:.4f}" for t in old_trans)))
    for name,data,_,_ in methods[1:]:
        audits=[a['frontier'] for a in data['audits'] if 'frontier'in a]
        rows.append((name,' / '.join(str(a['allowed_old_nodes']) for a in audits),
            ' / '.join(str(a['actual_old_path_edits']) for a in audits),
            ' / '.join(f"{t['levels'][0]['point_ari']:.4f}" for t in data['temporal']),
            ' / '.join(f"{t['levels'][1]['point_ari']:.4f}" for t in data['temporal'])))
    lines += [_table(['Structure','Allowed old frontier','Actual old-path changes','Coarse ARI','Fine ARI'],rows),'',
        'Historical allowed-frontier counts refer to scientific records; the old method additionally permitted dependent metadata edits. '
        'Its actual movement is recomputed from complete saved paths, rather than the narrower diagnostic `moved_ids` list. '
        'Append-only ARI can be perfect because old assignments are frozen; significance against random matching '
        'does not prove useful adaptation. Native boundary contractions are checked against complete floating objective deltas; '
        'quotient endpoint multiplicities are preserved and consumed by coarse refinement. '
        'Stage-delta audits check recorded consistency; intermediate solver assignments are not exported for a separate complete replay.','',
        'Conditional significance below permutes current labels within old-node types for two fixed fitted partitions '
        '(1,999 permutations), not independent corpus histories. P-values are unadjusted per-level/per-transition references; '
        'they do not establish semantic quality, successful adaptation, or an unconditional temporal CI. '
        'Overlapping source papers collapse into too few components, so unsupported process CIs remain unavailable.','']
    rows=[]
    for name,data,_,_ in methods[1:]:
        ts=data['temporal'];ps=[r['alignment_significance']['p_greater_equal'] for t in ts for r in t['levels']]
        components=[t['levels'][0]['blocks']['count'] for t in ts]
        intervals=[r['interval'] for t in ts for r in t['levels']]
        sensitivities=[r['source_delete_one']['maximum_absolute_change'] for t in ts for r in t['levels']
                       if r['source_delete_one']['maximum_absolute_change'] is not None]
        rows.append((name,f'{min(ps):g}–{max(ps):g}', ' / '.join(map(str,components)),
            'unavailable' if all(x is None for x in intervals) else 'conditional only; see source',
            f'{max(sensitivities):.4f}' if sensitivities else 'unavailable'))
    lines += [_table(['Structure','Conditional p range','Source components per transition','Process CI','Max source-deletion ARI change'],rows),'',
        'Historical chronology reported exact ARI without significance or process intervals. '
        'The new source-deletion statistic is fixed-partition sensitivity, not an uncertainty interval. '
        'Full per-transition estimands, movement fractions, component weights, null draws and limitations remain in the source metrics.','',
        '## Rebuild robustness under edge deletion','',
        'Each fresh study removes floor(10% of initial native hyperedges), including provenance edges, '
        'for five fixed masks and rebuilds the corresponding initial method with the same construction seed. '
        'Intervals are descriptive 95% Student-t intervals for the conditional mean across masks. '
        'They are not chronological-process CIs and do not measure robustness of later updates.','']
    rows=[]
    for row in historical['historical_perturbation']['levels']:
        low,high=row['conditional_95_t_interval']
        rows.append(('historical saved','reported full population',row['level'],f"{row['mean']:.4f}",f'{low:.4f}–{high:.4f}'))
    for version,study in studies:
        p=study['fresh_initial_perturbations']
        for population in ('all','scientific'):
            for level in ('0','1'):
                s=p['summaries'][population][level]
                rows.append((version+' '+p.get('initial_strategy','legacy'),population,level,
                    f"{s['mean']:.4f}",f"{s['low']:.4f}–{s['high']:.4f}"))
    lines += [_table(['Initial builder','Population','Level','Mean ARI','Conditional interval'],rows),'',
        '## Labels: fresh blinded support and coverage assessment','',
        'Ratings are joined to the hidden key by unique task ID after rating, never by row order. '
        '“Accurate” is a supplied-evidence judgment; “insufficient evidence” remains separate from “wrong”. '
        'Where supplied, the conditions are the intended group (own) and a substituted other group (cross-group control). '
        'These are fresh-context Codex judgments from one automated judge, not human expertise or independent scientific truth. '
        'Both label pilots concern the initial balanced 2022 hierarchy, not every later snapshot.','']
    rows=[]
    for name,counts,_ in labels:
        for condition,c in counts.items():
            rows.append((name,condition,*[c[x] for x in CATEGORIES],c['total'],
                f"{c['wrong']}/{c['total']}" if c['total'] else 'not sampled',
                f"{c['insufficient_evidence']}/{c['total']}" if c['total'] else 'not sampled'))
    lines += [_table(['Pilot','Condition','Accurate','Vague','Wrong','Insufficient','Total','Wrong fraction','Insufficient fraction'],rows),'',
        'Wrong/total is one conservative supplied-evidence error fraction, not a complete over-claim rate. '
        'Insufficient cases may contain unsupported specifics and cannot be counted as faithful; vague statements '
        'can also fail to identify a useful theme. Cross-group controls may share broad terminology and are not guaranteed false facts. '
        'A nonempty gloss or a high support-existence NLI score does not establish theme coverage. '
        'Generation provenance and input/support splits stay in the original pilot artifacts; the summary does not attach these ratings to changed memberships.','',
        '## Actual coarse-first retrieval, with selected-leaf scoring','',
        'This separate policy first scores coarse group documents and E5 prototypes, then candidate child groups, '
        'and scores only leaves inside selected fine groups. Leaf BM25 retains global corpus weights through selected-row sparse scoring. '
        'E5 dot products and native/source walks are restricted to the retained region. '
        'It differs from the earlier seed-first route, which had already computed full node channels. '
        'The following table uses BM25+E5 N+S.','']
    rows=[]
    for name,arms in tree['summary'].items():
        for arm in ('bm25_e5::tree_2_8','bm25_e5::tree_4_16','bm25_e5::tree_8_32','bm25_e5::all'):
            m=arms[arm]
            if m['targets']!=63 or not math.isclose(m['recall'],m['hits']/63,abs_tol=1e-14):
                raise ValueError('Tree primary denominator differs')
            rows.append((name,arm.split('::')[1],f"{m['hits']}/63 ({100*m['recall']:.1f}%)",
                f"{m['mean_A_leaf_scores']:.1f}",f"{m['mean_A_dense_dots']:.1f}"))
    lines += [_table(['Structure','Beam','Recall@30','Mean evaluated leaves','Mean leaf E5 dots'],rows),'',
        f"Full-width tree probabilities match the flat control with maximum absolute error {tree['all_width_max_probability_error']:g}. "
        'Narrower routes save leaf evaluations but lose targets. No tested route beats the full-width baseline. '
        'Fixed branch width is not equal work: unbalanced legacy groups retain more leaves. '
        'Reported counts exclude offline group indexing; sparse row slicing reads selected-row vocabulary entries. '
        'Single-pass cached timing is not a controlled speed benchmark.','',
        '## What remains unresolved','',
        '- No measured advantage over the strongest flat/native+source retrieval control. A better null excess or stable identity is not a substitute.\n'
        '- No complete, independently supported scientific labels for all groups at every snapshot; initial blinded pilots show substantial insufficiency.\n'
        '- Novelty birth/merge transactions are calibrated heuristics, not verified discovery of new fields. '
        'v1/v2 have no old-topic redistribution split; a later experimental split implementation requires its own measured acceptance evidence.\n'
        '- No justified unconditional chronological-process confidence interval, and no measured later-update edge-deletion robustness.\n'
        '- No untouched final evaluation. The repaired research and negative results do not yet constitute a finished scientific submission.','',
        'The implemented repair addresses bounded changes and budget enforcement with meaningful counterexamples and real chronological checks. '
        'The remaining scientific failures need further method work and evidence, not a formatting-only report revision.','',
        '## Reproduction and evidence provenance','',
        'Generate this document without modifying inputs: `python scripts/summarize_repair.py`. '
        'Append another compatible study using `--study name=path/to/metrics.json`. '
        'Every numeric table here is produced by this generator from the listed measured files. '
        'This is a supplementary repair audit; the final submission still requires its prescribed `make_tables.py` integration. '
        'Each study retains its original method protocol/source digests and prediction locks; '
        '`out/repair-study-v1/source_snapshot/` and `out/repair-study-v2/source_snapshot/` preserve the execution sources. '
        'Current files are not substituted for those historical versions.','']
    for relative,digest in sources.items():
        lines.append(f'- [{relative}](../{relative}) — SHA-256 `{digest}`')
    lines += ['',f'Generator SHA-256: `{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}`.','']
    for relative,digest in sources.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Measured input changed during summary generation')
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',action='append',default=[],help='Additional compatible assessment name=path')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/REPAIR_RESULTS.md')
    args=parser.parse_args();studies=[]
    for value in args.study:
        if '='not in value:parser.error('--study requires name=path')
        studies.append(tuple(value.split('=',1)))
    document=summarize(extra_studies=studies)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(document,encoding='utf-8')
    print('Generated',args.output)


if __name__=='__main__':main()
