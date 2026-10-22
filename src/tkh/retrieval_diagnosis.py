"""Read-only attribution of saved retrieval failures; never used by a predictor."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def verify_predictions(directory):
    directory=Path(directory)
    lock=json.loads((directory/'prediction_lock.json').read_text())
    raw=(directory/'predictions.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=lock['predictions_sha256']:
        raise ValueError('Saved prediction hash changed')
    return json.loads(raw)


def _positions(ranking):
    if len(ranking)!=len(set(ranking)):raise ValueError('Duplicate IDs in saved ranking')
    return {n:i+1 for i,n in enumerate(ranking)}


def _first(positions,ids):
    return min((positions[n] for n in ids if n in positions),default=None)


def compare_rankings(targets,before,after,*,k=30):
    if type(k)is not int or k<1:raise ValueError('Expected a positive integer cutoff')
    a,b=_positions(before),_positions(after);rows=[]
    for target in targets:
        old,new=_first(a,target['node_ids']),_first(b,target['node_ids'])
        old_hit=old is not None and old<=k;new_hit=new is not None and new<=k
        change='gained' if new_hit and not old_hit else 'lost' if old_hit and not new_hit else 'unchanged'
        rows.append({**target,'before_rank':old,'after_rank':new,'before_hit':old_hit,
                     'after_hit':new_hit,'change':change})
    return rows,{'targets':len(rows),'before_hits':sum(r['before_hit'] for r in rows),
        'after_hits':sum(r['after_hit'] for r in rows),'gained':sum(r['change']=='gained' for r in rows),
        'lost':sum(r['change']=='lost' for r in rows)}


def diagnose_route(targets,baseline,ranking,hierarchy,trace,nodes,*,k=30):
    groups={l['index']:{g['id']:g for g in l['groups']}
            for l in hierarchy['levels'] if l['index'] in (0,1)}
    if set(groups)!={0,1}:raise ValueError('Expected two hierarchy levels')
    owners={}
    for level,current in groups.items():
        owner={n:gid for gid,g in current.items() for n in g['members']}
        if set(owner)!=set(nodes) or sum(len(g['members']) for g in current.values())!=len(nodes):
            raise ValueError('Hierarchy does not partition graph records')
        owners[level]=owner
    if any(g['parent_id'] not in groups[0] or any(owners[0][n]!=g['parent_id'] for n in g['members'])
           for g in groups[1].values()):raise ValueError('Nonlaminar hierarchy')
    coarse=set(trace['selected_coarse_ids']);fine=set(trace['selected_fine_ids'])
    if coarse-set(groups[0]) or fine-set(groups[1]):raise ValueError('Unknown selected group')
    if any(groups[1][g]['parent_id'] not in coarse for g in fine):
        raise ValueError('Selected child has unselected parent')
    retained={n for g in fine for n in groups[1][g]['members']}
    if set(trace['retained_ids'])!=retained or len(trace['retained_ids'])!=len(retained):
        raise ValueError('Trace retained IDs differ from selected group members')
    if set(ranking)-retained or set(baseline)-set(nodes):raise ValueError('Saved ranking outside selected graph')
    rows,pair=compare_rankings(targets,baseline,ranking,k=k)
    for row in rows:
        ids=row['node_ids']
        if set(ids)-set(nodes):raise ValueError('Target matching references unknown graph records')
        if not ids:stage='unmatched'
        elif not any(owners[0][n] in coarse for n in ids):stage='coarse_rejected'
        elif not any(owners[1][n] in fine for n in ids):stage='fine_rejected'
        elif row['after_hit']:stage='hit'
        else:stage='retained_not_top_k'
        row.update(stage=stage,coarse_group_ids=sorted({owners[0][n] for n in ids}),
                   fine_group_ids=sorted({owners[1][n] for n in ids}),
                   retained_node_ids=sorted(set(ids)&retained))
    stages=dict(Counter(r['stage'] for r in rows))
    summary={'targets':len(rows),'matched':sum(bool(r['node_ids']) for r in rows),
             'hits':pair['after_hits'],'baseline_hits':pair['before_hits'],
             'gained':pair['gained'],'lost':pair['lost'],'stage_counts':stages,
             'retained_records':len(retained)}
    if sum(stages.values())!=len(targets):raise AssertionError('Target accounting lost a record')
    return rows,summary
