"""Authenticate full T5 outputs, prepare a blind sample and publish labelled views."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from copy import deepcopy

from .gemini_labels import digest,write
from .label_repair import _labels,_freeze
from .scientific_labels import support_documents
from .validation import validate_run


def resolve_judge_model(labels,requested=None):
    protocol=Path(labels)/'audit/api_check/protocol.json'
    from .label_judging import DEFAULT_MODEL
    model=requested or (json.loads(protocol.read_text())['judge_model'] if protocol.exists() else DEFAULT_MODEL)
    if model.startswith('google/gemini-'):raise ValueError('Gemini is disabled; archive that audit before selecting an NVIDIA checker')
    return model


def prepare_audit(labels):
    labels=Path(labels);protocol=json.loads((labels/'protocol.json').read_text())
    rows=json.loads((labels/'prompts.json').read_text())
    if digest(rows)!=protocol['jobs_sha256']:raise ValueError('Label input records changed')
    jobs={(j['year'],j['level'],j['id']):j for j in rows}
    outputs=_labels(labels,jobs)
    if not jobs or set(outputs)!=set(jobs):raise ValueError('Complete labels are required; some groups are missing')
    source=Path(protocol['source_run']);strata={}
    # A review ZIP preserves protocol bytes and hashes, but relocates its run.
    if not source.is_dir() and (labels.parent/'hierarchy').is_dir():source=labels.parent/'hierarchy'
    expected=set()
    for directory in (source/'snapshots').iterdir():
        if directory.is_dir() and directory.name.isdigit():
            h=json.loads((directory/'hierarchy.json').read_text())
            expected.update((int(directory.name),level['index'],g['id'])
                            for level in h['levels'][:2] for g in level['groups'])
    if set(jobs)!=expected:raise ValueError('Complete snapshot/group coverage is required; a selected pilot is incomplete')
    for key,j in jobs.items():
        if digest(json.loads((source/'snapshots'/str(j['year'])/'input.json').read_text()))!=j['input_sha256']:
            raise ValueError('Source graph changed since labelling')
        if digest(json.loads((source/'snapshots'/str(j['year'])/'hierarchy.json').read_text()))!=j['hierarchy_sha256']:
            raise ValueError('Source hierarchy changed since labelling')
        strata.setdefault(key[:2],[]).append(key)
    selected=[]
    for stratum,keys in sorted(strata.items()):
        selected.extend(sorted(keys,key=lambda k:digest([20261022,*k]))[:2])
    packet=[];key_rows=[]
    for i,key in enumerate(selected):
        j=jobs[key];label=outputs[key];evidence=j['evidence']
        names=[{**n,'evidence_role':'visible_member_name'} for n in evidence['members'] if n['type']!='claim']
        claims=[{**n,'evidence_role':'withheld_member_claim'} for n in j['input']['support']['nodes'] if n['type']=='claim']
        packet.append({'task_id':f'blind-{i:03d}','cutoff':j['year'],'level':j['level'],
            'label':label['label'],'gloss':label['gloss'],'held_out_members':names+claims,
            'held_out_claims':support_documents(j['input']),
            'assessment_design':{'names_visible_to_generator':True,
                'claim_texts_withheld_from_generator':True,
                'support_claim_documents':len(support_documents(j['input']))}})
        key_rows.append({'task_id':f'blind-{i:03d}','year':key[0],'level':key[1],'group_id':key[2]})
    design={'schema_version':'final-label-audit-v1','sample_groups':len(packet),'variants':[],
        'sample_role':'Two fixed hash-selected groups per cutoff/level; coarse groups oversampled; sample rates only.',
        'limitations':['Visible names overlap construction/label inputs; their scope audit is circular and is not held-out factual support.',
            'Only support claim sentences provide withheld-text evidence; groups without them are reported separately.',
            'Model-based ratings are not expert truth; sensitivity controls must pass before interpreting the rates.'],
        'sample_with_claim_evidence':sum(bool(p['held_out_claims']) for p in packet),
        'total_labelled_groups':len(jobs),'packet_sha256':digest(packet)}
    _freeze(labels/'audit/diagnostic_packet.json',packet)
    _freeze(labels/'audit/diagnostic_key.private.json',key_rows)
    _freeze(labels/'audit/review_design.json',design)
    return design


def publish_labels(run):
    run=Path(run);labels=run/'labels';design=prepare_audit(labels)
    source=run/'hierarchy';destination=run/'labelled_hierarchy'
    jobs={(j['year'],j['level'],j['id']):j for j in json.loads((labels/'prompts.json').read_text())}
    outputs=_labels(labels,jobs)
    with tempfile.TemporaryDirectory(prefix='.labelled-',dir=run) as tmp:
        pending=Path(tmp)/'hierarchy';pending.mkdir()
        protocol=json.loads((source/'protocol.json').read_text())
        write(pending/'protocol.json',protocol)
        for year in protocol['years']:
            d=pending/'snapshots'/str(year);d.mkdir(parents=True)
            for name in ('input.json','label_inputs.json','temporal_events.json','coarsened_hyperedges.json','trace.json'):
                shutil.copy2(source/'snapshots'/str(year)/name,d/name)
            original=json.loads((source/'snapshots'/str(year)/'hierarchy.json').read_text())
            named=json.loads((labels/'snapshots'/str(year)/'hierarchy.json').read_text())
            for level in named['levels'][:2]:
                for group in level['groups']:
                    row=outputs[(year,level['index'],group['id'])]
                    if (group['label']!=row['label'] or group['gloss']!=row['gloss']
                            or group['generation_prompt_sha256']!=row['prompt_sha256']):
                        raise ValueError('Exported label text differs from authenticated API response')
            def structure(h):
                return [(l['index'],g['id'],g['parent_id'],g['members']) for l in h['levels'] for g in l['groups']]
            if structure(original)!=structure(named):raise ValueError('Labelling changed the hierarchy structure')
            write(d/'hierarchy.json',named)
        errors=validate_run(pending)
        if errors:raise ValueError('Labelled hierarchy validation failed: '+'; '.join(errors))
        if destination.exists():
            for file in pending.rglob('*.json'):
                target=destination/file.relative_to(pending)
                if not target.exists() or json.loads(target.read_text())!=json.loads(file.read_text()):
                    raise ValueError('Published labelled hierarchy changed')
        else:pending.rename(destination)
    original_metrics=json.loads((run/'metrics.json').read_text())
    metrics=deepcopy(original_metrics)
    summary_path=labels/'audit/api_check/summary.json'
    check=None
    if summary_path.exists():
        from .label_judging import run_judging
        judge_protocol=json.loads((summary_path.parent/'protocol.json').read_text())
        check=run_judging(labels,summary_path.parent,prepare_only=True,
            model=judge_protocol['judge_model'],controls_first=judge_protocol['controls_first'],
            max_attempts=judge_protocol['max_attempts'],empty_evidence_rule=judge_protocol.get('empty_evidence_rule',False))
    metrics['labels']={'generated':design['total_labelled_groups'],'audit_design':design,
        'generator_model':json.loads((labels/'protocol.json').read_text())['model'],
        'judge_model':judge_protocol['judge_model'] if check else None,
        'judge':check,'published_hierarchy':'labelled_hierarchy',
        'annotation_replay':json.loads((labels/'replayed_from.json').read_text()) if (labels/'replayed_from.json').exists() else None,
        'rates_interpretable':bool(check and check['control_gate_passed'] and check['completed']==check['planned_requests'])}
    if check:
        metrics['not_measured']=[s for s in metrics['not_measured'] if s!='Independent label faithfulness and overclaim rate']
        if not metrics['labels']['rates_interpretable']:
            issue='Label judge failed controls or is unfinished; its overclaim rate is not reliable.'
            if issue not in metrics['not_measured']:metrics['not_measured'].append(issue)
    # Re-authentication is read-only when values agree. Another stage may have
    # canonicalized JSON key order; rewriting it would break report byte hashes.
    if metrics != original_metrics:write(run/'metrics.json',metrics)
    return metrics['labels']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('out/cpu'))
    p.add_argument('--check',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--replay-only',action='store_true',help='Require a complete saved audit; never call a model')
    p.add_argument('--judge-model',help='Explicit checker; resume otherwise retains the archived model')
    args=p.parse_args()
    try:
        design=prepare_audit(args.run/'labels')
        print(f'{design["total_labelled_groups"]} authenticated labels; {design["sample_groups"]} blind sample groups; '
              f'{design["sample_with_claim_evidence"]} have withheld claim evidence.',flush=True)
        if args.prepare_only:return 0
        if args.check:
            from .label_judging import run_judging
            path=args.run/'labels/audit/api_check/protocol.json'
            rule=json.loads(path.read_text()).get('empty_evidence_rule',False) if path.exists() else True
            result=run_judging(args.run/'labels',args.run/'labels/audit/api_check',controls_first=True,workers=4,
                model=resolve_judge_model(args.run/'labels',args.judge_model),empty_evidence_rule=rule,
                prepare_only=args.replay_only)
            stopped_on_controls=(not result['control_gate_passed'] and
                bool(result['controls']) and all(c.get('observed') is not None for c in result['controls']))
            if args.replay_only and result['completed']!=result['planned_requests'] and not stopped_on_controls:
                raise ValueError('Archived label audit is incomplete; fresh API calls require an explicit live run')
            print('Judge sensitivity controls passed:',result['control_gate_passed'],flush=True)
        publish_labels(args.run)
    except (ValueError,RuntimeError,OSError) as exc:
        p.exit(1,str(exc)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
