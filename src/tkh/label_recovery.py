"""Explicit offline migration of saved v3 prompt-only NVIDIA responses.

This compatibility path never runs during live response validation. Original
files and scientific text are retained; imported rows record the old format.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .gemini_labels import (batch_prompt,generation_prompt,digest,write,_response_text,extract_label,
    extract_batch_labels,label_length_warnings,label_quality_warnings,_parse_explanation,_export)

LEGACY_PREFIX='Here are the groups formatted as requested:\n\n'


def recover_legacy_labels(source,out):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError('Legacy and new output directories must be separate')
    previous=json.loads((source/'protocol.json').read_text());current=json.loads((out/'protocol.json').read_text())
    if (previous['schema_version']!='api-labels-v3' or previous['provider']!='nvidia'
            or previous['generation'].get('response_format') or current['provider']!='nvidia'
            or current['generation'].get('response_format') not in {'json_schema','json_object'}):
        raise ValueError('Expected a legacy prompt-only run and a prepared structured-output run')
    for key in ('source_run','model','provider','jobs_sha256','length_policy'):
        if previous.get(key)!=current.get(key):raise ValueError('Legacy input/model mismatch: '+key)
    jobs=json.loads((out/'prompts.json').read_text());old_jobs=json.loads((source/'prompts.json').read_text())
    if digest(jobs)!=current['jobs_sha256'] or digest(old_jobs)!=current['jobs_sha256']:
        raise ValueError('Legacy or current evidence changed')
    if list((out/'completed').glob('*.json')):raise ValueError('Recovery requires a prepared output with no completed labels')
    lookup={(j['year'],j['level'],j['id']):j for j in jobs}
    completed={};copies={};manifest=[]
    for response_path in sorted((source/'responses').glob('*.json')):
        sha=response_path.name.split('-',1)[0];request_path=source/'requests'/f'{sha}.json'
        request=json.loads(request_path.read_text());response=json.loads(response_path.read_text())
        if hashlib.sha256(request['prompt'].encode()).hexdigest()!=sha or request['prompt_sha256']!=sha:
            raise ValueError('Legacy request hash mismatch')
        ids=request['group_ids']
        if not ids or len(set(ids))!=len(ids):raise ValueError('Legacy request has invalid group IDs')
        try:batch=[lookup[(request['year'],request['level'],gid)] for gid in ids]
        except KeyError:raise ValueError('Legacy request has unknown group IDs') from None
        if request['prompt']!=batch_prompt(batch):raise ValueError('Legacy request evidence differs from current evidence')
        raw=_response_text(response);removed=raw.startswith(LEGACY_PREFIX)
        if removed:raw=raw[len(LEGACY_PREFIX):]
        normalized=deepcopy(response);normalized['choices'][0]['message']['content']=raw
        policy=current['length_policy']
        parsed=(extract_batch_labels(normalized,set(ids),length_policy=policy) if len(batch)>1 else
                {ids[0]:extract_label(normalized,length_policy=policy)})
        request_rel=f'requests/legacy-{request_path.name}';response_rel=f'responses/legacy-{response_path.name}'
        copies[request_rel]=request_path.read_bytes();copies[response_rel]=response_path.read_bytes()
        response_hash=hashlib.sha256(copies[response_rel]).hexdigest()
        for job in batch:
            token=digest([job['year'],job['level'],job['id']])
            if token in completed:raise ValueError('Multiple legacy responses for a group; explicit selection is required')
            label,text=parsed[job['id']]
            row={k:job[k] for k in ('year','id','level','member_count','prompt_sha256')}
            row.update(**label,raw_output=text,length_warnings=label_length_warnings(label),
                quality_warnings=label_quality_warnings(label),
                requested_model=current['model'],provider='nvidia',model_version=response.get('model'),
                usage=response.get('usage',{}),usage_scope='shared request' if len(batch)>1 else 'single request',
                batch_prompt_sha256=sha,request_file=request_rel,response_file=response_rel,
                response_sha256=response_hash,response_format='legacy-prompt-only-imported',
                legacy_prefix_removed=removed)
            completed[token]=row
        manifest.append({'response_file':str(response_path.relative_to(source)),
            'response_sha256':response_hash,'request_sha256':sha,'group_ids':ids,'prefix_removed':removed})
    # Complete validation precedes any import; no partial acceptance of a bad batch.
    for name,data in copies.items():
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for token,row in completed.items():write(out/'completed'/f'{token}.json',row)
    result={'source':str(source),'imported_labels':len(completed),'network_requests':0,
        'original_protocol_sha256':hashlib.sha256((source/'protocol.json').read_bytes()).hexdigest(),
        'policy':'Exact inputs/model; only the observed legacy prefix may be removed for parsing; original response bytes retained.',
        'responses':manifest}
    write(out/'legacy_import.json',result)
    plan=json.loads((out/'batch_plan.json').read_text())
    _export(Path(current['source_run']),out,jobs,completed,'partial',planned_requests=plan['planned_requests'])
    return result


def recover_saved_response(out,response_path):
    """Commit an exact, already saved structured reply after full validation."""
    out=Path(out).resolve();response_path=Path(response_path).resolve()
    if not response_path.is_relative_to(out/'responses'):
        raise ValueError('Response must belong to this output directory')
    protocol=json.loads((out/'protocol.json').read_text());jobs=json.loads((out/'prompts.json').read_text())
    if digest(jobs)!=protocol['jobs_sha256']:raise ValueError('Saved evidence changed')
    sha=response_path.name.split('-',1)[0]
    if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Invalid response filename')
    request_path=out/'requests'/f'{sha}.json';request=json.loads(request_path.read_text())
    lookup={(j['year'],j['level'],j['id']):j for j in jobs}
    ids=request['group_ids']
    if not ids or len(set(ids))!=len(ids):raise ValueError('Invalid requested group IDs')
    try:batch=[lookup[(request['year'],request['level'],gid)] for gid in ids]
    except KeyError:raise ValueError('Unknown requested group ID') from None
    if (request['prompt_sha256']!=sha or hashlib.sha256(request['prompt'].encode()).hexdigest()!=sha
            or request['prompt']!=generation_prompt(batch,provider=protocol['provider'],attempt=request.get('attempt',1))):
        raise ValueError('Saved request differs from the exact group evidence')
    response=json.loads(response_path.read_text());nvidia=protocol['provider']=='nvidia'
    if nvidia and protocol['generation'].get('response_format')=='guided_json':
        from .nvidia_labels import nvidia_response_format
        expected=nvidia_response_format(ids if len(ids)>1 else None)['json_schema']['schema']
        if (request.get('decoding')!='guided_json' or request.get('guided_json')!=expected
                or 'response_format' in request):
            raise ValueError('Saved guided request constraint differs from this run')
    policy=protocol['length_policy']
    parsed=(extract_batch_labels(response,set(ids),length_policy=policy,envelope=nvidia) if len(batch)>1 else
            {ids[0]:extract_label(response,length_policy=policy,structured=nvidia)})
    completed={}
    for cache in (out/'completed').glob('*.json'):
        row=json.loads(cache.read_text());key=(row['year'],row['level'],row['id']);job=lookup[key]
        if row['prompt_sha256']!=job['prompt_sha256'] or row['requested_model']!=protocol['model']:
            raise ValueError('Cached group input/model mismatch')
        label=_parse_explanation(row['raw_output'],length_policy=policy)
        if any(row[k]!=v for k,v in label.items()):raise ValueError('Cached label differs from its recorded output')
        row['quality_warnings']=label_quality_warnings(label)
        completed[digest(list(key))]=row
    pending={};response_hash=hashlib.sha256(response_path.read_bytes()).hexdigest()
    for job in batch:
        token=digest([job['year'],job['level'],job['id']]);label,text=parsed[job['id']]
        if token in completed:
            if any(completed[token][k]!=v for k,v in label.items()):
                raise ValueError('Recovery would overwrite a different generated label')
            continue
        row={k:job[k] for k in ('year','id','level','member_count','prompt_sha256')}
        row.update(**label,raw_output=text,length_warnings=label_length_warnings(label),
            quality_warnings=label_quality_warnings(label),requested_model=protocol['model'],provider=protocol['provider'],
            model_version=response.get('modelVersion',response.get('model')),usage=response.get('usageMetadata',response.get('usage',{})),
            usage_scope='shared request' if len(batch)>1 else 'single request',batch_prompt_sha256=sha,
            request_file=str(request_path.relative_to(out)),response_file=str(response_path.relative_to(out)),
            response_sha256=response_hash,
            response_format=request.get('decoding',request.get('response_format',{}).get('type','gemini-responseSchema')),
            recovered_from_saved_response=True)
        pending[token]=row
    for token,row in pending.items():write(out/'completed'/f'{token}.json',row)
    completed.update(pending)
    result={'imported_labels':len(pending),'network_requests':0,'response_file':str(response_path.relative_to(out)),
        'response_sha256':response_hash,'request_sha256':sha,'group_ids':ids,
        'policy':'Exact original structured response; all IDs and text checks applied; review hints retained; no text edits.'}
    write(out/'response_recoveries'/f'{response_hash}.json',result)
    plan=json.loads((out/'batch_plan.json').read_text())
    _export(Path(protocol['source_run']),out,jobs,completed,'complete' if len(completed)==len(jobs) else 'partial',
        planned_requests=plan['planned_requests'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--source',type=Path);choice.add_argument('--response',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=recover_legacy_labels(args.source,args.out) if args.source else recover_saved_response(args.out,args.response)
    print(f'{result["imported_labels"]} labels recovered offline; originals preserved; no API requests')


if __name__=='__main__':main()
