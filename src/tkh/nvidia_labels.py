"""NVIDIA hosted inference transport and a small user-run paired label pilot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time

from .gemini_labels import (ROOT,DEFAULT_RUN,DEFAULT_PROMPT,digest,write,
                           load_api_key,prepare_jobs,extract_label,label_length_warnings,LabelRequestError)
from .scientific_labels import support_documents

MODELS=('nvidia/nemotron-3.5-lightning-30b-a3b','nvidia/nemotron-3.5-super-vl-120b-a12b')


def nvidia_response_format(group_ids=None,*,model=None):
    """Canonical strict schema for generation and validation, for either model."""
    properties={'label':{'type':'string'},'gloss':{'type':'string'},
                'status':{'type':'string','enum':['coherent','mixed','insufficient']}}
    schema={'type':'object','properties':properties,'required':['label','gloss','status'],
            'additionalProperties':False}
    if group_ids is not None:
        ids=sorted(group_ids)
        if not ids or len(set(ids))!=len(ids) or any(not isinstance(i,str) for i in ids):
            raise ValueError('Batch group IDs must be nonempty, distinct strings')
        item={**schema,'properties':{**properties,'group_id':{'type':'string','enum':ids}},
              'required':['group_id','label','gloss','status']}
        schema={'type':'object','properties':{'groups':{'type':'array','items':item,
            'minItems':len(ids),'maxItems':len(ids)}},'required':['groups'],'additionalProperties':False}
    return {'type':'json_schema','json_schema':{'name':'group_labels','strict':True,'schema':schema}}


def nvidia_constraints(response_format=None,*,decoding='guided_json'):
    """Exact API constraint fields; ordinary JSON mode is never accepted."""
    if decoding not in {'guided_json','json_schema'}:raise ValueError('Unknown NVIDIA decoding mode')
    spec=nvidia_response_format() if response_format is None else response_format
    if (not isinstance(spec,dict) or spec.get('type')!='json_schema'
            or not isinstance(spec.get('json_schema'),dict)
            or spec['json_schema'].get('strict') is not True
            or not isinstance(spec['json_schema'].get('schema'),dict)):
        raise ValueError('NVIDIA generation requires a strict JSON Schema')
    return ({'guided_json':spec['json_schema']['schema']} if decoding=='guided_json' else {'response_format':spec})


def request_nvidia(model,prompt,key,*,thinking=False,response_format=None,decoding='guided_json',
                   http_client=None,timeout=90,max_tokens=512):
    """Official OpenAI SDK, schema constraints, one attempt, no format fallback."""
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',model):raise ValueError('Invalid NVIDIA model ID')
    constraints=nvidia_constraints(response_format,decoding=decoding)
    extra_body={'chat_template_kwargs':{'enable_thinking':thinking}}
    options={}
    if decoding=='guided_json':extra_body.update(constraints)
    else:options.update(constraints)
    try:
        from openai import OpenAI,APIStatusError,APIConnectionError
    except ImportError:
        raise RuntimeError('Install the label API extra: .venv/bin/python -m pip install -e ".[labels]"') from None
    try:
        with OpenAI(base_url='https://integrate.api.nvidia.com/v1',api_key=key,
                    timeout=timeout,max_retries=0,http_client=http_client) as client:
            response=client.chat.completions.with_raw_response.create(
                model=model,messages=[{'role':'user','content':prompt}],
                temperature=1.,top_p=.95,max_tokens=max_tokens,stream=False,
                extra_body=extra_body,**options)
            if response.status_code==202:
                raise LabelRequestError('NVIDIA request queued (HTTP202); no final completion received',retryable=True)
            completion=response.parse()
            if not hasattr(completion,'model_dump'):raise RuntimeError('NVIDIA returned a malformed completion')
            return completion.model_dump(mode='json')
    except APIStatusError as exc:
        body=exc.body
        message=(body.get('detail') or body.get('error')) if isinstance(body,dict) else body
        if isinstance(message,dict):message=message.get('message')
        safe=str(message or 'request failed').replace(key,'[REDACTED]')[:800]
        try:retry_after=float(exc.response.headers.get('Retry-After','nan'))
        except (TypeError,ValueError):retry_after=float('nan')
        if not math.isfinite(retry_after):retry_after=None
        raise LabelRequestError(f'NVIDIA HTTP {exc.status_code}: {safe}; no SDK retry or format fallback',
            retryable=exc.status_code in {408,409,429} or exc.status_code>=500,retry_after=retry_after) from None
    except APIConnectionError:
        raise LabelRequestError('NVIDIA network/timeout error; no SDK retry',retryable=True) from None


def select_pilot(jobs):
    """One median-size coarse, median-size fine and largest distinct fine group."""
    selected=[]
    for level in (0,1):
        rows=sorted((j for j in jobs if j['level']==level and j['evidence']['members']),
                    key=lambda j:(len(j['prompt']),j['id']))
        if not rows:continue
        selected.append(rows[len(rows)//2])
        if level==1:
            other=[j for j in rows if j['id']!=selected[-1]['id']]
            if other:selected.append(other[-1])
    if not selected:raise ValueError('No groups with generation evidence to compare')
    return selected


def _saved_label(out,row):
    """Inspect the recorded response, including text rejected for word counts."""
    if not row.get('response_file'):return None
    path=out/row['response_file']
    if (not path.resolve().is_relative_to(out.resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=row.get('response_sha256')):
        raise ValueError('Recorded response changed or is missing')
    response=json.loads(path.read_text())
    if row['status']=='valid':
        label,raw=extract_label(response)
        if label!=row['label'] or raw!=row['raw_output']:
            raise ValueError('Cached comparison label differs from its recorded response')
    try:return extract_label(response,length_policy='warn')[0]
    except ValueError:return None


def _request_stats(rows):
    received=sum(bool(r.get('response_file')) for r in rows)
    valid=sum(r['status']=='valid' for r in rows)
    times=[r['elapsed_seconds'] for r in rows]
    return {'completed_requests':len(rows),'responses_received':received,'valid_outputs':valid,
        'invalid_outputs':received-valid,'failed_requests':len(rows)-received,
        'median_request_seconds':statistics.median(times) if times else None,
        'min_request_seconds':min(times) if times else None,'max_request_seconds':max(times) if times else None}


def _outputs(out,jobs,models,records,*,report_out=None):
    report_out=out if report_out is None else report_out
    report_out.mkdir(parents=True,exist_ok=True)
    labels={token:_saved_label(out,row) for token,row in records.items()}
    summaries={}
    for model in models:
        rows=[r for r in records.values() if r['requested_model']==model]
        summaries[model]=_request_stats(rows)
    result={'planned_requests':len(jobs)*len(models),**_request_stats(list(records.values())),'models':summaries,
        'quality_status':'not independently assessed',
        'timing_scope':'All recorded attempts, including rejected responses and request failures; includes network and API wait, excludes preprocessing and client rate-limit pauses; not a throughput benchmark.'}
    write(report_out/'summary.json',result)
    lines=['# NVIDIA label comparison','',
        'Same three scientific groups and exact prompts. Format success is not scientific accuracy.','',
        'Recorded format decisions are preserved. Length-rejected text is shown without rewriting or declaring it valid.','',
        '| Model | Group | Seconds | Label | Explanation | Format / error |',
        '| --- | --- | --- | --- | --- | --- |']
    for j in jobs:
        for model in models:
            token=digest([model,j['year'],j['level'],j['id']]);r=records.get(token)
            if not r:continue
            label=labels[token] or {}
            state=('accepted; '+label.get('status','') if r['status']=='valid' else
                ('response received, rejected: ' if r.get('response_file') else 'request failed: ')+r.get('error',''))
            if label and label_length_warnings(label):state+='; '+'; '.join(label_length_warnings(label))
            cells=[model,j['id'],f'{r["elapsed_seconds"]:.2f}',label.get('label',''),label.get('gloss',''),
                   state]
            lines.append('| '+' | '.join(str(c).replace('|','/').replace('\n',' ') for c in cells)+' |')
    (report_out/'comparison.md').write_text('\n'.join(lines)+'\n')
    # A separate inspection packet does not reveal generator/model or timing.
    lookup={j['id']:j for j in jobs}
    ordered=[(r,labels[token]) for token,r in records.items() if labels[token] is not None]
    random.Random(7001).shuffle(ordered);tasks=[];key=[]
    for i,(row,label) in enumerate(ordered):
        job=lookup[row['id']];tid=f'blind-{i:03d}'
        tasks.append({'task_id':tid,'label':label['label'],'gloss':label['gloss'],
            'held_out_member_names':[n['surface_form'] for n in job['input']['support']['nodes']],
            'evidence':support_documents(job['input']),'rating':None,
            'allowed_ratings':['accurate','vague','wrong','insufficient_evidence']})
        key.append({'task_id':tid,'group_id':row['id'],'model':row['requested_model']})
    write(report_out/'blind_packet.json',{'completed':False,'tasks':tasks,
        'selection':'All responses passing non-length syntax checks, including length rejections; no accuracy ratings supplied.'})
    write(report_out/'blind_key.private.json',key)
    return result


def inspect_saved_comparison(out):
    """Read saved evidence only; leave the original run and decisions intact."""
    out=Path(out).resolve()
    protocol=json.loads((out/'protocol.json').read_text());jobs=json.loads((out/'prompts.json').read_text())
    if digest(jobs)!=protocol['input_sha256']:raise ValueError('Saved comparison prompts changed')
    records={}
    for job in jobs:
        for model in protocol['models']:
            token=digest([model,job['year'],job['level'],job['id']]);path=out/'completed'/f'{token}.json'
            if not path.exists():continue
            row=json.loads(path.read_text())
            if (row.get('requested_model')!=model or row.get('prompt_sha256')!=job['prompt_sha256']
                    or any(row.get(k)!=job[k] for k in ('year','level','id'))):
                raise ValueError('Cached comparison input mismatch')
            records[token]=row
    return _outputs(out,jobs,protocol['models'],records,report_out=out/'inspection')


def compare_models(run,out,*,year=2026,prompt_path=DEFAULT_PROMPT,models=MODELS,
                   env_file=ROOT/'.env',prepare_only=False,send=None,rpm=0.,thinking=False,
                   retry_failed=False,sleep=time.sleep,decoding='guided_json'):
    run,out=Path(run).resolve(),Path(out).resolve();models=list(models)
    if not models or len(models)!=len(set(models)):raise ValueError('Model IDs must be nonempty and unique')
    constraints=nvidia_constraints(decoding=decoding)
    if not math.isfinite(rpm) or rpm<0:raise ValueError('RPM must be nonnegative;0 means no artificial cap')
    if out==run or out.is_relative_to(run) or run.is_relative_to(out):raise ValueError('Output must be separate from input hierarchy')
    jobs=select_pilot(prepare_jobs(run,prompt_path,[year]))
    protocol={'schema_version':'nvidia-label-pilot-v2','year':year,'models':models,
        'input_sha256':digest(jobs),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'preparation_source_sha256':hashlib.sha256(Path(__file__).with_name('gemini_labels.py').read_bytes()).hexdigest(),
        'selected_groups':[{'id':j['id'],'level':j['level'],'prompt_sha256':j['prompt_sha256'],
                            'prompt_characters':len(j['prompt'])} for j in jobs],
        'selection':'One median-size coarse; median-size fine and largest distinct fine; no gold or generated labels used.',
        'generation':{'enable_thinking':thinking,'temperature':1.,'top_p':.95,'max_tokens':512,
                      'decoding':decoding,'constraints':{m:constraints for m in models},
                      'client':'openai-sdk','max_retries':0},
        'retries':0,'quality_policy':'A supplied-evidence comparison, not independent scientific faithfulness.'}
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Comparison protocol changed; choose another output directory')
    write(path,protocol);write(out/'prompts.json',jobs)
    records={}
    for job in jobs:
        for model in models:
            token=digest([model,job['year'],job['level'],job['id']]);cache=out/'completed'/f'{token}.json'
            if cache.exists():
                row=json.loads(cache.read_text())
                if row.get('prompt_sha256')!=job['prompt_sha256'] or row.get('requested_model')!=model:
                    raise ValueError('Cached comparison input mismatch')
                if row['status']=='valid':
                    saved_path=out/row['response_file']
                    if (not saved_path.resolve().is_relative_to(out) or not saved_path.is_file()
                            or hashlib.sha256(saved_path.read_bytes()).hexdigest()!=row.get('response_sha256')):
                        raise ValueError('Cached comparison response changed or is missing')
                    parsed,raw=extract_label(json.loads(saved_path.read_text()))
                    if parsed!=row['label'] or raw!=row['raw_output']:
                        raise ValueError('Cached comparison label differs from its recorded response')
                if row['status']=='valid' or not retry_failed:records[token]=row
    result=_outputs(out,jobs,models,records)
    print(f'{len(jobs)} groups × {len(models)} models = {result["planned_requests"]} planned requests; {len(records)} cached',flush=True)
    if prepare_only or len(records)==result['planned_requests']:return result
    key=load_api_key(env_file,variable='NVIDIA_API_KEY')
    sender=send or (lambda m,p,k:request_nvidia(m,p,k,thinking=thinking,decoding=decoding))
    last_call=None
    for index,job in enumerate(jobs):
        # Alternate model order within the shared group to reduce fixed order effects.
        for model in models if index%2==0 else list(reversed(models)):
            token=digest([model,job['year'],job['level'],job['id']])
            if token in records:continue
            if rpm and last_call is not None:sleep(max(0.,60/rpm-(time.monotonic()-last_call)))
            started=time.monotonic();last_call=started
            row={'id':job['id'],'year':job['year'],'level':job['level'],'requested_model':model,
                 'prompt_sha256':job['prompt_sha256'],'status':'error'}
            try:
                response=sender(model,job['prompt'],key)
                row['elapsed_seconds']=time.monotonic()-started
                response_path=out/'responses'/f'{token}-{time.time_ns()}.json';write(response_path,response)
                row['response_file']=str(response_path.relative_to(out))
                row['response_sha256']=hashlib.sha256(response_path.read_bytes()).hexdigest()
                label,raw=extract_label(response,structured=True)
                row.update(status='valid',label=label,raw_output=raw,model_version=response.get('model'),
                           usage=response.get('usage',{}))
            except (ValueError,RuntimeError,OSError) as exc:
                row.setdefault('elapsed_seconds',time.monotonic()-started)
                row['error']=str(exc).replace(key,'[REDACTED]')
            write(out/'completed'/f'{token}.json',row);records[token]=row
            result=_outputs(out,jobs,models,records)
            display_status=('valid' if row['status']=='valid' else
                            'format_error' if row.get('response_file') else 'request_error')
            print(f'{model} {job["id"]}: {display_status}, {row["elapsed_seconds"]:.2f}s',flush=True)
            if row['status']=='error' and re.search(r'NVIDIA HTTP (401|403)\b',row.get('error','')):
                return result
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,default=ROOT/'out/nvidia_label_comparison_guided')
    parser.add_argument('--year',type=int,default=2026)
    parser.add_argument('--models',nargs='+',default=MODELS)
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    parser.add_argument('--rpm',type=float,default=0.)
    parser.add_argument('--thinking',action='store_true')
    parser.add_argument('--decoding',choices=('guided_json','json_schema'),default='guided_json')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--inspect-saved',action='store_true',help='Inspect saved responses offline; preserve original run files')
    args=parser.parse_args()
    try:
        if args.inspect_saved:
            r=inspect_saved_comparison(args.out)
            print(f'{r["responses_received"]} responses; {r["invalid_outputs"]} format rejections; '
                  f'{r["failed_requests"]} request failures; see {args.out}/inspection/comparison.md')
            return 0
        r=compare_models(args.run,args.out,year=args.year,models=args.models,env_file=args.env_file,
            prepare_only=args.prepare_only,rpm=args.rpm,thinking=args.thinking,retry_failed=args.retry_failed,
            decoding=args.decoding)
        print(f'{r["valid_outputs"]}/{r["planned_requests"]} valid outputs; '
              f'{r["invalid_outputs"]} format rejections; {r["failed_requests"]} request failures; '
              f'see {args.out}/comparison.md')
        return 1 if r['failed_requests'] or r['invalid_outputs'] else 0
    except KeyboardInterrupt:print('\nStopped; completed requests are saved.',file=sys.stderr);return 130
    except (ValueError,RuntimeError,OSError) as exc:print(str(exc),file=sys.stderr);return 1
