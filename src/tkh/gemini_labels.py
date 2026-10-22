"""User-run, resumable Gemini annotation of an existing, immutable hierarchy.

No neural inference is performed locally. Preparing prompts needs no key/network.
Full-scope naming exposes nonclaim names; support claim sentences stay hidden.
Reference questions/answers are not read.
"""
import argparse
from collections import Counter,deque
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .scientific_labels import build_label_input,parse_label_output,TEMPLATE_GLOSS_PHRASES

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_RUN=ROOT/'out/cpu/hierarchy'
DEFAULT_PROMPT=ROOT/'prompts/group_labels.md'
DEFAULT_MODEL='gemini-3.8-flash'
LABEL_WORDS=(2,12)
GLOSS_WORDS=(1,40)
# This exact predecessor differs only by the terminal progress integration.
# Preserve its protocol and generated cache; record the new client separately.
PROGRESS_COMPATIBLE_CLIENT='3209bbc1acaa670fde6659c88ce04064f21262316eea2fa57a520d40679ccb2a'
CACHE_COMPATIBLE_CLIENTS={PROGRESS_COMPATIBLE_CLIENT,
    '64495c431ac888c10fe427eeb1f91fb811303d88ca2a914f66cf67bc6d4f2be8',
    '7d374741af300382aaa17db2d8b9143747534bfc455fd9d193bc2d0c5bbda3b1',
    # Same generation inputs and guided transport; sentence style becomes a review hint.
    '47dd61a36948a78a997bdb8e4a17bd07104afe9b044d73bc227cb608e76c26a1'}
SCHEMA={'type':'OBJECT','properties':{
    'label':{'type':'STRING'},'gloss':{'type':'STRING'},
    'status':{'type':'STRING','enum':['coherent','mixed','insufficient']}},
    'required':['label','gloss','status']}
BATCH_SCHEMA={'type':'ARRAY','items':{'type':'OBJECT',
    'properties':{**SCHEMA['properties'],'group_id':{'type':'STRING'}},
    'required':['group_id','label','gloss','status']}}
SINGLE_CONTRACT='Return ONLY one JSON object with exactly label, gloss and status fields.'
BATCH_CONTRACT=('For EACH group below, use only its OWN evidence block. Return ONLY a JSON ARRAY '
    'with exactly one object per supplied group, each containing group_id, label, gloss and status. '
    'Copy group_id exactly. Do not omit, duplicate or invent groups; do not transfer facts between groups.')
NVIDIA_BATCH_CONTRACT=('For EACH group below, use only its OWN evidence block. Return ONLY a JSON OBJECT '
    'of the form {"groups": [...]}, with exactly one object per supplied group in groups. '
    'Each object must contain group_id, label, gloss and status. Copy group_id exactly. '
    'Do not omit, duplicate or invent groups; do not transfer facts between groups.')


class LabelRequestError(RuntimeError):
    """A provider failure with an explicit, bounded-retry decision."""
    def __init__(self,message,*,retryable=False,retry_after=None):
        super().__init__(message)
        self.retryable=retryable
        self.retry_after=retry_after


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True);raise
    try:os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def load_api_key(env_file,environ=None,variable='GEMINI_API_KEY'):
    env=os.environ if environ is None else environ
    key=env.get(variable,'').strip()
    path=Path(env_file)
    if not key and path.is_file():
        for line in path.read_text().splitlines():
            line=line.strip().removeprefix('export ')
            name,sep,value=line.partition('=')
            if sep and name.strip()==variable:
                key=value.strip()
                if len(key)>=2 and key[0]==key[-1] and key[0] in '\"\'':key=key[1:-1]
    if not key or any(ord(c)<32 for c in key):
        raise ValueError(f'Set {variable} in the local .env file or environment')
    return key


def prepare_jobs(run,prompt_path=DEFAULT_PROMPT,years=None,*,evidence_policy='source_split',groups_file=None):
    run=Path(run);template=Path(prompt_path).read_text(encoding='utf-8').strip()
    if evidence_policy not in {'source_split','full_scope'}:raise ValueError('Unknown evidence policy')
    requested=None
    if groups_file is not None:
        rows=json.loads(Path(groups_file).read_text())
        if not isinstance(rows,list) or not rows:raise ValueError('Group selection must be a nonempty JSON list')
        try:requested={(r['year'],r['level'],r['group_id']) for r in rows}
        except (KeyError,TypeError):raise ValueError('Group selection requires year, level and group_id') from None
        if len(requested)!=len(rows):raise ValueError('Duplicate group selection')
    available=sorted(int(p.name) for p in (run/'snapshots').iterdir() if p.is_dir() and p.name.isdigit())
    selected=available if years is None else sorted(set(years))
    if not selected or set(selected)-set(available):raise ValueError('Requested snapshots are missing')
    jobs=[]
    for year in selected:
        directory=run/'snapshots'/str(year)
        graph=json.loads((directory/'input.json').read_text())
        hierarchy=json.loads((directory/'hierarchy.json').read_text())
        if graph['cutoff']!=year or hierarchy['cutoff']!=year:raise ValueError('Snapshot cutoff mismatch')
        for level in hierarchy['levels'][:2]:
            for group in sorted(level['groups'],key=lambda g:g['id']):
                if requested is not None and (year,level['index'],group['id']) not in requested:continue
                inp=build_label_input(graph,group['members'],year,seed=7001,
                    max_generation_members=max(1,len(group['members'])),selection_policy='science_evidence')
                members=[n for n in inp['generation']['nodes'] if n['type'] not in ('article','author')]
                mids={n['id'] for n in members}
                # All scientific generation members and claim contexts: no top24 truncation.
                context=[c for c in inp['generation']['claim_context'] if set(c['supporting_member_ids']) & mids]
                payload={'cutoff':year,'level':'coarse' if level['index']==0 else 'fine',
                    'group_id':group['id'],'total_member_count':len(group['members']),
                    'supplied_scientific_member_count':len(members),'members':members,
                    'hyperedges':[e for e in inp['generation']['hyperedges'] if set(e['members'])<=mids],
                    'claim_context':context}
                if evidence_policy=='full_scope':
                    from .label_scope import build_scope_evidence
                    payload=build_scope_evidence(graph,inp,level=level['index'],group_id=group['id'])
                from .label_scope import topic_representation
                payload['topic_representation']=topic_representation(payload)
                prompt=template+'\n\n'+SINGLE_CONTRACT+'\n\nEVIDENCE:\n'+json.dumps(payload,ensure_ascii=False,sort_keys=True)
                jobs.append({'year':year,'id':group['id'],'level':level['index'],
                    'member_count':len(group['members']),'input':inp,'prompt':prompt,
                    'instructions':template,'evidence':payload,
                    'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                    'input_sha256':digest(graph),'hierarchy_sha256':digest(hierarchy)})
    if requested is not None and {(j['year'],j['level'],j['id']) for j in jobs}!=requested:
        raise ValueError('Group selection contains missing snapshots, levels or group IDs')
    if len({(j['year'],j['level'],j['id']) for j in jobs})!=len(jobs):raise ValueError('Duplicate group IDs')
    return jobs


def batch_prompt(jobs,*,provider='gemini'):
    if not jobs:raise ValueError('Cannot generate an empty batch')
    if len(jobs)==1:return jobs[0]['prompt']
    contract=NVIDIA_BATCH_CONTRACT if provider=='nvidia' else BATCH_CONTRACT
    ids=[j['id'] for j in jobs]
    return jobs[0]['instructions']+'\n\n'+contract+f'\nExpected {len(ids)} group IDs, each exactly once: '+json.dumps(ids)+'\n\nBATCH_EVIDENCE:\n'+json.dumps(
        [j['evidence'] for j in jobs],ensure_ascii=False,sort_keys=True)


def generation_prompt(jobs,*,provider='gemini',attempt=1):
    prompt=batch_prompt(jobs,provider=provider)
    if attempt>1:
        prompt=(f'Retry attempt {attempt}: the previous attempt failed validation. Return compact JSON only: no commentary, '
                'word counts, blank lines or padding. Close the JSON and stop immediately. '
                f'Use every requested group ID exactly once. Keep each gloss to at most {GLOSS_WORDS[1]} words.\n\n'+prompt)
    return prompt


def plan_batches(jobs,*,batch_size=10,max_chars=120000,provider='gemini'):
    """First-fit by cutoff/level with a strict cross-group source-leakage guard."""
    if type(batch_size)is not int or batch_size<1 or type(max_chars)is not int or max_chars<1:
        raise ValueError('Batch size and maximum characters must be positive integers')
    batches=[]
    for job in jobs:
        if len(batch_prompt([job],provider=provider))>max_chars:
            raise ValueError(f'Group {job["id"]} too large: increase --max-batch-chars; evidence was not truncated')
        gen=set(job['input']['generation']['source_ids']);held=set(job['input']['support']['source_ids'])
        for batch in batches:
            if len(batch)>=batch_size or (batch[0]['year'],batch[0]['level'])!=(job['year'],job['level']):continue
            other_gen=set().union(*(set(j['input']['generation']['source_ids']) for j in batch))
            other_held=set().union(*(set(j['input']['support']['source_ids']) for j in batch))
            if gen & other_held or held & other_gen:continue
            if len(batch_prompt([*batch,job],provider=provider))>max_chars:continue
            batch.append(job);break
        else:batches.append([job])
    return batches


def request_gemini(model,prompt,key,*,opener=None,sleep=time.sleep,timeout=120,retries=2,
                   schema=None,max_output_tokens=4096):
    if not re.fullmatch(r'[a-zA-Z0-9._-]+',model):raise ValueError('Invalid Gemini model ID')
    config={'temperature':1.,'maxOutputTokens':max_output_tokens,
            'responseMimeType':'application/json','responseSchema':SCHEMA if schema is None else schema}
    if model.startswith('gemini-3'):config['thinkingConfig']={'thinkingLevel':'LOW'}
    request=Request(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
        data=json.dumps({'contents':[{'role':'user','parts':[{'text':prompt}]}],
                         'generationConfig':config}).encode(),
        headers={'Content-Type':'application/json','x-goog-api-key':key},method='POST')
    opener=urlopen if opener is None else opener
    for attempt in range(retries+1):
        try:
            with opener(request,timeout=timeout) as response:return json.load(response)
        except HTTPError as exc:
            body=exc.read().decode('utf-8',errors='replace').replace(key,'[REDACTED]')
            try:message=json.loads(body).get('error',{}).get('message','')
            except (ValueError,AttributeError):message=''
            if exc.code in {429,500,502,503,504} and attempt<retries:
                try:delay=float(exc.headers.get('Retry-After',10*2**attempt))
                except (TypeError,ValueError):delay=10*2**attempt
                if not math.isfinite(delay) or delay>120:
                    raise RuntimeError(f'Gemini HTTP {exc.code}: retry later; progress was saved') from None
                sleep(max(0.,delay));continue
            try:retry_after=float(exc.headers.get('Retry-After','nan'))
            except (TypeError,ValueError):retry_after=float('nan')
            if not math.isfinite(retry_after):retry_after=None
            raise LabelRequestError(f'Gemini HTTP {exc.code}: {message[:800] or "request failed"}; progress was saved',
                retryable=exc.code in {429,500,502,503,504},retry_after=retry_after) from None
        except (URLError,TimeoutError):
            if attempt<retries:sleep(10*2**attempt);continue
            raise LabelRequestError('Gemini network/timeout error; progress was saved',retryable=True) from None


def request_gemini_batch(model,prompt,key,**kwargs):
    return request_gemini(model,prompt,key,schema=BATCH_SCHEMA,max_output_tokens=8192,**kwargs)


def _response_text(response):
    if 'choices' in response:
        choices=response['choices']
        if not choices:raise ValueError('NVIDIA returned no completion choice')
        if choices[0].get('finish_reason')!='stop':
            raise ValueError('Incomplete NVIDIA response: '+str(choices[0].get('finish_reason')))
        content=choices[0].get('message',{}).get('content')
        if not isinstance(content,str) or not content.strip():raise ValueError('NVIDIA returned no final text')
        return content
    candidates=response.get('candidates',[])
    if not candidates:raise ValueError('Gemini returned no candidate (blocked or empty response)')
    candidate=candidates[0]
    if candidate.get('finishReason')!='STOP':
        raise ValueError('Incomplete Gemini response: '+str(candidate.get('finishReason')))
    return ''.join(part.get('text','') for part in candidate.get('content',{}).get('parts',[]) if not part.get('thought'))


def label_length_warnings(label):
    warnings=[]
    for field,(low,high) in [('label',LABEL_WORDS),('gloss',GLOSS_WORDS)]:
        count=len(label[field].split())
        if not low<=count<=high:warnings.append(f'{field}: {count} words; requested {low}–{high}')
    return warnings


def label_quality_warnings(label):
    """Lexical review hints, never a scientific judgment or a format failure."""
    warnings=[f'Possible template phrase: {p}; review the actual scientific content.'
              for p in TEMPLATE_GLOSS_PHRASES if p in label['gloss'].lower()]
    if len(re.findall(r'[.!?](?:\s|$)',label['gloss']))!=1:
        warnings.append('Style review: sentence punctuation differs from the requested single-sentence form; original text retained.')
    return warnings


def _parse_explanation(raw,*,length_policy='strict'):
    if length_policy not in {'strict','warn'}:raise ValueError('Unknown length policy')
    label=parse_label_output(raw,enforce_length=False,enforce_template=False,enforce_sentence=False)
    if length_policy=='strict' and label_length_warnings(label):
        raise ValueError('; '.join(label_length_warnings(label)))
    return label


def _validate_nvidia_schema(value,expected_ids=None):
    from jsonschema import Draft202012Validator,ValidationError
    from .nvidia_labels import nvidia_response_format
    try:Draft202012Validator(nvidia_response_format(expected_ids)['json_schema']['schema']).validate(value)
    except ValidationError as exc:
        raise ValueError('Response violates JSON Schema at '+'.'.join(map(str,exc.path))+': '+exc.message) from None


def extract_label(response,*,length_policy='strict',structured=False):
    raw=_response_text(response)
    if structured:
        try:value=json.loads(raw)
        except (ValueError,TypeError):raise ValueError('Response is not valid JSON') from None
        _validate_nvidia_schema(value)
    label=_parse_explanation(raw,length_policy=length_policy)
    return label,raw


def extract_batch_labels(response,expected_ids,*,length_policy='strict',envelope=False):
    raw=_response_text(response)
    try:rows=json.loads(raw)
    except (ValueError,TypeError) as exc:raise ValueError('Batch response is not valid JSON') from exc
    if envelope:
        _validate_nvidia_schema(rows,expected_ids)
        rows=rows['groups']
    if not isinstance(rows,list) or any(not isinstance(r,dict) or set(r)!={'group_id','label','gloss','status'} for r in rows):
        raise ValueError('Batch response must contain group_id, label, gloss and status per object')
    ids=[r['group_id'] for r in rows]
    if any(not isinstance(g,str) for g in ids) or len(set(ids))!=len(ids) or set(ids)!=set(expected_ids):
        raise ValueError('Batch response has missing, duplicate or unexpected group IDs')
    result={}
    for row in rows:
        raw=json.dumps({k:row[k] for k in ('label','gloss','status')},ensure_ascii=False)
        result[row['group_id']]=(_parse_explanation(raw,length_policy=length_policy),raw)
    return result


def _export(run,out,jobs,completed,status,planned_requests=None,rejected_this_invocation=0,
            failed_groups_this_invocation=0,attempted_requests_this_invocation=0,retried_requests_this_invocation=0):
    rows=list(completed.values())
    for year in sorted({j['year'] for j in jobs}):
        year_rows=[r for r in rows if r['year']==year]
        write(out/'snapshots'/str(year)/'labels.json',year_rows)
        hierarchy=json.loads((run/'snapshots'/str(year)/'hierarchy.json').read_text())
        expected={(level['index'],g['id']) for level in hierarchy['levels'][:2] for g in level['groups']}
        if {(r['level'],r['id']) for r in year_rows}!=expected:continue
        lookup={(r['level'],r['id']):r for r in year_rows}
        for level in hierarchy['levels'][:2]:
            for group in level['groups']:
                row=lookup[(level['index'],group['id'])]
                group.update(label=row['label'],gloss=row['gloss'],label_status=row['status'],
                             generation_prompt_sha256=row['prompt_sha256'],
                             label_quality_warnings=row.get('quality_warnings',[]))
        write(out/'snapshots'/str(year)/'hierarchy.json',hierarchy)
    summary={'status':status,'total':len(jobs),'completed':len(rows),'remaining':len(jobs)-len(rows),
             'planned_requests':planned_requests,
             'completed_requests':len({r['response_file'] for r in rows}),
             'generation_status_counts':dict(Counter(r['status'] for r in rows)),
             'labels_with_length_warnings':sum(bool(r.get('length_warnings')) for r in rows),
             'labels_with_quality_warnings':sum(bool(r.get('quality_warnings')) for r in rows),
             'rejected_this_invocation':rejected_this_invocation,
             'failed_groups_this_invocation':failed_groups_this_invocation,
             'attempted_requests_this_invocation':attempted_requests_this_invocation,
             'retried_requests_this_invocation':retried_requests_this_invocation,
             'rejected_responses':len(list((out/'rejected').glob('*.json'))),
             'imported_legacy_labels':sum(r.get('response_format')=='legacy-prompt-only-imported' for r in rows),
             'quality_status':'Generated labels; independent scientific faithfulness has not been measured.'}
    write(out/'summary.json',summary)
    write(out/'label_review.json',{'quality_status':'Heuristic review hints; scientific faithfulness remains unassessed.',
        'items':[{k:r[k] for k in ('year','level','id','label','gloss','quality_warnings','response_file')}
                 for r in rows if r.get('quality_warnings')]})
    lines=['# Generated group labels','',f'Completed: {len(rows)}/{len(jobs)}.','',
           '| Year | Level | Group | Label | Explanation | Model status | Length warnings | Review hints |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in sorted(rows,key=lambda r:(r['year'],r['level'],r['id'])):
        cells=[r['year'],'coarse' if r['level']==0 else 'fine',r['id'],r['label'],r['gloss'],r['status'],
               '; '.join(r.get('length_warnings',[])),'; '.join(r.get('quality_warnings',[]))]
        lines.append('| '+' | '.join(str(c).replace('|','/').replace('\n',' ') for c in cells)+' |')
    (out/'labels.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def run_labels(run,out,*,prompt_path=DEFAULT_PROMPT,model=DEFAULT_MODEL,years=None,
               env_file=ROOT/'.env',prepare_only=False,limit=None,rpm=0.,send=None,sleep=time.sleep,
               batch_size=3,max_batch_chars=120000,provider='gemini',thinking=False,length_policy='warn',
               decoding='guided_json',max_output_tokens=512,max_attempts=3,retry_delay=2.,
               evidence_policy='source_split',groups_file=None):
    run,out=Path(run).resolve(),Path(out).resolve()
    if out==run or out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('Output and source hierarchy directories must be separate')
    if not math.isfinite(rpm) or rpm<0 or (limit is not None and limit<1):raise ValueError('Invalid RPM or limit')
    if provider not in {'gemini','nvidia'}:raise ValueError('Unknown label API provider')
    if length_policy not in {'strict','warn'}:raise ValueError('Unknown length policy')
    if type(max_output_tokens)is not int or max_output_tokens<1:raise ValueError('Token limit must be a positive integer')
    if type(max_attempts)is not int or max_attempts<1:raise ValueError('Maximum attempts must be a positive integer')
    if not math.isfinite(retry_delay) or retry_delay<0:raise ValueError('Retry delay must be finite and nonnegative')
    jobs=prepare_jobs(run,prompt_path,years,evidence_policy=evidence_policy,groups_file=groups_file)
    batches=plan_batches(jobs,batch_size=batch_size,max_chars=max_batch_chars,provider=provider)
    generation=({'temperature':1.,'maxOutputTokens':max_output_tokens,'thinking':'LOW for Gemini3','transport_retries':0}
        if provider=='gemini' else {'temperature':1.,'top_p':.95,'max_tokens':max_output_tokens,'enable_thinking':thinking,
                                   'response_format':'json_schema','batch_envelope':'groups','client':'openai-sdk','max_retries':0})
    if provider=='nvidia':
        from .nvidia_labels import nvidia_constraints
        nvidia_constraints(decoding=decoding)
        generation['response_format']=decoding
    protocol={'schema_version':'api-labels-v6',
        'source_run':str(run),'model':model,'provider':provider,
        'jobs_sha256':digest(jobs),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'generation':generation,
        'batch_size':batch_size,'max_batch_chars':max_batch_chars,
        'batch_policy':'Same cutoff and hierarchy level; no generation source may be held out for another group in the request.',
        'source_split_seed':7001,'evidence_policy':'All scientific generation members and source-safe claims; heldout sources are not sent.',
        'length_policy':length_policy,
        'requested_style':{'label_words':list(LABEL_WORDS),'gloss_words':list(GLOSS_WORDS),'gloss_sentences':1},
        'retry_policy':{'max_attempts_per_group':max_attempts,'initial_delay_seconds':retry_delay,
            'invalid_response':'Retry individual groups; inherited attempt budget; no format fallback.',
            'transient_api_error':'Retry same batch; exponential backoff and Retry-After; stop if wait exceeds120seconds.'},
        'quality_policy':'Syntax and coverage checks only; generation status is not independent evaluation.'}
    if provider=='nvidia':
        protocol['nvidia_transport_sha256']=hashlib.sha256(Path(__file__).with_name('nvidia_labels.py').read_bytes()).hexdigest()
    if evidence_policy=='full_scope':
        protocol.update(schema_version='api-labels-v8',evidence_policy_id=evidence_policy,
            evidence_policy='Keywords and four representative records; full nonclaim scientific member scope; generation-side claims; bounded adjacent context.',
            independence={'scope_names_hidden':False,'support_claim_texts_hidden':True,
                'limitation':'Member names may overlap clustering inputs; a scope audit is not held-out validation.'},
            scope_code_sha256=hashlib.sha256(Path(__file__).with_name('label_scope.py').read_bytes()).hexdigest())
    if groups_file is not None:
        protocol['group_selection']=[{'year':j['year'],'level':j['level'],'group_id':j['id']} for j in jobs]
    out.mkdir(parents=True,exist_ok=True)
    path=out/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=protocol:
            compatible={**previous,'source_sha256':protocol['source_sha256']}
            if previous.get('source_sha256') not in CACHE_COMPATIBLE_CLIENTS or compatible!=protocol:
                raise ValueError('Protocol changed: use a different output directory for different inputs, code or model')
            write(out/'client_code_versions.json',{
                'original_source_sha256':previous['source_sha256'],'current_source_sha256':protocol['source_sha256'],
                'compatibility':'Exact generation inputs/settings retained; client progress, review hints and rejection handling updated.',
                'sentence_policy':'Requested single-sentence style is a review hint; generated text is retained unchanged.'})
    else:write(path,protocol)
    write(out/'prompts.json',jobs)
    completed={}
    for job in jobs:
        token=digest([job['year'],job['level'],job['id']]);cache=out/'completed'/f'{token}.json'
        if cache.exists():
            row=json.loads(cache.read_text())
            if provider=='nvidia' and decoding=='guided_json' and row.get('response_format')!='guided_json':
                raise ValueError('Cached label decoding differs from guided_json; use a fresh output directory')
            if any(row[k]!=job[k] for k in ('year','id','level','member_count','prompt_sha256')):
                raise ValueError('Cached generation does not match exact group input')
            label=_parse_explanation(row['raw_output'],length_policy=length_policy)
            if any(row.get(k)!=v for k,v in label.items()):raise ValueError('Cached label text changed')
            if row.get('length_warnings',[])!=label_length_warnings(label):raise ValueError('Cached length warnings changed')
            row['quality_warnings']=label_quality_warnings(label)
            completed[token]=row
    rejected_this_invocation=0
    failed_groups_this_invocation=attempted_requests_this_invocation=retried_requests_this_invocation=0
    def save(status):return _export(run,out,jobs,completed,status,planned_requests=len(batches),
        rejected_this_invocation=rejected_this_invocation,failed_groups_this_invocation=failed_groups_this_invocation,
        attempted_requests_this_invocation=attempted_requests_this_invocation,
        retried_requests_this_invocation=retried_requests_this_invocation)
    result=save('prepared' if prepare_only else 'partial')
    pending=[j for j in jobs if digest([j['year'],j['level'],j['id']]) not in completed]
    if limit is not None:pending=pending[:limit]
    pending_batches=plan_batches(pending,batch_size=batch_size,max_chars=max_batch_chars,provider=provider)
    write(out/'batch_plan.json',{'planned_requests':len(batches),'remaining_requests_this_invocation':len(pending_batches),
        'batches':[{'year':b[0]['year'],'level':b[0]['level'],'group_ids':[j['id'] for j in b],
                   'prompt_characters':len(batch_prompt(b,provider=provider))} for b in pending_batches]})
    print(f'{len(jobs)} groups; {len(completed)} cached; {len(jobs)-len(completed)} remaining',flush=True)
    print(f'{len(pending_batches)} requests planned for this invocation (up to {batch_size} groups each)',flush=True)
    if prepare_only or len(completed)==len(jobs):
        if not prepare_only:result=save('complete')
        return result
    key=load_api_key(env_file,variable='NVIDIA_API_KEY' if provider=='nvidia' else 'GEMINI_API_KEY')
    from .label_progress import label_progress
    progress=label_progress(len(pending_batches),len(completed),len(jobs))
    last_call=None
    queue=deque((b,1,None,0.) for b in pending_batches)
    def schedule_retry(batch,attempt,request_hash,delay,*,split):
        nonlocal failed_groups_this_invocation
        if attempt>=max_attempts:
            failed_groups_this_invocation+=len(batch)
            return False
        next_batches=[[j] for j in batch] if split else [batch]
        for b in reversed(next_batches):queue.appendleft((b,attempt+1,request_hash,delay))
        progress.total+=len(next_batches);progress.refresh()
        return True
    try:
        while queue:
            batch,attempt,retry_of,delay=queue.popleft()
            prompt=generation_prompt(batch,provider=provider,attempt=attempt);request_hash=hashlib.sha256(prompt.encode()).hexdigest()
            request={'prompt':prompt,'prompt_sha256':request_hash,
                'year':batch[0]['year'],'level':batch[0]['level'],'group_ids':[j['id'] for j in batch],
                'attempt':attempt,'retry_of':retry_of,'max_output_tokens':max_output_tokens}
            sender=send or (lambda m,p,k:request_gemini(m,p,k,retries=0,max_output_tokens=max_output_tokens,
                schema=BATCH_SCHEMA if len(batch)>1 else SCHEMA))
            if provider=='nvidia':
                from .nvidia_labels import request_nvidia,nvidia_response_format,nvidia_constraints
                spec=nvidia_response_format(request['group_ids'] if len(batch)>1 else None)
                request.update(nvidia_constraints(spec,decoding=decoding))
                request['decoding']=decoding
                request['validation_schema']=spec['json_schema']['schema']
                if send is None:
                    sender=lambda m,p,k:request_nvidia(m,p,k,thinking=thinking,response_format=spec,decoding=decoding,
                                                     max_tokens=max_output_tokens)
            write(out/'requests'/f'{request_hash}.json',request)
            if delay:sleep(delay)
            if rpm and last_call is not None:sleep(max(0.,60/rpm-(time.monotonic()-last_call)))
            last_call=time.monotonic()
            attempted_requests_this_invocation+=1
            retried_requests_this_invocation+=int(attempt>1)
            try:response=sender(model,prompt,key)
            except RuntimeError as exc:
                retryable=getattr(exc,'retryable',False)
                wait=max(retry_delay*2**(attempt-1),getattr(exc,'retry_after',None) or 0.)
                error=str(exc).replace(key,'[REDACTED]')
                write(out/'request_errors'/f'{request_hash}-{time.time_ns()}.json',{
                    'error':error,'attempt':attempt,'group_ids':request['group_ids'],
                    'request_file':f'requests/{request_hash}.json','retryable':retryable})
                if not retryable:raise
                if wait>120:raise RuntimeError(f'{error}; server requested a long wait; rerun later to resume') from None
                will_retry=schedule_retry(batch,attempt,request_hash,wait,split=False)
                progress.write(f'API error ({attempt}/{max_attempts}): {error}. '+
                    ('Retrying.' if will_retry else 'Attempt limit reached; stopping API calls.'),file=sys.stderr)
                result=save('partial');progress.update(1)
                if not will_retry:raise RuntimeError(f'API attempt limit reached: {error}; completed labels saved') from None
                continue
            response_path=out/'responses'/f'{request_hash}-{time.time_ns()}.json'
            write(response_path,response)
            # Validate the ENTIRE response before committing even one label.
            try:
                parsed=(extract_batch_labels(response,{j['id'] for j in batch},length_policy=length_policy,
                            envelope=provider=='nvidia') if len(batch)>1 else
                        {batch[0]['id']:extract_label(response,length_policy=length_policy,structured=provider=='nvidia')})
            except ValueError as exc:
                rejected_this_invocation+=1
                write(out/'rejected'/f'{response_path.stem}.json',{
                    'year':batch[0]['year'],'level':batch[0]['level'],'group_ids':[j['id'] for j in batch],
                    'error':str(exc),'response_file':str(response_path.relative_to(out)),
                    'response_sha256':hashlib.sha256(response_path.read_bytes()).hexdigest(),
                    'request_file':f'requests/{request_hash}.json','requested_model':model,'attempt':attempt,
                    'will_retry':attempt<max_attempts})
                will_retry=schedule_retry(batch,attempt,request_hash,retry_delay*2**(attempt-1),split=True)
                progress.write(f'Rejected response ({len(batch)} groups, attempt {attempt}/{max_attempts}): {exc}; '
                    f'raw response saved at {response_path}. '+
                    ('Retrying groups individually.' if will_retry else 'Attempt limit reached; continuing.'),file=sys.stderr)
                result=save('partial');progress.update(1)
                continue
            for job in batch:
                token=digest([job['year'],job['level'],job['id']]);label,raw=parsed[job['id']]
                row={k:job[k] for k in ('year','id','level','member_count','prompt_sha256')}
                row.update(**label,raw_output=raw,requested_model=model,provider=provider,
                    length_warnings=label_length_warnings(label),
                    quality_warnings=label_quality_warnings(label),
                    response_format=decoding if provider=='nvidia' else 'gemini-responseSchema',
                    model_version=response.get('modelVersion',response.get('model')),
                    usage=response.get('usageMetadata',response.get('usage',{})),usage_scope='shared request' if len(batch)>1 else 'single request',
                    batch_prompt_sha256=request_hash,request_file=f'requests/{request_hash}.json',
                    response_file=str(response_path.relative_to(out)))
                write(out/'completed'/f'{token}.json',row);completed[token]=row
                note='; needs review' if row['quality_warnings'] else ''
                progress.write(f'[{len(completed)}/{len(jobs)}] {job["year"]} L{job["level"]} {job["id"]}: {label["label"]} ({label["status"]}{note})',file=sys.stdout)
            result=save('partial')
            progress.set_postfix(groups=f'{len(completed)}/{len(jobs)}',refresh=False)
            progress.update(1)
    except BaseException:
        save('interrupted_or_error');raise
    finally:progress.close()
    return save('complete' if len(completed)==len(jobs) else 'partial')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--provider',choices=('gemini','nvidia'),default='gemini')
    parser.add_argument('--model')
    parser.add_argument('--thinking',action='store_true',help='Enable NVIDIA reasoning; disabled by default for short labels')
    parser.add_argument('--decoding',choices=('guided_json','json_schema'),default='guided_json',
        help='NVIDIA schema-constrained decoding; no ordinary JSON mode or automatic fallback')
    parser.add_argument('--length-policy',choices=('strict','warn'),default='warn',
        help='warn preserves text with length warnings; sentence style is a review hint; JSON and status remain mandatory')
    parser.add_argument('--prompt',type=Path,default=DEFAULT_PROMPT)
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env')
    parser.add_argument('--years',type=int,nargs='+')
    parser.add_argument('--evidence-policy',choices=('full_scope','source_split'),default='full_scope',
        help='full_scope shows all nonclaim member names, withholding support claim texts; source_split is the legacy comparison')
    parser.add_argument('--groups-file',type=Path,help='Optional JSON list of year/level/group_id for a small repair pilot')
    parser.add_argument('--limit',type=int,help='Maximum NEW group generations in this invocation')
    parser.add_argument('--batch-size',type=int,default=3,help='Maximum groups per request; default3 fits short responses')
    parser.add_argument('--max-batch-chars',type=int,default=120000,help='Maximum input characters per request; no evidence truncation')
    parser.add_argument('--max-output-tokens',type=int,default=512,help='Maximum tokens per response, including retries')
    parser.add_argument('--max-attempts',type=int,default=3,help='Maximum attempts per group, including its initial shared request')
    parser.add_argument('--retry-delay',type=float,default=2.,help='Initial retry pause in seconds; doubles on subsequent failures')
    parser.add_argument('--rpm',type=float,default=0.,help='Optional client requests/minute cap; default0 means no artificial delay')
    parser.add_argument('--prepare-only',action='store_true',help='Write prompts without reading the key or using the API')
    args=parser.parse_args()
    if args.model is None:
        args.model=DEFAULT_MODEL if args.provider=='gemini' else 'nvidia/nemotron-3.5-lightning-30b-a3b'
    if args.out is None:args.out=ROOT/('out/gemini_topic_labels_v4' if args.provider=='gemini' else 'out/nvidia_topic_labels_v4')
    try:
        result=run_labels(args.run,args.out,prompt_path=args.prompt,model=args.model,years=args.years,
            env_file=args.env_file,prepare_only=args.prepare_only,limit=args.limit,rpm=args.rpm,
            batch_size=args.batch_size,max_batch_chars=args.max_batch_chars,provider=args.provider,thinking=args.thinking,
            length_policy=args.length_policy,decoding=args.decoding,max_output_tokens=args.max_output_tokens,
            max_attempts=args.max_attempts,retry_delay=args.retry_delay,
            evidence_policy=args.evidence_policy,groups_file=args.groups_file)
        print(f'Saved {result["completed"]}/{result["total"]} labels to {args.out}; '
              f'{result["labels_with_quality_warnings"]} need review; '
              f'{result["rejected_this_invocation"]} rejected attempts; '
              f'{result["failed_groups_this_invocation"]} groups exhausted their attempt limit')
    except KeyboardInterrupt:
        print('\nStopped; rerun the same command to resume.',file=sys.stderr);return 130
    except (ValueError,RuntimeError,OSError) as exc:
        print(str(exc),file=sys.stderr);return 1
    return 1 if result['failed_groups_this_invocation'] else 0
