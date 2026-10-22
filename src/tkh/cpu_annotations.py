"""Generate T5 annotations or replay exact archived evidence without a secret key."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from .gemini_labels import ROOT,DEFAULT_PROMPT,digest,load_api_key,run_labels,write
from .label_repair import _labels
from .submission_pipeline import _sha


def resolve_checkpoint(checkpoint,*,cache_dir='.cache/tkh/annotations'):
    """Unpack the small, authenticated response archive; never fetch a model."""
    checkpoint=Path(checkpoint)
    if checkpoint.is_dir():return checkpoint
    expected=checkpoint.with_name(checkpoint.name+'.sha256').read_text().split()[0]
    actual=_sha(checkpoint)
    if actual!=expected:raise ValueError('Annotation archive checksum mismatch')
    target=Path(cache_dir)/actual
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as pending:
            with tarfile.open(checkpoint,'r:gz') as archive:
                archive.extractall(pending,filter='data')
            if not (Path(pending)/'labels/protocol.json').is_file():
                raise ValueError('Annotation archive has no labels protocol')
            Path(pending).rename(target)
    return target/'labels'


def replay_labels(checkpoint,hierarchy,out,*,prompt_path=DEFAULT_PROMPT):
    checkpoint,hierarchy,out=map(Path,(checkpoint,hierarchy,out))
    protocol=json.loads((checkpoint/'protocol.json').read_text())
    jobs=json.loads((checkpoint/'prompts.json').read_text())
    if digest(jobs)!=protocol['jobs_sha256']:raise ValueError('Archived label inputs changed')
    known={(j['year'],j['level'],j['id']):j for j in jobs}
    records=_labels(checkpoint,known)
    if set(records)!=set(known):raise ValueError('Archived labels are incomplete')
    generation=protocol['generation']
    options={'provider':protocol['provider'],'model':protocol['model'],'evidence_policy':'full_scope',
        'prompt_path':prompt_path,'batch_size':protocol['batch_size'],'max_batch_chars':protocol['max_batch_chars'],
        'max_output_tokens':generation.get('max_tokens',generation.get('maxOutputTokens',512)),
        'decoding':generation.get('response_format','guided_json'),'prepare_only':True,
        'thinking':generation.get('enable_thinking',False),'length_policy':protocol['length_policy'],
        'max_attempts':protocol['retry_policy']['max_attempts_per_group'],
        'retry_delay':protocol['retry_policy']['initial_delay_seconds']}
    run_labels(hierarchy,out,**options)
    current=json.loads((out/'protocol.json').read_text())
    if current['jobs_sha256']!=protocol['jobs_sha256'] or current['generation']!=generation:
        raise ValueError('Saved annotation evidence/hierarchy differs; provide NVIDIA_API_KEY for new inputs')
    for directory in ('completed','requests','responses','audit'):
        source=checkpoint/directory
        if source.exists():shutil.copytree(source,out/directory,dirs_exist_ok=True)
    write(out/'replayed_from.json',{'source_protocol_sha256':_sha(checkpoint/'protocol.json'),
        'jobs_sha256':protocol['jobs_sha256'],'new_labeller_calls':0,
        'policy':'Exact graph, hierarchy, evidence and prompts; original responses retained, not regenerated.'})
    return run_labels(hierarchy,out,**options)


def available_key():
    try:load_api_key(ROOT/'.env',variable='NVIDIA_API_KEY');return True
    except ValueError:return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('out/cpu'))
    default=Path('annotations/nvidia_labels.tar.gz')
    p.add_argument('--checkpoint',type=Path,default=default if default.is_file() else Path('results/labels'))
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--replay-only',action='store_true',help='Use authenticated saved responses even if a key exists')
    args=p.parse_args()
    try:
        live=False if args.replay_only else available_key()
        if not live:args.checkpoint=resolve_checkpoint(args.checkpoint)
        if args.preflight:
            if not live and not (args.checkpoint/'audit/api_check/summary.json').is_file():
                raise ValueError('Full T5 needs NVIDIA_API_KEY or a completed results/labels checkpoint; make offline runs the local-only study')
            return 0
        if live:
            result=run_labels(args.run/'hierarchy',args.run/'labels',provider='nvidia',
                model='nvidia/nemotron-3.5-lightning-30b-a3b',evidence_policy='full_scope',batch_size=3,rpm=0)
        else:
            result=replay_labels(args.checkpoint,args.run/'hierarchy',args.run/'labels')
        if result['completed']!=result['total']:raise RuntimeError('Label generation is incomplete; repeat make labels to resume')
    except (ValueError,RuntimeError,OSError) as exc:p.exit(1,str(exc)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
