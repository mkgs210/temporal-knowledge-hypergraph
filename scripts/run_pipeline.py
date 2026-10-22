"""One-command CPU reproduction, with wall-clock timings and a complete log."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def pipeline_steps(out):
    run=str(out)
    return [
        ('annotation_preflight',['-m','tkh.cpu_annotations','--run',run,'--preflight','--replay-only']),
        ('build',['-m','tkh.cpu_pipeline','--out',run,'--build-only']),
        ('validate',['validate.py',str(out/'hierarchy')]),
        ('retrieval',['-m','tkh.cpu_pipeline','--out',run,'--assess-only']),
        ('scientific_checks',['-m','tkh.cpu_science',run]),
        ('component_controls',['-m','tkh.cpu_components',run]),
        ('labels',['-m','tkh.cpu_annotations','--run',run,'--replay-only']),
        ('label_audit',['-m','tkh.cpu_label_audit','--run',run,'--check','--replay-only']),
        ('validate_labels',['validate.py',str(out/'labelled_hierarchy')]),
        ('hierarchy_figure',['-m','tkh.cpu_figures',run]),
        ('report',['make_tables.py',run,'--report','report.md','--pdf','report.pdf']),
        ('package',['-m','tkh.cpu_package','--run',run])]


def run_steps(root,python,out,steps=None):
    root,out=Path(root).resolve(),Path(out)
    if not out.is_absolute():out=root/out
    out.parent.mkdir(parents=True,exist_ok=True)
    timing_path=out.with_name(out.name+'-timing.json')
    log_path=out.with_name(out.name+'.log')
    env={**os.environ,'PYTHONPATH':str(root/'src'),'OPENBLAS_NUM_THREADS':'4',
         'PYTHONUNBUFFERED':'1'}
    started=time.perf_counter()
    result={'started_at_utc':datetime.now(timezone.utc).isoformat(),'python':str(python),
        'output':str(out),'annotation_mode':'authenticated archive; no new API requests',
        'steps':[],'exit_code':None}
    with log_path.open('w',encoding='utf-8') as log:
        for name,args in pipeline_steps(out) if steps is None else steps:
            command=[str(python),*map(str,args)]
            header=f'[{name}] {shlex.join(command)}\n'
            print(header,end='',flush=True);log.write(header);log.flush()
            begin=time.perf_counter()
            with subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line)
                code=process.wait()
            elapsed=time.perf_counter()-begin
            result['steps'].append({'name':name,'command':command,'elapsed_seconds':elapsed,'exit_code':code})
            result.update(elapsed_seconds=time.perf_counter()-started,exit_code=code)
            timing_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
            footer=f'[{name}] finished in {elapsed:.2f}s; exit {code}\n'
            print(footer,end='',flush=True);log.write(footer);log.flush()
            if code:break
    result.update(elapsed_seconds=time.perf_counter()-started,
                  finished_at_utc=datetime.now(timezone.utc).isoformat())
    timing_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'Total: {result["elapsed_seconds"]:.2f}s; timings: {timing_path}',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',default='.venv-cpu/bin/python')
    parser.add_argument('--out',type=Path,default=Path('out/cpu'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    python=Path(args.python)
    if not python.is_absolute():python=root/python
    out=args.out if args.out.is_absolute() else root/args.out
    if out.exists():
        out=out.with_name(out.name+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    setup=[]
    if not python.exists():
        if python!=root/'.venv-cpu/bin/python':parser.error('Specified Python does not exist')
        setup=[('create_environment',['-m','venv',str(python.parent.parent)]),
               ('install_dependencies',['-m','pip','install','-r','requirements-cpu.lock']),
               ('install_project',['-m','pip','install','--no-deps','--no-build-isolation','-e','.'])]
    started=time.perf_counter()
    # Environment creation needs the already available bootstrap Python.
    if setup:
        first=run_steps(root,sys.executable,out.with_name(out.name+'-setup'),setup[:1])
        if first['exit_code']:return first['exit_code']
        install=run_steps(root,python,out.with_name(out.name+'-install'),setup[1:])
        if install['exit_code']:return install['exit_code']
    result=run_steps(root,python,out)
    result['total_including_setup_seconds']=time.perf_counter()-started
    result['environment_created']=bool(setup)
    result['setup_steps']=(first['steps']+install['steps']) if setup else []
    out.with_name(out.name+'-timing.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'Full command including setup: {result["total_including_setup_seconds"]:.2f}s',flush=True)
    return result['exit_code']


if __name__=='__main__':raise SystemExit(main())
