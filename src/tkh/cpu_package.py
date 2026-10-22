"""Build a local review ZIP bound to current CPU metrics and generated reports."""
import argparse
import json
from pathlib import Path
import zipfile

from .cpu_label_audit import publish_labels
from .submission_pipeline import _sha
from .validation import validate_run


def package_run(root,run,destination):
    root,run,destination=map(lambda p:Path(p).resolve(),(root,run,destination))
    if destination.is_relative_to(run):raise ValueError('Archive must be outside the run directory')
    registry=json.loads((run/'report_artifacts.json').read_text())
    if registry['metrics_sha256']!=_sha(run/'metrics.json'):
        raise ValueError('Report metrics changed; regenerate report before packaging')
    reports={}
    for name,entry in registry['files'].items():
        path=root/name
        if not path.is_file():path=Path(entry['path'])
        if _sha(path)!=entry['sha256']:raise ValueError('Generated report/figure changed')
        reports[name]=path
    if not {'report.md','report.pdf'}<=registry['files'].keys():raise ValueError('Generate both report formats before packaging')
    if not (run/'labels/audit/api_check/summary.json').is_file():
        raise ValueError('Complete the T5 audit before packaging its reproducibility checkpoint')
    publish_labels(run)
    if registry['metrics_sha256']!=_sha(run/'metrics.json'):
        raise ValueError('Label audit changed metrics; regenerate report')
    if json.loads((run/'metrics.json').read_text()).get('components'):
        from .cpu_components import verify_components
        verify_components(root/'data',run)
    if validate_run(run/'labelled_hierarchy'):raise ValueError('Published hierarchy is invalid')
    protocol=json.loads((run/'hierarchy/protocol.json').read_text())
    if _sha(root/'data/tkh_collection10.json')!=protocol['input_sha256']:
        raise ValueError('Packaged raw data differs from evaluated input')
    for name in ('README.md','AI_USAGE.md','Makefile','pyproject.toml','requirements-cpu.lock',
                 'requirements.lock','report.cpu.template.md','make_tables.py','validate.py',
                 'src/tkh/cpu_pipeline.py','scripts/name_groups.py','schema/hierarchy.schema.json'):
        if not (root/name).is_file():raise ValueError('Required runnable source file is missing: '+name)
    files={}
    for name in ('.gitignore','.env.example','README.md','AI_USAGE.md','Makefile','pyproject.toml',
        'requirements.lock','requirements-cpu.lock','make_tables.py','validate.py',
        'report.template.md','report.cpu.template.md',
        'PhD_Multi_Resolution_Semantic_Abstraction_over_an_Evolving_Knowledge.md'):
        if (root/name).is_file():files[name]=root/name
    def collect(directory,prefix):
        for path in directory.rglob('*'):
            if (path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts
                    and not any(p.endswith('.egg-info') for p in path.parts)
                    and path.suffix not in {'.pyc','.sqlite3'} and path.name!='.env'):
                files[prefix+str(path.relative_to(directory))]=path
    for name in ('src','tests','schema','data','prompts','scripts'):
        if (root/name).is_dir():collect(root/name,name+'/')
    if (root/'annotations').is_dir():collect(root/'annotations','annotations/')
    for name in ('CODE_ORIGIN.md','CPU_RUN.md','REPRODUCIBILITY.md','REQUIREMENTS.md','COMPONENT_CONTROLS.md'):
        path=root/'docs'/name
        if path.is_file():files['docs/'+name]=path
    collect(run,'results/')
    files.update(reports)
    manifest={'files_sha256':{name:_sha(path) for name,path in sorted(files.items())},
        'policy':'Fresh CPU run and authenticated labels; no old E5 assets, model weights, .env or text cache.',
        'scientific_status':'Review artifact; consult measured gaps and failed controls before submission.'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):archive.write(path,name)
        archive.writestr('submission_manifest.json',json.dumps(manifest,indent=2))
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('out/cpu'))
    p.add_argument('--output',type=Path)
    args=p.parse_args();destination=args.output or args.run.with_name(args.run.name+'-submission.zip')
    print(package_run(Path(__file__).resolve().parents[2],args.run,destination))
