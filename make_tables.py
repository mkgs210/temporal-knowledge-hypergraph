"""Regenerate every report table, figure, Markdown and optional PDF from metrics."""
import argparse
import json
from pathlib import Path


def main():
    from tkh.submission_report import render_report,figures,write_pdf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--report',type=Path,default=Path('report.md'))
    parser.add_argument('--pdf',type=Path)
    args=parser.parse_args()
    metrics=json.loads((args.run/'metrics.json').read_text())
    if metrics.get('schema_version')=='cpu-run-v1':
        from tkh.cpu_report import render_report as render_cpu,figures as cpu_figures
        args.report.parent.mkdir(parents=True,exist_ok=True)
        cpu_figures(metrics,args.report.parent/'figures')
        rendered=render_cpu(metrics)
        args.report.write_text(rendered,encoding='utf-8')
        (args.run/'report.md').write_text(rendered,encoding='utf-8')
        if args.pdf:write_pdf(rendered,args.pdf,args.report.parent)
        from tkh.submission_pipeline import _sha,_write
        artifacts={'report.md':args.report,'figures/cpu_retrieval.png':args.report.parent/'figures/cpu_retrieval.png'}
        if metrics.get('hierarchy_example'):
            artifacts.update({f'figures/cpu_hierarchy.{ext}':args.report.parent/f'figures/cpu_hierarchy.{ext}'
                              for ext in ('png','svg','pdf')})
        if args.pdf:artifacts['report.pdf']=args.pdf
        _write(args.run/'report_artifacts.json',{'metrics_sha256':_sha(args.run/'metrics.json'),
            'files':{name:{'path':str(path.resolve()),'sha256':_sha(path)} for name,path in artifacts.items()}})
        print('Generated CPU measurements',args.report,args.pdf or '')
        return
    if metrics.get('schema_version')!='submission-report-v1':raise ValueError('Expected submission metrics schema')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    figures(metrics,args.report.parent/'figures')
    template=Path(__file__).with_name('report.template.md').read_text()
    rendered=render_report(metrics,template)
    args.report.write_text(rendered,encoding='utf-8')
    (args.run/'report.md').write_text(rendered,encoding='utf-8')
    if args.pdf:write_pdf(rendered,args.pdf,args.report.parent)
    print('Generated',args.report,args.pdf or '')


if __name__=='__main__':main()
