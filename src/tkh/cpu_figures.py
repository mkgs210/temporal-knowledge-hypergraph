"""A deterministic, metrics-backed excerpt of the authenticated main hierarchy."""
from collections import Counter
import argparse
import json
from pathlib import Path
import textwrap

from .submission_pipeline import _sha, _write
from .validation import validate_run


def hierarchy_excerpt(view, hierarchy):
    nodes={n['id']:n for n in view['nodes']}
    levels={l['index']:l['groups'] for l in hierarchy['levels']}
    if hierarchy['cutoff']!=view['cutoff']:raise ValueError('Hierarchy cutoff differs')
    for level in (0,1):
        if Counter(n for g in levels[level] for n in g['members'])!=Counter(nodes.keys()):
            raise ValueError('Hierarchy is not a complete nonoverlapping partition')
    coarse={g['id']:g for g in levels[0]}
    for g in levels[1]:
        if g['parent_id'] not in coarse or not set(g['members'])<=set(coarse[g['parent_id']]['members']):
            raise ValueError('Fine members do not belong to their parent')
    first=min(levels[0],key=lambda g:g['id'])
    children=sorted([g for g in levels[1] if g['parent_id']==first['id']],key=lambda g:g['id'])[:2]
    if not children:raise ValueError('Coarse group has no fine children')
    priority={'method':0,'dataset':1,'technique':2,'component':3,'cited_work':4,'article':99,'author':100}
    def group(g):return {'id':g['id'],'label':g['label'],'size':len(g['members'])}
    fine=[]
    for g in children:
        node=min((nodes[n] for n in g['members']),key=lambda n:(priority.get(n['type'],20),n['id']))
        fine.append({**group(g),'parent_id':first['id'],
                     'leaf':{'id':node['id'],'type':node['type'],'text':node['surface_form']}})
    return {'cutoff':view['cutoff'],'coarse':group(first),'fine':fine,
        'total_coarse':len(levels[0]),'total_fine':len(levels[1]),
        'selection':'First persistent coarse ID, first two child IDs, first method ID per child (typed fallback). No selection by quality.',
        'interpretation':'Illustrative subset; omitted siblings and other branches remain in the full hierarchy. Labels are generated descriptions, not expert-verified topic truth.'}


def draw_excerpt(example, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def wrap(text,width,lines):
        parts=textwrap.wrap(text,width=width,break_long_words=True,break_on_hyphens=False)
        return '\n'.join(parts if len(parts)<=lines else parts[:lines-1]+[textwrap.shorten(' '.join(parts[lines-1:]),width=width,placeholder='…')])
    with plt.rc_context({'font.size':8,'svg.fonttype':'none','pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(3.4,3.7));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
        def box(x,y,w,h,text,color):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',
                                      linewidth=.8,edgecolor='#333333',facecolor=color))
            ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
        g=example['coarse']
        ax.text(.5,.97,f'{example["cutoff"]}: one branch of the saved hierarchy',ha='center',fontsize=9)
        box(.035,.70,.93,.20,wrap(g['label'],43,3)+f'\n{g["id"]} · {g["size"]} records','#dbe9f4')
        count=len(example['fine']);width=.44 if count==2 else .75
        positions=[.025,.535] if count==2 else [.125]
        for x,g in zip(positions,example['fine'],strict=True):
            mid=x+width/2
            ax.annotate('',xy=(mid,.61),xytext=(.5,.685),arrowprops={'arrowstyle':'->','lw':.8})
            box(x,.31,width,.29,wrap(g['label'],20 if count==2 else 36,5)+f'\n{g["size"]} records','#f0f0f0')
            ax.annotate('',xy=(mid,.23),xytext=(mid,.295),arrowprops={'arrowstyle':'->','lw':.8})
            node=g['leaf']
            box(x,.065,width,.15,wrap(node['text'],20 if count==2 else 36,3)+f'\n{node["id"]}','white')
        ax.text(.5,.015,'Arrows show membership; most branches are omitted.',ha='center',fontsize=7)
        for ext in ('png','svg','pdf'):fig.savefig(directory/f'cpu_hierarchy.{ext}',dpi=300,facecolor='white')
        plt.close(fig)


def attach_excerpt(run):
    run=Path(run);root=run/'labelled_hierarchy'
    errors=validate_run(root)
    if errors:raise ValueError('Cannot illustrate invalid labelled hierarchy: '+'; '.join(errors))
    metrics=json.loads((run/'metrics.json').read_text())
    year=metrics['construction']['protocol']['years'][-1]
    folder=root/'snapshots'/str(year)
    view=json.loads((folder/'input.json').read_text());hierarchy=json.loads((folder/'hierarchy.json').read_text())
    example=hierarchy_excerpt(view,hierarchy)
    example['source_sha256']={'input':_sha(folder/'input.json'),'hierarchy':_sha(folder/'hierarchy.json')}
    metrics['hierarchy_example']=example;_write(run/'metrics.json',metrics)
    return example


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    attach_excerpt(p.parse_args().run)
