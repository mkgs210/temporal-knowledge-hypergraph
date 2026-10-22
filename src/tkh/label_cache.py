"""Authenticate recorded LLM explanations against fresh source-separated inputs.

Offline replay validates provenance and syntax, not scientific correctness.
Annotation never changes a partition, objective, event, or search score.
"""
from copy import deepcopy
from .scientific_labels import build_label_input, parse_label_output


def annotate_labels(view, hierarchy, records, generator_protocol):
    groups={(level['index'],g['id']):g for level in hierarchy['levels'][:2]
            for g in level['groups']}
    keys=[(r['level'],r['id']) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=set(groups):
        raise ValueError('Recorded labels do not uniquely cover current upper groups')
    annotated=deepcopy(hierarchy)
    targets={(level['index'],g['id']):g for level in annotated['levels'][:2]
             for g in level['groups']}
    artifacts=[]
    for record in records:
        key=(record['level'],record['id']); group=groups[key]
        inp=build_label_input(view,group['members'],view['cutoff'],selection_policy='science_evidence')
        if record.get('prompt_sha256')!=inp['prompt_sha256'] or record.get('member_count')!=len(group['members']):
            raise ValueError('Stale label prompt or member count')
        label=parse_label_output(record['raw_output'])
        if any(record.get(k)!=v for k,v in label.items()):
            raise ValueError('Label text differs from recorded generation')
        targets[key].update(label=label['label'],gloss=label['gloss'],label_status=label['status'],
                            generation_prompt_sha256=inp['prompt_sha256'])
        artifacts.append({'id':key[1],'level':key[0],'input':inp,'output':label,
                          'raw_output':record['raw_output'],'member_count':len(group['members'])})
    return annotated,{'schema_version':'scientific-label-annotations-v1',
        'cutoff':view['cutoff'],'groups':artifacts,'generator_protocol':deepcopy(generator_protocol),
        'claim':'Exact recorded-input replay; scientific accuracy requires separate assessment.'}


def validate_annotations(view, hierarchy, artifact):
    if artifact.get('cutoff')!=view['cutoff']:
        raise ValueError('Scientific label cutoff mismatch')
    records=[]
    for row in artifact['groups']:
        output=parse_label_output(row['raw_output'])
        if output!=row['output']:
            raise ValueError('Scientific label output differs from original text')
        records.append({'id':row['id'],'level':row['level'],'member_count':row['member_count'],
                        'prompt_sha256':row['input']['prompt_sha256'],'raw_output':row['raw_output'],**output})
    expected,actual=annotate_labels(view,hierarchy,records,artifact['generator_protocol'])
    if expected!=hierarchy or actual!=artifact:
        raise ValueError('Scientific labels or exact evidence inputs were changed')
