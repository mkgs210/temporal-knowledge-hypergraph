"""A displayed branch must come from memberships, not a handpicked taxonomy."""
from copy import deepcopy

import pytest


def test_excerpt_uses_first_ids_and_real_parent_child_memberships(tmp_path):
    from tkh.cpu_figures import hierarchy_excerpt, draw_excerpt
    nodes=[{'id':'b','type':'method','surface_form':'Beta model'},
           {'id':'a','type':'method','surface_form':'Alpha model'},
           {'id':'c','type':'method','surface_form':'Gamma model'}]
    hierarchy={'cutoff':2026,'levels':[
        {'index':0,'groups':[{'id':'z','label':'Other topic','members':['c']},
                            {'id':'a','label':'First topic','members':['a','b']}]},
        {'index':1,'groups':[{'id':'f2','label':'Second detail','parent_id':'a','members':['b']},
                            {'id':'f1','label':'First detail','parent_id':'a','members':['a']},
                            {'id':'f3','label':'Other detail','parent_id':'z','members':['c']}]}]}
    example=hierarchy_excerpt({'cutoff':2026,'nodes':nodes},hierarchy)
    assert example['coarse']['id']=='a' and example['coarse']['size']==2
    assert [g['id'] for g in example['fine']]==['f1','f2']
    assert [g['leaf']['text'] for g in example['fine']]==['Alpha model','Beta model']
    draw_excerpt(example,tmp_path)
    assert (tmp_path/'cpu_hierarchy.png').stat().st_size>1000
    assert (tmp_path/'cpu_hierarchy.svg').read_text().startswith('<?xml')
    broken=deepcopy(hierarchy);broken['levels'][1]['groups'][1]['members']=['c']
    with pytest.raises(ValueError,match='parent|member|partition'):
        hierarchy_excerpt({'cutoff':2026,'nodes':nodes},broken)


def test_component_table_reads_measured_values_and_never_inherits_label_scores():
    from tkh.cpu_report import component_table
    row={'snapshots':[{'citation_coherence':{'1':{'excess':.125}}}],
         'transitions':[{'levels':[{'ari_shared_old_nodes':1.},{'ari_shared_old_nodes':.75}]}],
         'retrieval':{'tree_8_32':{'tracks':{'A':{'metrics':{'30':
             {'hits':3,'targets_all':7,'recall_all_micro':3/7}}}}}}}
    text=component_table({'variants':{'full':row,'no_native_cost':deepcopy(row),'unbounded_edits':deepcopy(row)}})
    assert '0.125' in text and '0.750' in text and '3/7' in text
    assert 'Full' in text and 'No native cost' in text and 'Unbounded edits' in text
