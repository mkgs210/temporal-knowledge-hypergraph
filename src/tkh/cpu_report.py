"""Generate a fresh CPU measurement report; never import historical scores."""
from pathlib import Path

from .submission_report import table

NAMES={'bm25':'BM25','minilm':'MiniLM','bm25_minilm':'BM25 + MiniLM',
    'bm25_native_sources':'BM25 + native/source',
    'bm25_native_sources_hierarchy':'BM25 + native/source + hierarchy',
    'bm25_minilm_native_sources':'BM25 + MiniLM + native/source',
    'bm25_minilm_native_sources_hierarchy':'BM25 + MiniLM + native/source + hierarchy',
    'tree_2_8':'Tree 2 coarse / 8 fine','tree_8_32':'Tree 8 coarse / 32 fine','tree_all':'Tree all branches'}

def retrieval_rows(metrics):
    rows = []
    for name, report in metrics['retrieval']['arms'].items():
        track = report['tracks'].get('A')
        if track and '30' in track['metrics']:
            value = track['metrics']['30']
            rows.append([NAMES.get(name,name), f'{value["hits"]}/{value["targets_all"]}', f'{value["recall_all_micro"]:.1%}'])
    return rows


def component_table(components):
    rows=[]
    for key,name in [('full','Full'),('no_native_cost','No native cost'),('unbounded_edits','Unbounded edits')]:
        row=components['variants'][key]
        coherence=row['snapshots'][-1]['citation_coherence']['1']['excess']
        aris=[t['levels'][1]['ari_shared_old_nodes'] for t in row['transitions']]
        score=row['retrieval']['tree_8_32']['tracks']['A']['metrics']['30']
        rows.append([name,'unavailable' if coherence is None else f'{coherence:.3f}',
            'unavailable' if not aris else f'{sum(aris)/len(aris):.3f}',f'{score["hits"]}/{score["targets_all"]}'])
    return table(['Variant','Citation excess (fine)','Mean yearly ARI (fine)','Tree hits@30'],rows)


def render_report(metrics):
    if 'science' in metrics:return _scientific_report(metrics)
    construction = metrics['construction']
    counts = [[r['snapshot']['cutoff'], r['snapshot']['nodes'], r['snapshot']['hyperedges']]
              for r in construction['snapshots']]
    transitions = [[f'{t["from_cutoff"]}–{t["to_cutoff"]}',
        *[f'{t["levels"][i]["ari_shared_old_nodes"]:.4f}' for i in (0, 1)]]
        for t in construction['transitions']]
    budgets = construction['protocol']['budgets']
    lines = ['# CPU hierarchy experiment', '', '## Abstract', '',
        'This report records a new CPU experiment with a small pretrained sentence '
        'encoder, a bounded temporal hypergraph hierarchy and matched retrieval '
        'comparisons. All numerical results below come from this run; unmeasured '
        'scientific requirements are listed explicitly.', '', '<!-- ARTICLE BODY -->', '',
        '## Method', '',
        f'The encoder is {metrics["encoder"]["model_id"]}, running on CPU. '
        f'Display limits are {budgets[0]} coarse groups and {budgets[1]} fine groups in total.',
        'Own record names are encoded with a fixed pretrained model. The initial partition '
        'uses recursive balanced native hypergraph cuts over semantic neighbours and supplied '
        'scientific hyperedges. Later views load saved checkpoints and restrict changes to at '
        'most 10% of old scientific paths per update. The original hyperedges and endpoint '
        'multiplicities are retained. No model fine-tuning is performed.', '',
        '## Data and structural checks', '', table(['Year', 'Nodes', 'Hyperedges'], counts), '',
        'Every published snapshot passes the standalone structural validator. Publication-year '
        'views are retrospective approximations, not historical ingestion logs.', '',
        '## Cross-snapshot agreement', '', table(['Transition', 'Coarse ARI', 'Fine ARI'], transitions), '',
        'These are descriptive measurements; agreement alone does not establish useful adaptation.', '',
        '## Retrieval evaluation', '',
        'The primary outcome is micro Recall@30 over all Type A target occurrences. '
        'Predictions are saved and hashed before reference access. All arms use the same '
        'questions, matching rule and result types. The paired native/source baselines '
        'differ from their hierarchy variants only by the added group links. Tree variants '
        'instead select coarse/fine regions before scoring their leaf records.', '',
        table(['Search', 'Found / targets', 'Recall@30'], retrieval_rows(metrics)), '',
        '![Recall@30 comparison](figures/cpu_retrieval.png)', '',
        'No historical measurements are attached to this model. No retrieval improvement '
        'is assumed in advance; compare the paired rows above.', '',
        '## Unmeasured requirements and limits', '',
        *['- '+item for item in metrics['not_measured']], '',
        'This generated run report is not evidence that the full scientific assessment is complete. '
        'The model truncates inputs after 256 word-piece tokens. API group naming is a separate '
        'operation; existing retrieval does not read generated labels.', '']
    return '\n'.join(lines)


def _scientific_report(metrics):
    science=metrics['science'];construction=metrics['construction'];protocol=construction['protocol']
    def number(v):return 'unavailable' if v is None else f'{v:.3f}'
    def interval(values):return 'unavailable' if values is None else '['+', '.join(number(v) for v in values)+']'
    snapshots=table(['Year','Records','Native edges'],[
        [r['snapshot']['cutoff'],r['snapshot']['nodes'],r['snapshot']['hyperedges']] for r in construction['snapshots']])
    coherence=table(['Level','Citation score','Null mean','p (null)'],[
        [name,number(r['observed']),number(r['null_mean']),number(r['permutation_pvalue'])]
        for name,r in [('Coarse',science['snapshots'][-1]['citation_coherence']['0']),
                       ('Fine',science['snapshots'][-1]['citation_coherence']['1'])]])
    stability=table(['Transition / level','ARI','Conditional CI','p correspondence'],[
        [f'{r["uncertainty"]["previous_cutoff"]}–{r["uncertainty"]["current_cutoff"]} / {level["level"]}',
         number(level['point_ari']),interval(level['interval']),
         number(level['alignment_significance']['p_greater_equal'])]
        for r in science['temporal'] for level in r['uncertainty']['levels']])
    perturb=table(['Initial deletion','Mean ARI','Conditional 95%'],[
        [name,number(r['mean']),interval(None if r['low'] is None else [r['low'],r['high']])]
        for name,r in [('Coarse',science['perturbations']['summaries']['scientific']['0']),
                       ('Fine',science['perturbations']['summaries']['scientific']['1'])]])
    paired=table(['Comparison','Delta recall','Paired 95%'],[
        [NAMES.get(name,name),f'{r["delta_recall_at_30"]:+.3f}',interval(r['paired_question_bootstrap_95'])]
        for name,r in science['paired_retrieval'].items()])
    labels=metrics.get('labels');label_text='No complete API annotation and checking run has been attached. Label faithfulness is unmeasured.'
    if labels:
        check=labels.get('judge');design=labels['audit_design']
        label_text=(f'{labels["generated"]} group labels were authenticated. The fixed audit sample contains '
                    f'{design["sample_groups"]} groups; {design["sample_with_claim_evidence"]} have withheld claim evidence. ')
        label_text+=f'Labeller: {labels.get("generator_model","see saved protocol")}; checker: {labels.get("judge_model","unmeasured")}. '
        if check:
            label_text+='\n\n'+table(['Audit outcome','Count'],[[name,check['sample_ratings'].get(name,0)]
                for name in ('accurate','vague','wrong','insufficient_evidence')])
            label_text+=f'\n\nSensitivity controls passed: {check["control_gate_passed"]}. '
            label_text+=f'{check.get("rule_assessments",0)} empty-evidence assessments used an explicit deterministic rule, not a model call. '
            label_text+=f'Observed wrong fraction in the checked sample: {number(check["observed_wrong_fraction_in_sample"])} '
            label_text+=f'over {check["sample_size"]} rated labels. '
            label_text+=('These are model-based sample ratings, not population accuracy or human truth.' if labels['rates_interpretable']
                         else 'The checker is incomplete or failed controls; its apparent overclaim fraction is not a reliable estimate.')
    bm=metrics['retrieval']['arms']['bm25_native_sources']['tracks']['A']
    bg=metrics['retrieval']['arms']['bm25_native_sources_hierarchy']['tracks']['A']
    candidates=[r for key,r in science['paired_retrieval'].items() if not key.startswith('coarse_first')]
    supported=any(r['delta_recall_at_30']>0 and r['paired_question_bootstrap_95'][0]>0 for r in candidates)
    decision=('At least one group overlay has a positive difference with a positive conditional interval lower bound. '
              'That supports a development-set candidate, not an untouched-test guarantee; exact target swaps and tree losses remain part of the decision.'
              if supported else 'The study does not establish a reliable retrieval gain from the hierarchy. We would retain the stronger flat/native-source search '
              'as the retrieval default and expose the bounded hierarchy for inspection while treating traversal as experimental. '
              'This is a measured negative result for the tested configurations; the comparison was performed and no positive utility is claimed.')
    replacements={'NODES':construction['snapshots'][-1]['snapshot']['nodes'],
        'QUESTIONS':bm['questions'],'TARGETS':bm['metrics']['30']['targets_all'],
        'BASE_HITS':bm['metrics']['30']['hits'],'GROUP_HITS':bg['metrics']['30']['hits'],
        'COARSE':protocol['budgets'][0],'FINE':protocol['budgets'][1],
        'GAMMA':protocol['gamma'],'EDIT_FRACTION':f'{protocol["max_old_fraction"]:.0%}',
        'MODEL':metrics['encoder']['model_id'],'SNAPSHOTS':snapshots,'COHERENCE':coherence,
        'STABILITY':stability,'PERTURBATIONS':perturb,'LABELS':label_text,
        'RETRIEVAL':table(['Search','Found / targets','Recall@30'],retrieval_rows(metrics)),
        'PAIRED':paired,'DECISION':decision,'GAPS':'\n\n'.join(
            'No untouched final test is available; these questions were inspected during development.'
            if s.startswith('An untouched final test;') else s for s in metrics['not_measured'])}
    replacements['COMPONENTS']=''
    if metrics.get('components'):
        replacements['COMPONENTS']=(
            'Two predeclared component controls reuse the exact CPU vectors, cutoffs, budgets and search. '
            'The zero-native-cost control removes the scientific hyperedge cut term; provenance attachment and retrieval relations remain. '
            'Unbounded refinement removes the old-scientific edit cap and movement penalty, while retaining warm starts, fixed topic slots and saved affinities. '
            'Its initial partition is byte-identical to the full method. These are conditional component comparisons, not a new search for a winning configuration.\n\n'
            +component_table(metrics['components'])+'\n\n'
            'Citation excess is the final fine-level score minus its matched null. Yearly ARI is the descriptive mean across shared-old-record transitions, '
            'not independent replications. Tree retrieval uses the same 8-coarse/32-fine route in every row. '
            'Per-cutoff scores, five deletion rebuilds per control, movement audits, overlays and paired question intervals are saved in metrics. '
            'Changed controls receive no transferred labels or faithfulness scores; T5 applies only to the full method. '
            'We retain the preselected full method for bounded edits, without claiming that these controls establish a universally optimal trade-off. '
            +component_findings(metrics['components']))
    replacements['HIERARCHY_EXAMPLE']=''
    if metrics.get('hierarchy_example'):
        replacements['HIERARCHY_EXAMPLE']=(
            '![Saved hierarchy excerpt](figures/cpu_hierarchy.png)\n\n'
            '**Figure 1.** Actual final-cutoff branch, selected by persistent IDs rather than quality. '
            'Arrows show membership from coarse group through fine group to an original record. '
            'Most branches are omitted; long texts may be shortened in the illustration. Generated names are not expert topic judgments.')
    text=Path(__file__).resolve().parents[2].joinpath('report.cpu.template.md').read_text()
    for key,value in replacements.items():text=text.replace('{{'+key+'}}',str(value))
    if '{{' in text:raise ValueError('Unresolved scientific-report placeholder')
    return '\n'.join(line.rstrip() for line in text.splitlines())+'\n'


def component_findings(components):
    rows=components['variants']; full=rows['full']; no_native=rows['no_native_cost']
    def excess(row):return row['snapshots'][-1]['citation_coherence']['1']['excess']
    def hits(row):return row['retrieval']['tree_8_32']['tracks']['A']['metrics']['30']['hits']
    def agreement(row):
        values=[t['levels'][1]['ari_shared_old_nodes'] for t in row['transitions']]
        return sum(values)/len(values) if values else None
    findings=[]
    if excess(full) is not None and excess(no_native) is not None and excess(no_native)>=excess(full) and hits(no_native)>=hits(full):
        findings.append('Removing the native term does not lower fine citation excess or routed recall in this run; its inclusion therefore has no demonstrated advantage on these point estimates. '
                        'Native relational fidelity remains an explicit design requirement, not a quality claim inferred from this comparison.')
    if agreement(full) is not None and agreement(rows['unbounded_edits'])<agreement(full):
        findings.append('Removing the edit cap and penalty lowers chronological agreement. This supports their role in retaining old memberships, '
                        'but imposed stability does not establish scientifically correct adaptation.')
    return ' '.join(findings)


def figures(metrics, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = retrieval_rows(metrics)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.85))
    values = [report['tracks']['A']['metrics']['30']['recall_all_micro']
              for report in metrics['retrieval']['arms'].values()
              if report['tracks'].get('A') and '30' in report['tracks']['A']['metrics']]
    short={'BM25 + MiniLM + native/source':'Mixed + graph',
        'BM25 + MiniLM + native/source + hierarchy':'Mixed + graph + groups',
        'BM25 + native/source':'BM25 + graph',
        'BM25 + native/source + hierarchy':'BM25 + graph + groups',
        'Tree 2 coarse / 8 fine':'Tree 2/8','Tree 8 coarse / 32 fine':'Tree 8/32',
        'Tree all branches':'Tree all'}
    ax.barh([short.get(r[0],r[0]) for r in rows], values)
    ax.set_xlim(0, 1)
    ax.set_xticks([0,.5,1],['0%','50%','100%'])
    ax.tick_params(labelsize=7)
    ax.set_xlabel('Type A micro Recall@30',fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(directory/'cpu_retrieval.png', dpi=300)
    plt.close(fig)
    if metrics.get('hierarchy_example'):
        from .cpu_figures import draw_excerpt
        draw_excerpt(metrics['hierarchy_example'],directory)
