"""Small, user-run label repair followed by a blinded paired development audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .gemini_labels import (ROOT, DEFAULT_RUN, digest, write, run_labels,
                            extract_label, extract_batch_labels)
from .scientific_labels import support_documents


def _read(run, relative):
    path = (run/relative).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError('Recorded artifact is outside its run')
    return json.loads(path.read_text())


def _labels(run, jobs):
    """Check the actual raw replies, not just editable completed metadata."""
    result = {}
    for path in sorted((run/'completed').glob('*.json')):
        row = json.loads(path.read_text()); key = (row['year'], row['level'], row['id'])
        if key not in jobs or key in result or row['prompt_sha256'] != jobs[key]['prompt_sha256']:
            raise ValueError('Label record has duplicate, missing or changed prompt input')
        request = _read(run, row['request_file'])
        if hashlib.sha256(request['prompt'].encode()).hexdigest() != row['batch_prompt_sha256']:
            raise ValueError('Recorded request prompt changed')
        if row['id'] not in request['group_ids']:
            raise ValueError('Recorded request does not contain this group')
        response = _read(run, row['response_file'])
        nvidia = row.get('provider') == 'nvidia'
        if len(request['group_ids']) > 1:
            parsed = extract_batch_labels(response, set(request['group_ids']),
                                          length_policy='warn', envelope=nvidia)[row['id']]
        else:
            parsed = extract_label(response, length_policy='warn', structured=nvidia)
        label, raw = parsed
        if raw != row['raw_output'] or any(row[k] != label[k] for k in label):
            raise ValueError('Label record text differs from the recorded response')
        result[key] = row
    return result


def _freeze(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Review input changed; choose a fresh output directory: {path}')
    write(path, value)


def prepare_paired_review(baseline, repaired):
    baseline, repaired = Path(baseline).resolve(), Path(repaired).resolve()
    if baseline == repaired:
        raise ValueError('Original and repaired runs must be separate')
    old_jobs = {(j['year'], j['level'], j['id']): j for j in _read(baseline, 'prompts.json')}
    new_jobs = {(j['year'], j['level'], j['id']): j for j in _read(repaired, 'prompts.json')}
    protocol = _read(repaired, 'protocol.json')
    if protocol.get('evidence_policy_id') != 'full_scope':
        raise ValueError('Paired scope review requires full_scope evidence')
    old_labels, new_labels = _labels(baseline, old_jobs), _labels(repaired, new_jobs)
    if not new_jobs or set(new_labels) != set(new_jobs):
        raise ValueError('Finish all selected label generations before paired review')
    packet, key_rows = [], []
    for key, job in sorted(new_jobs.items()):
        if key not in old_jobs or key not in old_labels:
            raise ValueError('Baseline label missing for selected group')
        old = old_jobs[key]
        if any(job[k] != old[k] for k in ('input_sha256', 'hierarchy_sha256', 'member_count')):
            raise ValueError('Source graph or hierarchy membership changed between label variants')
        if job['input']['member_ids'] != old['input']['member_ids']:
            raise ValueError('Group membership changed between label variants')
        if job['input']['support'] != old['input']['support']:
            raise ValueError('Held-out evidence split changed between label variants')
        evidence = job['evidence']
        # All visible names plus genuinely withheld claim-type member sentences.
        # The misleading legacy field name is retained for checker compatibility;
        # explicit roles and the paired protocol state which text was visible.
        names = [{**n, 'evidence_role': 'visible_member_name'} for n in evidence['members']]
        claims = [{**n, 'evidence_role': 'withheld_member_claim'}
                  for n in job['input']['support']['nodes'] if n['type'] == 'claim']
        support = support_documents(job['input'])
        pair_id = digest(list(key))[:20]
        for variant, row in [('original', old_labels[key]), ('repaired', new_labels[key])]:
            task_id = f'paired-{len(packet):03d}'
            packet.append({'task_id': task_id, 'cutoff': job['year'], 'level': job['level'],
                'label': row['label'], 'gloss': row['gloss'],
                'held_out_members': names + claims, 'held_out_claims': support,
                'assessment_design': {
                    'scope': 'Nonclaim member names were visible to the repaired generator.',
                    'claims': 'Only support-side claim-type sentences are withheld.',
                    'scientific_member_count': evidence['scope_summary']['scientific_member_count'],
                    'visible_nonclaim_members': len(names)},
                'pair_id': pair_id, 'variant': variant})
            key_rows.append({'task_id': task_id, 'year': key[0], 'level': key[1],
                             'group_id': key[2], 'pair_id': pair_id, 'variant': variant})
    design = {'schema_version': 'paired-label-review-v1', 'sample_groups': len(new_jobs),
        'variants': ['original', 'repaired'], 'same_evidence_for_both_variants': True,
        'scope_names_hidden_from_repaired_generator': False,
        'support_claim_texts_hidden_from_both_generators': True,
        'is_independent_final_test': False,
        'sample_role': 'Previously inspected development groups; no population estimate or final holdout claim.',
        'baseline_protocol_sha256': digest(_read(baseline, 'protocol.json')),
        'repaired_protocol_sha256': digest(protocol), 'packet_sha256': digest(packet),
        'limitations': [
            'Visible member names overlap construction and label inputs; scope ratings are a model audit, not unseen-data validation.',
            'Claim-type support text is withheld; adjacent claims can describe an example rather than the whole group.',
            'This development sample has already informed the repair; improvements do not estimate all-group quality.',
            'The original20-group source-held-out ratings remain unchanged and are not directly comparable to this new evidence protocol.']}
    audit = repaired/'audit'
    _freeze(audit/'diagnostic_packet.json', packet)
    _freeze(audit/'diagnostic_key.private.json', key_rows)
    _freeze(audit/'review_design.json', design)
    return design


def write_comparison(out, summary):
    variants = summary.get('variants', {})
    lines = ['# Paired label repair audit', '',
        'Original and repaired labels, same group and exact checking evidence; versions hidden from the judge.',
        'Previously inspected development sample. No population error rate, significance or retrieval gain is established.', '',
        '| Variant | Rated labels | Accurate | Wrong | Vague | Insufficient | Too narrow | Too generic |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for variant in ('original', 'repaired'):
        r = variants.get(variant, {}); ratings = r.get('ratings', {}); coverage = r.get('coverage', {})
        cells = [variant, r.get('count', 0), *[ratings.get(k, 0) for k in
                 ('accurate', 'wrong', 'vague', 'insufficient_evidence')],
                 coverage.get('too_narrow', 0), coverage.get('too_generic', 0)]
        lines.append('| ' + ' | '.join(map(str, cells)) + ' |')
    lines += ['', f'Synthetic sensitivity controls passed: {summary["control_gate_passed"]}.',
              'Coverage and rating are separate dimensions; do not add their counts.',
              'If controls fail, interpret neither variant as a reliable quality assessment.']
    Path(out, 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=ROOT/'out/nvidia_guided_labels_v2')
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--out', type=Path, default=ROOT/'out/nvidia_scope_labels_pilot_v3')
    p.add_argument('--groups-file', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--request-timeout',type=float,default=25.)
    p.add_argument('--time-budget',type=float,default=300.)
    args = p.parse_args()
    selection = args.groups_file or args.baseline/'audit/diagnostic_key.private.json'
    try:
        result = run_labels(args.run, args.out, provider='nvidia',
            model='nvidia/nemotron-3.5-lightning-30b-a3b', evidence_policy='full_scope',
            groups_file=selection, prepare_only=args.prepare_only)
        if args.prepare_only:
            from .label_judging import _controls, _scope_controls
            controls=len(_controls())+len(_scope_controls())
            print(f'Offline preparation only: {result["total"]} groups; '
                  f'{result["planned_requests"]} initial generation requests; '
                  f'{2*result["total"]+controls} paired checking requests, with {controls} controls first. '
                  'Completed generations are reused. No API called.')
            return 0
        if result['completed'] != result['total']:
            print('Generation incomplete; rerun to resume. Checking has not started.', file=sys.stderr)
            return 1
        prepare_paired_review(args.baseline, args.out)
        from .label_judging import run_judging
        check_out = args.out/'audit/api_check_v2'
        summary = run_judging(args.out, check_out, controls_first=True,workers=args.workers,
            request_timeout=args.request_timeout,time_budget=args.time_budget)
        write_comparison(check_out, summary)
        print(f'Paired development results: {check_out}/comparison.md')
        return 0 if summary['completed'] == summary['planned_requests'] and summary['control_gate_passed'] else 1
    except KeyboardInterrupt:
        print('Stopped; rerun the same command to resume.', file=sys.stderr); return 130
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
