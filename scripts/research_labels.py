#!/usr/bin/env python3
"""Local label/support pilot. No API, training, gold questions or frozen outputs.

Optional inference environment: torch, transformers, sentencepiece. Model weights
are pinned and read locally. Qwen weights may be fetched beforehand with hf CLI.
The output is research evidence, not a validated scientific overclaim metric.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import time

from tkh.scientific_labels import (build_label_input, parse_label_output,
                                  support_documents, calibrate_checker,
                                  assess_support, summarize_support, cross_group_controls, extract_model_label,
                                  attach_generated_labels)

GENERATOR = 'Qwen/Qwen2.5-1.5B-Instruct'
GEN_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
CHECKER = 'lytang/MiniCheck-Flan-T5-Large'
CHECK_REVISION = '96eafd01cee2d16cf81aaa2fb226b14f422a37b3'

# Hand-authored controls precede all pilot outputs. They are not gold-label data
# and do not calibrate scientific gloss correctness; they test basic sensitivity.
CONTROLS = [
    ('A graph neural network predicts molecular energy from atoms and bonds.',
     'The graph neural network predicts molecular energy.',
     'The method predicts musical rhythm and does not use atoms or bonds.'),
    ('The dataset contains crystal structures and their formation energies.',
     'Crystal structures and formation energies are included in the dataset.',
     'The dataset contains only spoken sentences and no crystal structures.'),
    ('The model predicts atomic forces using a differentiable energy function.',
     'A differentiable energy function is used to predict atomic forces.',
     'The model cannot predict atomic forces and does not use an energy function.'),
    ('The method uses density functional theory to compute electronic structure.',
     'Electronic structure is computed using density functional theory.',
     'The method does not use density functional theory and only classifies photographs.'),
]


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def run(args):
    global GENERATOR,GEN_REVISION
    if args.generator=='qwen3-1.7b':
        GENERATOR='Qwen/Qwen3-1.7B'
        GEN_REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graph = json.loads(Path(args.input).read_text())
    hierarchy = json.loads(Path(args.hierarchy).read_text())
    groups = []
    for level in hierarchy['levels'][:2]:
        candidates = sorted(level['groups'], key=lambda x: hashlib.sha256(x['id'].encode()).hexdigest())
        if level['index'] == 1 and args.fine_limit:
            candidates = candidates[:args.fine_limit]
        groups.extend((level['index'], x) for x in candidates)
    protocol = {
        'generator': GENERATOR, 'generator_revision': GEN_REVISION,
        'checker': CHECKER, 'checker_revision': CHECK_REVISION,
        'input_path': args.input, 'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'hierarchy_path': args.hierarchy,
        'hierarchy_sha256': hashlib.sha256(Path(args.hierarchy).read_bytes()).hexdigest(),
        'cutoff': graph['cutoff'], 'seed': args.seed,
        'selected_groups': [{'level': l, 'id': g['id']} for l, g in groups],
        'group_selection': 'all coarse; hash-sorted first fine_limit fine groups, chosen before inference',
        'max_generation_members': 24,
        'selection_policy':args.selection_policy,
        'generation': {'do_sample': False, 'max_new_tokens': 220, 'dtype': 'bfloat16',
                       'enable_thinking':False},
        'checker_max_tokens': 512, 'checker_truncation': False,
        'controls': CONTROLS, 'python': platform.python_version(),
        'checker_decision_bands': {'supported_at_least': .8, 'unsupported_at_most': .2,
                                 'selection': 'fixed during baseline pilot; sensitivity bands, not calibrated probabilities'},
        'checker_evidence': 'held-out member claim sentences plus source-safe incident claims hyperedges',
        'source_code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path('src/tkh/scientific_labels.py')]},
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'limitations': ['Existing 2022 baseline hierarchy, not repaired hierarchy results.',
                       'No full paper texts; only extracted facts available.',
                       'Both evidence halves were visible to clustering; source split only separates label generation/checking.',
                       'Synthetic controls are a sensitivity test, not human gold calibration.',
                       'Model pretraining can contain later knowledge; prompts constrain evidence but cannot guarantee absence.',
                       'Support existence does not establish group coverage or factual truth.',
                       'No human-validated overclaim rate or blind vague/wrong labels is claimed.'],
        'sources': ['https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct',
                    'https://huggingface.co/lytang/MiniCheck-Flan-T5-Large',
                    'https://github.com/Liyan06/MiniCheck/blob/main/minicheck/inference.py'],
    }
    if args.generated:
        if args.recheck_existing or not args.generator_protocol:
            raise ValueError('Imported generation requires its actual protocol and a new checking run')
        imported_protocol=json.loads(Path(args.generator_protocol).read_text())
        protocol.update(generator='external recorded assistant condition',generator_revision=None,
            imported_generator_protocol=imported_protocol,
            imported_generation_sha256=hashlib.sha256(Path(args.generated).read_bytes()).hexdigest())
    if args.recheck_existing:
        old_protocol = json.loads((out / 'protocol.json').read_text())
        for key in ('input_sha256', 'hierarchy_sha256', 'generator_revision', 'seed'):
            if old_protocol[key] != protocol[key]:
                raise ValueError('stored generation provenance differs: ' + key)
    write(out / 'protocol.json', protocol)
    start = time.monotonic()
    if args.generated:
        expected=[{'id':g['id'],'level':level,'member_count':len(g['members']),
            'input':build_label_input(graph,g['members'],graph['cutoff'],seed=args.seed,
                selection_policy=args.selection_policy)} for level,g in groups]
        records=attach_generated_labels(expected,json.loads(Path(args.generated).read_text()))
        write(out/'labels.json',records)
    elif args.recheck_existing:
        records = json.loads((out / 'labels.json').read_text())
        expected = {(l, g['id']) for l, g in groups}
        if {(r['level'], r['id']) for r in records} != expected:
            raise ValueError('stored pilot groups differ from selected groups')
        group_by_id = {g['id']: g for _, g in groups}
        for rec in records:
            updated_input = build_label_input(graph, group_by_id[rec['id']]['members'], graph['cutoff'], seed=args.seed,
                                             selection_policy=args.selection_policy)
            if updated_input['prompt_sha256'] != rec['input']['prompt_sha256']:
                raise ValueError('generation prompt changed; cannot reuse existing output')
            rec['input'] = updated_input
            try:
                extracted = extract_model_label(rec.get('raw_output', ''))
                rec.update(extracted)
                rec.pop('generation_error', None)
            except ValueError as exc:
                rec['generation_error'] = str(exc)
    else:
        gen_path = snapshot_download(GENERATOR, revision=GEN_REVISION, local_files_only=True)
        tok = AutoTokenizer.from_pretrained(gen_path)
        model = AutoModelForCausalLM.from_pretrained(gen_path, torch_dtype=torch.bfloat16).to(args.device).eval()
        records = []
        for level, group in groups:
            inp = build_label_input(graph, group['members'], graph['cutoff'], seed=args.seed,
                                    selection_policy=args.selection_policy)
            rec = {'id': group['id'], 'level': level, 'input': inp, 'member_count': len(group['members'])}
            prompt = tok.apply_chat_template([{'role': 'user', 'content': inp['prompt']}], tokenize=False,
                                             add_generation_prompt=True,enable_thinking=False)
            tokens = tok(prompt, return_tensors='pt').to(args.device)
            rec['chat_prompt'] = prompt
            rec['input_token_count'] = tokens['input_ids'].shape[1]
            if rec['input_token_count'] > 8192:
                rec['generation_error'] = 'input exceeds fixed 8192-token pilot limit; not truncated'
            else:
                with torch.inference_mode():
                    generated = model.generate(**tokens, do_sample=False, max_new_tokens=220,
                                               pad_token_id=tok.eos_token_id)
                raw = tok.decode(generated[0, tokens['input_ids'].shape[1]:], skip_special_tokens=True)
                rec['raw_output'] = raw
                try:
                    rec.update(extract_model_label(raw))
                except ValueError as exc:
                    rec['generation_error'] = str(exc)
            records.append(rec)
            write(out / 'labels.json', records)
            print('generated', level, group['id'], rec.get('label', rec.get('generation_error')), flush=True)
        del model, tok
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    check_path = snapshot_download(CHECKER, revision=CHECK_REVISION, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(check_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(check_path, torch_dtype=torch.float32).to(args.device).eval()

    def score(document, claim):
        text = 'predict: ' + document + tok.eos_token + claim
        inputs = tok(text, return_tensors='pt', truncation=False).to(args.device)
        length = inputs['input_ids'].shape[1]
        if length > 512:
            return {'input': text, 'tokens': length, 'score': None, 'error': 'exceeds 512 tokens; not truncated'}
        with torch.inference_mode():
            logits = model(**inputs, decoder_input_ids=torch.zeros((1, 1), dtype=torch.long, device=args.device)).logits[0, 0]
            # MiniCheck's documented two-class decoder labels: unsupported=3, supported=209.
            value = logits[[3, 209]].float().softmax(-1)[1].item()
        return {'input': text, 'tokens': length, 'score': value}

    controls = []
    for doc, pos, neg in CONTROLS:
        controls += [{'kind': 'positive', **score(doc, pos)}, {'kind': 'negative', **score(doc, neg)}]
    calibration = calibrate_checker([x['score'] for x in controls if x['kind'] == 'positive'],
                                    [x['score'] for x in controls if x['kind'] == 'negative'])
    write(out / 'checker_controls.json', {'records': controls, 'calibration': calibration})
    for rec in records:
        label = rec.get('label')
        checked = []
        if label:
            for doc in support_documents(rec['input']):
                checked.append({**doc, **score(doc['text'], label['gloss'])})
        rec['support_checks'] = checked
        rec['assessment'] = assess_support(label['gloss'] if label else '',
                                           [x['score'] for x in checked if x['score'] is not None], calibration,
                                           generation_status=label['status'] if label else 'insufficient')
        write(out / 'labels.json', records)
        print('checked', rec['id'], rec['assessment']['status'], len(checked), flush=True)
    controls_by_group = cross_group_controls(records)
    by_id = {r['id']: r for r in records}
    for control in controls_by_group:
        control['checks'] = [dict(doc, **score(doc['text'], control['gloss']))
                             for doc in support_documents(by_id[control['recipient_id']]['input'])]
        control['assessment'] = assess_support(control['gloss'],
            [x['score'] for x in control['checks'] if x['score'] is not None], calibration)
    write(out / 'cross_group_controls.json', controls_by_group)
    # A real blind judge can distinguish accurate, vague, wrong and insufficient.
    # No model-generated responses are inserted into this uncompleted rating pack.
    tasks, key = [], []
    candidates = [(r['id'], r['label']['gloss'], 'own') for r in records if r.get('label')]
    candidates += [(r['recipient_id'], r['gloss'], 'cross_group') for r in controls_by_group]
    random.Random(args.seed).shuffle(candidates)
    for index, (recipient, gloss, condition) in enumerate(candidates):
        rec = by_id[recipient]
        tid = 'blind-%03d' % index
        tasks.append({'task_id': tid, 'gloss': gloss,
                      'held_out_member_names': [n['surface_form'] for n in rec['input']['support']['nodes']],
                      'evidence': support_documents(rec['input']),
                      'rating': None, 'evidence_ids_justifying_rating': [],
                      'allowed_ratings': ['accurate', 'vague', 'wrong', 'insufficient_evidence']})
        key.append({'task_id': tid, 'recipient_id': recipient, 'condition': condition})
    write(out / 'blind_rating_packet.json', {
        'instructions': 'Judge only supplied evidence; broad vacuous labels are vague, not accurate. No evidence is insufficient, not accurate. Rate scientific claims and theme coverage separately in your justification.',
        'completed': False, 'tasks': tasks})
    write(out / 'blind_rating_key.private.json', key)
    summary = {'overall': summarize_support([r['assessment'] for r in records]),
               'levels': {str(l): summarize_support([r['assessment'] for r in records if r['level'] == l]) for l in (0, 1)},
               'generation_errors': sum('generation_error' in r for r in records),
               'format_noncompliance': sum(r.get('format_compliant') is False for r in records),
               'cross_group_control_summary': summarize_support([r['assessment'] for r in controls_by_group]),
               'calibration': calibration, 'elapsed_seconds': time.monotonic() - start,
               'limitations': protocol['limitations']}
    write(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='results/snapshots/2022/input.json')
    p.add_argument('--hierarchy', default='results/snapshots/2022/hierarchy.json')
    p.add_argument('--out', default='out/repair-labels/baseline-2022')
    p.add_argument('--seed', type=int, default=7001)
    p.add_argument('--fine-limit', type=int, default=8, help='0 uses all fine groups')
    p.add_argument('--device', default='cuda')
    p.add_argument('--recheck-existing', action='store_true', help='Preserve generation; rerun format parsing/checking only')
    p.add_argument('--generated',help='Exact prompt-bound JSON from an explicitly recorded different generator')
    p.add_argument('--generator-protocol',help='Actual provenance of imported generation; never mislabel as local Qwen')
    p.add_argument('--selection-policy',choices=('hashed_members','science_evidence'),default='hashed_members')
    p.add_argument('--generator',choices=('qwen2.5-1.5b','qwen3-1.7b'),default='qwen2.5-1.5b')
    run(p.parse_args())
