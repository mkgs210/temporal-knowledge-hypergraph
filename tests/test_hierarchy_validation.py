"""Independent acceptance checks over persisted outputs, including broken hand fixtures."""
import json
from pathlib import Path
import subprocess
import sys
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'hierarchy_fixtures'


def write_run(tmp_path, hierarchy=None):
    run = tmp_path / 'run'
    snap = run / 'snapshots' / '2022'
    snap.mkdir(parents=True)
    (run / 'protocol.json').write_text(json.dumps({'budgets': [1, 2], 'years': [2022]}))
    (snap / 'input.json').write_text(json.dumps({
        'cutoff': 2022, 'nodes': [{'id': key, 'type': 'method', 'surface_form': key} for key in ['a', 'b', 'c']],
        'hyperedges': [{'id': 'e', 'members': ['a', 'b', 'c']}]}))
    if hierarchy is None:
        hierarchy = json.loads((FIXTURES / 'valid.json').read_text())
    (snap / 'hierarchy.json').write_text(json.dumps(hierarchy))
    (snap / 'label_inputs.json').write_text(json.dumps({
        group['id']: [{'id': key, 'type': 'method', 'surface_form': key} for key in sorted(group['members'])]
        for level in hierarchy['levels'] for group in level['groups']}))
    (snap / 'temporal_events.json').write_text('[]')
    return run, snap


def validator(run):
    return subprocess.run([sys.executable, str(ROOT / 'validate.py'), str(run)],
                          capture_output=True, text=True, cwd='/tmp')


def test_standalone_accepts_valid_all_node_laminar_hierarchy(tmp_path):
    run, _ = write_run(tmp_path)
    result = validator(run)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('broken', [None,'hierarchy','labelled_hierarchy'])
def test_standalone_validates_both_structures_from_cpu_output_root(tmp_path,broken):
    run,_=write_run(tmp_path/'initial');output=tmp_path/'cpu';output.mkdir()
    shutil.copytree(run,output/'hierarchy');shutil.copytree(run,output/'labelled_hierarchy')
    if broken:
        path=output/broken/'snapshots/2022/hierarchy.json'
        path.write_text((FIXTURES/'budget.json').read_text())
    result=validator(output)
    assert result.returncode==(1 if broken else 0),result.stdout+result.stderr
    if broken:assert 'budget' in result.stdout.lower(),result.stdout


@pytest.mark.parametrize('name,violation', [
    ('overlap', 'partition'), ('budget', 'budget'), ('dangling_parent', 'parent'),
])
def test_cli_rejects_handwritten_broken_fixtures(tmp_path, name, violation):
    run, _ = write_run(tmp_path, json.loads((FIXTURES / f'{name}.json').read_text()))
    result = validator(run)
    assert result.returncode == 1
    assert violation in result.stdout.lower(), result.stdout + result.stderr


def test_standalone_accepts_release_with_only_published_hierarchy(tmp_path):
    run,_=write_run(tmp_path/'initial')
    release=tmp_path/'release';release.mkdir()
    shutil.copytree(run,release/'labelled_hierarchy')
    result=validator(release)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('mutation,violation', [
    ('missing_node', 'partition'), ('future_node', 'unknown member'),
    ('wrong_parent', 'laminar'), ('wrong_budget', 'budget'),
    ('wrong_cutoff', 'cutoff'), ('not_singleton', 'singleton'),
    ('duplicate_group_id', 'duplicate group'), ('blank_gloss', 'schema'),
    ('future_evidence', 'evidence'), ('bad_hash', 'label input hash'),
])
def test_rejects_specific_semantic_failures(tmp_path, mutation, violation):
    run, snap = write_run(tmp_path)
    hierarchy = json.loads((snap / 'hierarchy.json').read_text())
    if mutation == 'missing_node':
        hierarchy['levels'][1]['groups'][1]['members'] = ['b']
    elif mutation == 'future_node':
        hierarchy['levels'][1]['groups'][1]['members'].append('future')
    elif mutation == 'wrong_parent':
        hierarchy['levels'][2]['groups'][2]['parent_id'] = 'mid1'
    elif mutation == 'wrong_budget':
        hierarchy['budgets'][0] = 99
    elif mutation == 'wrong_cutoff':
        hierarchy['cutoff'] = 2023
    elif mutation == 'not_singleton':
        hierarchy['levels'][2]['groups'][1]['members'] = ['b', 'c']
    elif mutation == 'duplicate_group_id':
        hierarchy['levels'][2]['groups'][0]['id'] = 'mid1'
    elif mutation == 'blank_gloss':
        hierarchy['levels'][0]['groups'][0]['gloss'] = ' '
    elif mutation == 'future_evidence':
        hierarchy['levels'][0]['groups'][0]['evidence'] = ['future']
    elif mutation == 'bad_hash':
        (snap / 'label_inputs.json').write_text(json.dumps({'top': {'text': 'changed evidence'}}))
    (snap / 'hierarchy.json').write_text(json.dumps(hierarchy))
    result = validator(run)
    assert result.returncode == 1
    assert violation in result.stdout.lower(), result.stdout + result.stderr


def test_validator_rejects_missing_expected_snapshot(tmp_path):
    run, _ = write_run(tmp_path)
    (run / 'protocol.json').write_text(json.dumps({'budgets': [1, 2], 'years': [2022, 2023]}))
    result = validator(run)
    assert result.returncode == 1
    assert 'missing snapshot' in result.stdout.lower()


def test_events_cannot_refer_to_nonexistent_current_group(tmp_path):
    run, snap = write_run(tmp_path)
    (snap / 'temporal_events.json').write_text(json.dumps([
        {'kind': 'growth', 'level': 1, 'group_id': 'absent', 'node_ids': ['a'], 'previous_group_ids': []}]))
    result = validator(run)
    assert result.returncode == 1
    assert 'event' in result.stdout.lower()


def test_cross_snapshot_events_reference_actual_previous_ids(tmp_path):
    run, snap = write_run(tmp_path)
    (run / 'protocol.json').write_text(json.dumps({'budgets': [1, 2], 'years': [2022, 2023]}))
    later = run / 'snapshots' / '2023'
    later.mkdir()
    for filename in ['input.json', 'hierarchy.json']:
        payload = json.loads((snap / filename).read_text())
        payload['cutoff'] = 2023
        (later / filename).write_text(json.dumps(payload))
    (later / 'label_inputs.json').write_bytes((snap / 'label_inputs.json').read_bytes())
    events = [{'kind': 'growth', 'level': 1, 'group_id': 'mid1', 'node_ids': ['a'], 'previous_group_ids': ['missing']}]
    (later / 'temporal_events.json').write_text(json.dumps(events))
    result = validator(run)
    assert result.returncode == 1
    assert 'previous' in result.stdout.lower()
    events[0]['previous_group_ids'] = ['mid1']
    (later / 'temporal_events.json').write_text(json.dumps(events))
    assert validator(run).returncode == 0


def test_unknown_future_input_metadata_and_invalid_input_edges_rejected(tmp_path):
    run, snap = write_run(tmp_path)
    payload = json.loads((snap / 'input.json').read_text())
    payload['nodes'][0]['available_year'] = 2023
    payload['hyperedges'][0]['members'].append('unknown')
    (snap / 'input.json').write_text(json.dumps(payload))
    result = validator(run)
    assert result.returncode == 1
    assert 'future' in result.stdout.lower()
    assert 'input edge' in result.stdout.lower()


@pytest.mark.parametrize('payload', [
    [{'kind': 'growth', 'level': 1, 'group_id': {}, 'node_ids': ['a'], 'previous_group_ids': []}],
    [{'kind': 'growth', 'level': 1, 'group_id': 'mid1', 'node_ids': [{}], 'previous_group_ids': []}],
    [{'kind': 'growth', 'level': 1, 'group_id': 'mid1', 'node_ids': ['a'], 'previous_group_ids': [[]]}],
])
def test_malformed_event_ids_return_violations_without_traceback(tmp_path, payload):
    run, snap = write_run(tmp_path)
    (snap / 'temporal_events.json').write_text(json.dumps(payload))
    result = validator(run)
    assert result.returncode == 1
    assert 'event' in result.stdout.lower()
    assert 'traceback' not in result.stderr.lower()


def test_missing_label_input_file_cannot_pass(tmp_path):
    run, snap = write_run(tmp_path)
    (snap / 'label_inputs.json').unlink()
    result = validator(run)
    assert result.returncode == 1
    assert 'label_inputs.json' in result.stdout


def test_schema_rejects_duplicate_members(tmp_path):
    run, snap = write_run(tmp_path)
    hierarchy = json.loads((snap / 'hierarchy.json').read_text())
    hierarchy['levels'][0]['groups'][0]['members'].append('a')
    (snap / 'hierarchy.json').write_text(json.dumps(hierarchy))
    result = validator(run)
    assert result.returncode == 1
    assert 'schema' in result.stdout


@pytest.mark.parametrize('target,field', [('node', 'observed_at'), ('edge', 'year'), ('edge', 'article_year')])
def test_future_input_dates_rejected_even_when_membership_is_consistent(tmp_path, target, field):
    run, snap = write_run(tmp_path)
    payload = json.loads((snap / 'input.json').read_text())
    if target == 'node':
        payload['nodes'][0][field] = 2023
    elif field == 'article_year':
        payload['hyperedges'][0]['provenance'] = {field: 2023}
    else:
        payload['hyperedges'][0][field] = 2023
    (snap / 'input.json').write_text(json.dumps(payload))
    result = validator(run)
    assert result.returncode == 1
    assert 'future' in result.stdout.lower()


def test_malformed_observation_date_cannot_bypass_cutoff(tmp_path):
    run, snap = write_run(tmp_path)
    payload = json.loads((snap / 'input.json').read_text())
    payload['nodes'][0]['observed_at'] = '2023'
    (snap / 'input.json').write_text(json.dumps(payload))
    result = validator(run)
    assert result.returncode == 1
    assert 'invalid' in result.stdout.lower()
