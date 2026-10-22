"""Independent structural validation of persisted hierarchy experiments.

Inputs and frozen protocol are the authority, never group counts in output.
No algorithm, encoder, or retrieval module is imported here.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / 'schema' / 'hierarchy.schema.json'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _read(path, errors):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        errors.append(f'{path}: unreadable JSON: {exc}')
        return None


def _events(payload, current, previous, nodes, year, errors):
    if not isinstance(payload, list):
        errors.append(f'{year}: events must be an array')
        return
    for i, event in enumerate(payload):
        prefix = f'{year}: event {i}'
        if not isinstance(event, dict):
            errors.append(f'{prefix}: event must be an object')
            continue
        kind = event.get('kind')
        level = event.get('level')
        gid = event.get('group_id')
        if not isinstance(kind, str) or kind not in {'birth', 'growth', 'move', 'merge', 'split', 'death'}:
            errors.append(f'{prefix}: unknown kind {kind!r}')
        if not isinstance(level, int) or isinstance(level, bool) or level < 0:
            errors.append(f'{prefix}: invalid level')
            continue
        target = previous if kind == 'death' else current
        if not isinstance(gid, str) or target.get(gid) != level:
            errors.append(f'{prefix}: group {gid!r} absent at level {level} in referenced snapshot')
        old_ids = event.get('previous_group_ids')
        if not isinstance(old_ids, list) or any(not isinstance(g, str) for g in old_ids):
            errors.append(f'{prefix}: previous_group_ids must be an ID array')
        else:
            for old in old_ids:
                if previous.get(old) != level:
                    errors.append(f'{prefix}: previous group {old!r} absent at level {level}')
        node_ids = event.get('node_ids')
        if not isinstance(node_ids, list) or any(not isinstance(n, str) or n not in nodes for n in node_ids):
            errors.append(f'{prefix}: unknown or malformed event node IDs')


def validate_run(directory):
    """Return one human-readable string per violation; never mutate artifacts."""
    directory = Path(directory)
    errors = []
    protocol = _read(directory / 'protocol.json', errors)
    if not isinstance(protocol, dict):
        return errors or ['protocol must be an object']
    budgets = protocol.get('budgets')
    years = protocol.get('years')
    if (not isinstance(budgets, list) or not budgets or
            any(not isinstance(b, int) or isinstance(b, bool) or b < 1 for b in budgets)):
        errors.append('protocol: budgets must be positive integer array')
    if (not isinstance(years, list) or not years or
            any(not isinstance(y, int) or isinstance(y, bool) for y in years) or
            len(set(years)) != len(years) or years != sorted(years)):
        errors.append('protocol: years must be a nonempty sorted unique integer array')
    if errors:
        return errors
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding='utf-8'))
        schema_validator = Draft202012Validator(schema)
    except (OSError, ValueError) as exc:
        return [f'schema: {exc}']
    actual_dirs = {p.name for p in (directory / 'snapshots').glob('*') if p.is_dir()}
    for extra in sorted(actual_dirs - {str(y) for y in years}):
        errors.append(f'unexpected snapshot directory {extra}')
    previous = {}
    for year in years:
        snap = directory / 'snapshots' / str(year)
        if not snap.is_dir():
            errors.append(f'{year}: missing snapshot directory')
            continue
        graph = _read(snap / 'input.json', errors)
        hierarchy = _read(snap / 'hierarchy.json', errors)
        if not isinstance(graph, dict) or not isinstance(hierarchy, dict):
            errors.append(f'{year}: input and hierarchy must be objects')
            continue
        schema_errors = sorted(schema_validator.iter_errors(hierarchy), key=lambda e: str(list(e.path)))
        for exc in schema_errors:
            errors.append(f'{year}: schema {list(exc.path)}: {exc.message}')
        if schema_errors:
            continue
        if graph.get('cutoff') != year or hierarchy.get('cutoff') != year:
            errors.append(f'{year}: cutoff differs from protocol snapshot year')
        rows = graph.get('nodes')
        edges = graph.get('hyperedges')
        if not isinstance(rows, list) or any(not isinstance(n, dict) or not isinstance(n.get('id'), str) for n in rows):
            errors.append(f'{year}: malformed input nodes')
            continue
        ids = [n['id'] for n in rows]
        if len(set(ids)) != len(ids):
            errors.append(f'{year}: duplicate input node IDs')
        node_map = {n['id']: n for n in rows}
        nodes = set(ids)
        for n in rows:
            for field in ('available_year', 'observed_at'):
                available = n.get(field)
                if field in n and type(available) is not int:
                    errors.append(f'{year}: invalid input node date {n["id"]}: {field}')
                if isinstance(available, int) and available > year:
                    errors.append(f'{year}: future input node {n["id"]}: {field}={available}')
        if not isinstance(edges, list):
            errors.append(f'{year}: input hyperedges must be an array')
        else:
            edge_ids = []
            for edge in edges:
                if (not isinstance(edge, dict) or not isinstance(edge.get('id'), str) or
                    not isinstance(edge.get('members'), list) or not edge['members']):
                    errors.append(f'{year}: malformed input edge')
                    continue
                edge_ids.append(edge['id'])
                if any(not isinstance(n, str) or n not in nodes for n in edge['members']):
                    errors.append(f'{year}: input edge {edge["id"]} has unknown member')
                dates = [edge.get('available_year'), edge.get('year')]
                provenance = edge.get('provenance', {})
                if isinstance(provenance, dict):
                    dates.append(provenance.get('article_year'))
                if any(isinstance(value, int) and value > year for value in dates):
                    errors.append(f'{year}: future input edge {edge["id"]}')
            if len(set(edge_ids)) != len(edge_ids):
                errors.append(f'{year}: duplicate input edge IDs')
        levels = hierarchy['levels']
        expected_budgets = budgets + [len(nodes)]
        if hierarchy['budgets'] != expected_budgets:
            errors.append(f'{year}: output budget differs from trusted protocol/singleton policy')
        if len(levels) != len(expected_budgets):
            errors.append(f'{year}: level count differs from budget policy')
        current = {}
        previous_level = {}
        for index, level in enumerate(levels):
            if level['index'] != index:
                errors.append(f'{year}: level index {level["index"]} is not position {index}')
            groups = level['groups']
            if index < len(expected_budgets) and len(groups) > expected_budgets[index]:
                errors.append(f'{year}: budget overflow level {index}: {len(groups)} > {expected_budgets[index]}')
            counts = Counter(n for g in groups for n in g['members'])
            if set(counts) != nodes or any(c != 1 for c in counts.values()):
                missing = sorted(nodes - set(counts))[:5]
                repeated = sorted(n for n, c in counts.items() if c != 1)[:5]
                errors.append(f'{year}: partition level {index}: missing={missing}, repeated={repeated}')
            unknown = set(counts) - nodes
            if unknown:
                errors.append(f'{year}: unknown member level {index}: {sorted(unknown)[:5]}')
            children = Counter()
            this_level = {}
            for group in groups:
                gid = group['id']
                members = set(group['members'])
                if gid in current:
                    errors.append(f'{year}: duplicate group ID {gid!r}')
                current[gid] = index
                this_level[gid] = members
                if index == 0:
                    if group['parent_id'] is not None:
                        errors.append(f'{year}: top group {gid} must have null parent')
                else:
                    parent = group['parent_id']
                    if parent not in previous_level:
                        errors.append(f'{year}: dangling/wrong-level parent {parent!r} for {gid}')
                    elif not members <= previous_level[parent]:
                        errors.append(f'{year}: laminar child {gid} is not contained in parent {parent}')
                    children.update((parent, n) for n in members)
                if index == len(levels) - 1 and len(members) != 1:
                    errors.append(f'{year}: finest group {gid} is not singleton')
                if not set(group['evidence']) <= members:
                    errors.append(f'{year}: evidence of {gid} is not in its members')
            if index:
                for parent, members in previous_level.items():
                    actual = {n for (p, n), c in children.items() if p == parent}
                    if actual != members:
                        errors.append(f'{year}: laminar children do not cover parent {parent}')
            previous_level = this_level
        label_inputs = _read(snap / 'label_inputs.json', errors)
        if isinstance(label_inputs, dict):
            if set(label_inputs) != set(current):
                errors.append(f'{year}: label input group IDs differ from hierarchy')
            for level in levels:
                for group in level['groups']:
                    gid = group['id']
                    payload = label_inputs.get(gid)
                    if payload is None or canonical_hash(payload) != group['label_input_hash']:
                        errors.append(f'{year}: label input hash mismatch for {gid}')
                    expected = [{k: node_map[n].get(k) for k in ('id', 'type', 'surface_form')}
                                for n in sorted(group['members']) if n in node_map]
                    if payload != expected:
                        errors.append(f'{year}: label input evidence text/membership mismatch for {gid}')
        elif label_inputs is not None:
            errors.append(f'{year}: label_inputs must be an object')
        events_path = snap / 'temporal_events.json'
        if not events_path.exists() and (snap / 'events.json').exists():
            events_path = snap / 'events.json'
        events = _read(events_path, errors)
        if events is not None:
            _events(events, current, previous, nodes, year, errors)
        previous = current
    return errors


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print('usage: python validate.py OUTPUT_DIRECTORY')
        return 2
    violations = validate_run(args[0])
    for error in violations:
        print(error.replace('\n', ' '))
    if not violations:
        print('PASS: all protocol snapshots satisfy schema, partitions, budgets, evidence and event references')
    return int(bool(violations))
