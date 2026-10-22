"""A fixed display budget must survive a hundred additional source papers."""
from copy import deepcopy
import json

import numpy as np

from tkh.divisive_hierarchy import build_divisive, update_divisive, DivisiveState
from tkh.temporal_input import temporal_view
from tkh.validation import validate_run


def article_corpus(count):
    nodes, edges, vectors = [], [], {}
    for paper in range(count):
        year = 2022 if paper < 30 else 2023 + (paper - 30) // 20
        source = paper + 1
        members = []
        for offset in range(4):
            nid = f'concept-{paper:03d}-{offset}'
            members.append(nid)
            nodes.append({'id': nid, 'type': 'method', 'surface_form': f'Topic {paper % 12} method {paper}-{offset}',
                          'first_seen_year': year, 'provenance': {'articles': [source]}})
            vector = np.zeros(12); vector[paper % 12] = 1.
            vectors[nid] = vector
        nid = f'paper-{paper:03d}'
        nodes.append({'id': nid, 'type': 'article', 'surface_form': f'Source paper {paper}',
                      'first_seen_year': year, 'provenance': {'articles': [source]}})
        vectors[nid] = vectors[members[0]].copy()
        for relation, endpoints in [('claims', members), ('presents', [nid] + members)]:
            edges.append({'id': f'{relation}-{paper}', 'relation_type': relation, 'members': endpoints,
                          'year': year, 'provenance': {'article_id': source, 'article_year': year}})
    return {'nodes': nodes, 'hyperedges': edges}, vectors


def check_hundred_article_growth(tmp_path, updater):
    graph, vectors = article_corpus(130)
    run = tmp_path / 'run'; run.mkdir()
    years = []
    initial = None
    previous = None
    for year in range(2022, 2028):
        view = temporal_view(graph, year)
        x = np.stack([vectors[n['id']] for n in view['nodes']])
        if previous is None:
            current = build_divisive(view, x, budgets=(12, 50), gamma=.03)
            initial = deepcopy(current)
        else:
            current = updater(previous, view, x)
            assert current.trace['initial_builder_calls'] == 0
            assert current.trace['resumed_from_saved_state']
        assert current.config['budgets'] == [12, 50]
        hierarchy, label_inputs = current.export()
        assert hierarchy['budgets'] == [12, 50, len(view['nodes'])]
        assert len(hierarchy['levels'][0]['groups']) <= 12
        assert len(hierarchy['levels'][1]['groups']) <= 50
        assert len(hierarchy['levels'][2]['groups']) == len(view['nodes'])
        assert set(initial.assignment) <= set(current.assignment)
        current_index = {n['id']: i for i, n in enumerate(view['nodes'])}
        for i, node in enumerate(initial.view['nodes']):
            np.testing.assert_array_equal(current.vectors[current_index[node['id']]], initial.vectors[i])
        snap = run / 'snapshots' / str(year); snap.mkdir(parents=True)
        for name, value in [('input', view), ('hierarchy', hierarchy),
                            ('label_inputs', label_inputs), ('temporal_events', current.events)]:
            (snap / f'{name}.json').write_text(json.dumps(value))
        years.append(year)
        (run / 'protocol.json').write_text(json.dumps({'budgets': [12, 50], 'years': years}))
        assert validate_run(run) == []
        current.save(snap / 'checkpoint')
        previous = DivisiveState.load(snap / 'checkpoint')
    assert sum(n['type'] == 'article' for n in current.view['nodes']) == 130
    assert sum(n['type'] == 'article' for n in initial.view['nodes']) == 30
    assert len(current.assignment) - len(initial.assignment) == 500
    return run


def test_five_batches_of_twenty_new_articles_preserve_original_12_50_budgets(tmp_path):
    run = check_hundred_article_growth(tmp_path, update_divisive)
    # An output cannot quietly authorize larger limits for itself.
    path = run / 'snapshots/2027/hierarchy.json'
    hierarchy = json.loads(path.read_text())
    hierarchy['budgets'][:2] = [13, 51]
    path.write_text(json.dumps(hierarchy))
    assert any('budget' in error for error in validate_run(run))
