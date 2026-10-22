"""Exact floating KM1 accounting and pinned Mt-KaHyPar primitives.

Counts are endpoint multiplicities, while the objective depends on distinct
occupied parts. Semantic edges are already weighted arity-two edges here.
"""
from collections import Counter
from copy import deepcopy
import importlib.metadata
import math
import time

SCALE = 100_000_000
_INITIALIZER = None


def objective(edges, assignment):
    return math.fsum(float(e['weight']) * max(0, len({assignment[n] for n in e['members']}) - 1)
                     for e in edges)


def quotient(edges, assignment):
    result = []
    for edge in edges:
        counts = Counter()
        for node, count in edge.get('counts', dict.fromkeys(edge['members'], 1)).items():
            counts[assignment[node]] += count
        row = deepcopy({k:v for k,v in edge.items() if k not in ('members','counts')})
        row.update(members=sorted(counts, key=str), counts=dict(counts))
        result.append(row)
    return result


def boundary_quotient(edges, editable, assignment):
    """Contract fixed vertices by GLOBAL part, retaining one terminal per part."""
    editable = set(editable)
    outside = {group for n,group in assignment.items() if n not in editable}
    # Tuples avoid collisions with real string IDs, including adversarial names.
    terminals = {('__fixed_terminal__', g):g for g in outside}
    mapping = {n:n if n in editable else ('__fixed_terminal__', assignment[n])
               for e in edges for n in e['members']}
    return quotient(edges, mapping), terminals


def integer_weights(edges, scale=SCALE):
    if not isinstance(scale, int) or scale < 1:
        raise ValueError('Integer scale must be positive')
    weights, errors = [], []
    for edge in edges:
        weight = float(edge['weight'])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Hyperedge weights must be finite and nonnegative')
        scaled = weight * scale
        if not math.isfinite(scaled) or scaled > 2**31 - 1:
            raise ValueError('Integer hyperedge weight overflow')
        rounded = int(round(scaled))
        weights.append(rounded)
        errors.append(abs(rounded / scale - weight) * max(0, len(set(edge['members'])) - 1))
    if sum(weights) > 2**31 - 1:
        raise ValueError('Total integer hyperedge weight overflow')
    return weights, {'integer_scale':scale, 'rounding':'nearest_ties_to_even',
                     'rounding_error_bound':math.fsum(errors),
                     'positive_edges_rounded_to_zero':sum(float(e['weight']) > 0 and w == 0 for e,w in zip(edges,weights))}


def solve_partition(ids, edges, k, *, initial=None, fixed=None, masses=None,
                    seed=20261019, epsilon=.5, scale=SCALE, cycles=2, block_caps=None):
    """Partition or refine; reject invalid output and increasing floating cost.

    A zero-objective instance gets an explicitly marked deterministic legal
    assignment. It is never reported as a successful native optimization.
    """
    ids = list(ids)
    index = {n:i for i,n in enumerate(ids)}
    if len(index) != len(ids) or not ids or type(k) is not int or not 1 <= k <= len(ids):
        raise ValueError('Infeasible node IDs or partition budget')
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Balance epsilon must be finite and nonnegative')
    fixed = fixed or {}
    if any(n not in index or type(g) is not int or not 0 <= g < k for n,g in fixed.items()):
        raise ValueError('Invalid fixed part')
    if any(set(e['members']) - index.keys() for e in edges):
        raise ValueError('Unknown edge endpoint')
    masses = {n:1 for n in ids} if masses is None else masses
    if set(masses) != set(ids) or any(type(v) is not int or v < 1 for v in masses.values()):
        raise ValueError('Node masses must be positive integers')
    if sum(masses.values()) > 2**31 - 1:
        raise ValueError('Integer node weight overflow')
    if block_caps is not None and (not isinstance(block_caps, (list, tuple)) or
            len(block_caps) != k or any(type(v) is not int or not 1 <= v <= 2**31 - 1 for v in block_caps) or
            sum(block_caps) < sum(masses.values())):
        raise ValueError('Block caps must be k positive integer capacities covering total mass')
    if initial is not None and (set(initial) != set(ids) or set(initial.values()) != set(range(k)) or
                               any(initial[n] != g for n,g in fixed.items())):
        raise ValueError('Invalid initial partition')
    integers, numerical = integer_weights(edges, scale)
    trace = {'solver':'mtkahypar','version':importlib.metadata.version('mtkahypar'),
             'preset':'DETERMINISTIC_QUALITY','objective':'KM1','seed':seed,'epsilon':epsilon,
             'cycles':cycles, 'fixed_vertices':len(fixed), 'nodes':len(ids), 'edges':len(edges), **numerical}
    max_weight = math.floor((1 + epsilon) * math.ceil(sum(masses.values()) / k))
    trace['maximum_block_weight'] = max_weight
    caps = [max_weight] * k if block_caps is None else list(block_caps)
    if block_caps is not None:
        trace['block_caps'] = caps
        trace['maximum_block_weight'] = max(caps)
    positive = [(e,w) for e,w in zip(edges,integers) if w and len(set(e['members'])) >= 2]
    fixed_loads = [sum(masses[n] for n,g in fixed.items() if g == i) for i in range(k)]
    infeasible = max(masses.values()) > max(caps) or any(a > b for a,b in zip(fixed_loads,caps,strict=True))
    if initial is not None:
        initial_loads = [sum(masses[n] for n,g in initial.items() if g == i) for i in range(k)]
        trace['initial_balance_feasible'] = all(a <= b for a,b in zip(initial_loads,caps,strict=True))
        if not trace['initial_balance_feasible'] and (not positive or infeasible):
            # Saved assignments remain legal hierarchy paths when append growth
            # makes the requested refinement balance infeasible. Retain them as
            # an explicitly REJECTED operation, never an accepted balanced solve.
            cost = objective(edges,initial)
            trace.update(status='rejected_initial_imbalance', accepted=False,
                         retained_initial=True, balance_feasible=False,
                         block_weights=initial_loads, objective_before=cost,
                         objective_after=cost, seconds=0.)
            return dict(initial), trace
    if infeasible:
        raise ValueError('Requested balance is infeasible for indivisible or fixed vertex masses')
    if not positive or k == 1:
        result = dict(initial or fixed)
        loads = [sum(masses[n] for n,g in result.items() if g == i) for i in range(k)]
        for node in sorted(ids,key=lambda n:(-masses[n],index[n])):
            if node not in result:
                feasible = [i for i in range(k) if loads[i] + masses[node] <= caps[i]]
                if not feasible:
                    raise ValueError('Degenerate assignment did not meet requested balance capacity')
                group = min(feasible, key=lambda i:(loads[i], i))
                result[node] = group
                loads[group] += masses[node]
        if any(a > b for a,b in zip(loads,caps,strict=True)):
            raise ValueError('Degenerate assignment did not meet requested balance')
        if set(result.values()) != set(range(k)):
            raise ValueError('Degenerate assignment cannot meet occupied-part budget')
        trace.update(status='trivial_one_part' if k == 1 else 'degenerate_zero_objective',
                     accepted=True, objective_after=objective(edges,result), seconds=0.,
                     block_weights=loads, balance_feasible=True)
        return result, trace
    import mtkahypar as mt
    global _INITIALIZER
    if _INITIALIZER is None:
        _INITIALIZER = mt.initialize(4)
    initializer = _INITIALIZER
    mt.set_seed(seed)
    context = initializer.context_from_preset(mt.PresetType.DETERMINISTIC_QUALITY)
    context.set_partitioning_parameters(k, epsilon, mt.Objective.KM1)
    if block_caps is not None:
        context.set_individual_target_block_weights(caps)
    context.logging = False
    hg = initializer.create_hypergraph(context, len(ids), len(positive),
                                      [[index[n] for n in dict.fromkeys(e['members'])] for e,w in positive],
                                      [masses[n] for n in ids], [w for e,w in positive])
    if fixed:
        hg.add_fixed_vertices([fixed.get(n,-1) for n in ids], k)
    tick = time.perf_counter()
    if initial is None:
        partition = hg.partition(context)
    else:
        partition = hg.create_partitioned_hypergraph(context,k,[initial[n] for n in ids])
        partition.improve_partition(context,cycles)
    result = dict(zip(ids, map(int, partition.get_partition()), strict=True))
    trace.update(seconds=time.perf_counter()-tick, integer_km1=int(partition.km1()),
                 objective_after=objective(edges,result))
    loads = [sum(masses[n] for n,g in result.items() if g == i) for i in range(k)]
    invalid = (set(result.values()) != set(range(k)) or any(result[n] != g for n,g in fixed.items()) or
               any(a > b for a,b in zip(loads,caps,strict=True)))
    before = None if initial is None else objective(edges, initial)
    trace.update(objective_before=before, block_weights=loads, balance_feasible=not invalid)
    increases = before is not None and trace['objective_after'] > before + 1e-12
    if invalid or increases:
        reason = 'invalid_or_imbalanced_output' if invalid else 'floating_objective_increased'
        if initial is None:
            raise ValueError('Native solve rejected: ' + reason)
        trace.update(status='rejected_'+reason, accepted=False, objective_after=before,
                     proposed_block_weights=loads, retained_initial=True,
                     block_weights=initial_loads, balance_feasible=all(a<=b for a,b in zip(initial_loads,caps,strict=True)))
        return dict(initial), trace
    trace.update(status='native_solved', accepted=True)
    return result, trace
