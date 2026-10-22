import numpy as np
import pytest
from tkh.divisive_solver import objective, quotient, boundary_quotient, integer_weights, solve_partition


def test_native_connectivity_and_exact_counted_quotient():
    edges = [{'members': ['a','b','c','d'], 'weight': 2.}, {'members':['a','d'], 'weight': .5}]
    fine = {'a':'x','b':'x','c':'y','d':'z'}
    q = quotient(edges, fine)
    assert q[0]['counts'] == {'x':2,'y':1,'z':1}
    coarse = {'x':0,'y':1,'z':2}
    assert objective(edges, {n:coarse[g] for n,g in fine.items()}) == objective(q, coarse) == 4.5


def test_boundary_terminals_preserve_three_part_deltas_and_leaving_own_part():
    edges = [{'members':['a','b','c','d'], 'weight':2.}, {'members':['a','b'],'weight':.7}]
    assignment = {'a':0,'b':0,'c':1,'d':2}
    q, fixed = boundary_quotient(edges, {'a'}, assignment)
    assert len(fixed) == 3
    assert sorted(q[0]['counts'].values()) == [1,1,1,1]
    start = {'a':0, **fixed}
    for target in (0,1,2,3):
        end = {**start, 'a':target}
        original = {**assignment, 'a':target}
        assert objective(q,end)-objective(q,start) == pytest.approx(objective(edges,original)-objective(edges,assignment))


@pytest.mark.parametrize('weight', [float('nan'), float('inf'), -1.])
def test_reject_invalid_numeric_weight(weight):
    with pytest.raises(ValueError):
        integer_weights([{'members':['a','b'],'weight':weight}])


def test_rounding_bound_and_zero_not_fabricated():
    edges = [{'members':['a','b','c'],'weight':.123456789}, {'members':['a','b'],'weight':0.}]
    ints, check = integer_weights(edges, scale=100)
    assert ints == [12,0]
    assert check['rounding_error_bound'] == pytest.approx(.003456789 * 2)
    assignment, trace = solve_partition(['a','b'], [], 2)
    assert len(set(assignment.values())) == 2
    assert trace['status'] == 'degenerate_zero_objective'


def test_native_solver_fixed_part_identities_and_float_nonincrease():
    edges = [{'members':['a','b'],'weight':1.}, {'members':['a','c','d'],'weight':.1}]
    start = {'a':0,'b':0,'c':1,'d':2}
    result, trace = solve_partition(list(start), edges, 3, initial=start, fixed={'b':0,'c':1,'d':2})
    assert result['b'] == 0 and result['c'] == 1 and result['d'] == 2
    assert objective(edges,result) <= objective(edges,start) + 1e-12
    assert trace['solver'] == 'mtkahypar'


def test_solver_rejects_infeasible_budget_and_integer_overflow():
    with pytest.raises(ValueError):
        solve_partition(['a'], [], 2)
    with pytest.raises(ValueError):
        integer_weights([{'members':['a','b'],'weight':1e20}])


def test_boundary_keeps_fixed_part_with_isolated_vertices():
    q,fixed=boundary_quotient([{'members':['a','b'],'weight':1.}], {'a'}, {'a':0,'b':1,'isolated':2})
    assert set(fixed.values()) == {1,2}


def test_zero_objective_rejects_mass_over_balance_capacity():
    with pytest.raises(ValueError,match='balance'):
        solve_partition(['a','b','c'],[],2,masses={'a':100,'b':1,'c':1})


def test_imbalanced_start_retained_only_as_explicit_rejected_refinement():
    start={'a':0,'b':1,'c':1}
    result,trace=solve_partition(list(start),[],2,initial=start,masses={'a':100,'b':1,'c':1})
    assert result==start
    assert trace['accepted'] is False
    assert trace['status']=='rejected_initial_imbalance'
    assert trace['balance_feasible'] is False
    assert trace['block_weights']==[100,2]
    assert trace['maximum_block_weight']==76
    assert trace['retained_initial']


@pytest.mark.parametrize('epsilon',[float('nan'),float('inf'),-1.])
def test_invalid_balance_parameter_rejected(epsilon):
    with pytest.raises(ValueError,match='epsilon'):
        solve_partition(['a','b'],[],2,epsilon=epsilon)


def test_zero_objective_honors_fixed_balance_capacity():
    with pytest.raises(ValueError,match='balance'):
        solve_partition(['a','b','c','d'],[],2,fixed={'a':0,'b':0,'c':0,'d':0},epsilon=0.)


def test_feasible_imbalanced_start_is_repaired_by_native_refinement():
    ids=[f'n{i}' for i in range(8)]
    edges=[{'members':ids[i:i+2],'weight':1.} for i in range(0,8,2)]
    start={n:int(i==7) for i,n in enumerate(ids)}
    result,trace=solve_partition(ids,edges,2,initial=start)
    assert trace['accepted'] is True
    assert max(sum(g==part for g in result.values()) for part in range(2))<=6
    assert objective(edges,result)<objective(edges,start)


def test_explicit_unequal_caps_keep_an_existing_topic_feasible_without_enlarging_group_budget():
    start = {'a': 0, 'b': 0, 'c': 0, 'd': 1}
    result, trace = solve_partition(list(start), [{'members': ['a', 'b', 'c'], 'weight': 1.}],
        2, initial=start, fixed={'a': 0, 'd': 1}, block_caps=[3, 2])
    assert trace['accepted']
    assert result['a'] == 0 and result['d'] == 1
    assert len(set(result.values())) == 2
    assert trace['block_caps'] == [3, 2]
    assert trace['balance_feasible']


@pytest.mark.parametrize('caps', [[0, 2], [3], [1, 1], [3, float('inf')], [True, 4]])
def test_explicit_caps_are_validated(caps):
    with pytest.raises(ValueError, match='cap'):
        solve_partition(['a', 'b', 'c'], [], 2, block_caps=caps)
