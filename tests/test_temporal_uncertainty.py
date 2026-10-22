"""Block dependence, pairing and abstention matter more than narrow intervals."""
from copy import deepcopy

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score


def test_point_ari_aligns_ids_excludes_new_nodes_and_counts_persistent_id_changes():
    from tkh.temporal_uncertainty import source_block_ari
    before = dict(zip("abcd", ["A", "A", "B", "B"]))
    after = {"d": "D", "c": "C", "b": "D", "a": "C", "new": "C"}
    report = source_block_ari(before, after, dict.fromkeys("abcd", [1]), n_resamples=10)
    assert report["point_ari"] == -.5
    assert report["shared_old_nodes"] == 4
    assert report["new_nodes"] == 1
    assert report["persistent_id_movement"] == 4
    assert report["movement_fraction"] == 1
    assert report["interval"] is None


def test_transitive_source_overlap_is_one_block_and_cannot_invent_a_ci():
    from tkh.temporal_uncertainty import source_block_ari
    before = dict(zip("abcd", [0, 0, 1, 1]))
    after = dict(zip("abcd", [0, 1, 0, 1]))
    sources = {"a": [1], "b": [1, 2], "c": [2, 3], "d": [3]}
    report = source_block_ari(before, after, sources, n_resamples=100)
    assert report["blocks"]["count"] == 1
    assert report["blocks"]["sources"] == [[1, 2, 3]]
    assert report["blocks"]["overlapping_nodes"] == 2
    assert report["blocks"]["largest_node_fraction"] == 1
    assert report["interval"] is None
    assert report["uncertainty_status"] == "insufficient_source_components"
    assert report["bootstrap_ari"] == []


def test_bootstrap_repeats_whole_blocks_with_the_same_paired_labels():
    from tkh.temporal_uncertainty import source_block_ari
    before = {f"n{i}": i // 4 for i in range(12)}
    after = {f"n{i}": (i // 4 if i != 3 else 1) for i in range(12)}
    sources = {f"n{i}": [i // 4] for i in range(12)}
    report = source_block_ari(before, after, sources, seed=9,
                              n_resamples=25, min_blocks=2)
    # Independent expansion oracle: no node sampling, preserve every node per paper.
    rng = np.random.default_rng(9)
    expected = []
    for _ in range(25):
        chosen = rng.integers(0, 3, size=3)
        nodes = [f"n{i}" for paper in chosen for i in range(4 * paper, 4 * paper + 4)]
        expected.append(adjusted_rand_score([before[n] for n in nodes],
                                            [after[n] for n in nodes]))
    assert report["bootstrap_ari"] == pytest.approx(expected)
    assert report["interval"]["low"] == pytest.approx(np.quantile(expected, .025))
    assert report["interval"]["high"] == pytest.approx(np.quantile(expected, .975))
    assert report["interval"]["coverage_verified"] is False
    assert report == source_block_ari(dict(reversed(list(before.items()))), after,
                                       sources, seed=9, n_resamples=25, min_blocks=2)


def test_permuted_cluster_names_keep_ari_but_change_identity_movement():
    from tkh.temporal_uncertainty import source_block_ari
    before = dict(zip("abcd", ["A", "A", "B", "B"]))
    after = dict(zip("abcd", ["B", "B", "A", "A"]))
    report = source_block_ari(before, after, dict.fromkeys("abcd", [1]))
    assert report["point_ari"] == 1
    assert report["persistent_id_movement"] == 4


def test_few_or_dominating_components_are_reported_without_an_interval():
    from tkh.temporal_uncertainty import source_block_ari
    before = {str(i): i % 2 for i in range(100)}
    sources = {str(i): [0 if i < 99 else 1] for i in range(100)}
    report = source_block_ari(before, before, sources)
    assert report["blocks"]["effective_count_by_node_mass"] < 1.03
    assert report["interval"] is None
    assert report["source_delete_one"]["interpretation"] == "fixed-partition sensitivity, not a confidence interval"


def test_delete_source_sensitivity_removes_shared_nodes_without_changing_exact_point():
    from tkh.temporal_uncertainty import source_block_ari
    before = dict(zip("abcd", [0, 0, 1, 1]))
    after = dict(zip("abcd", [0, 1, 0, 1]))
    report = source_block_ari(before, after, {"a": [1], "b": [1], "c": [2], "d": [2]})
    assert report["point_ari"] == -.5
    assert report["source_delete_one"]["rows"] == [
        {"source": 1, "removed_nodes": 2, "retained_nodes": 2, "ari": 0.},
        {"source": 2, "removed_nodes": 2, "retained_nodes": 2, "ari": 0.},
    ]


@pytest.mark.parametrize("kwargs", [
    {"node_sources": {"a": [], "b": [1]}},
    {"node_sources": {"a": [1]}},
    {"node_sources": {"a": "paper", "b": [1]}},
    {"node_sources": {"a": [None], "b": [1]}},
    {"n_resamples": 0}, {"confidence": 1}, {"min_blocks": 1},
])
def test_invalid_input_is_not_silently_dropped(kwargs):
    from tkh.temporal_uncertainty import source_block_ari
    args = {"node_sources": {"a": [1], "b": [2]}}
    args.update(kwargs)
    with pytest.raises(ValueError):
        source_block_ari({"a": 0, "b": 1}, {"a": 0, "b": 1}, **args)


def test_too_few_shared_nodes_is_na_instead_of_sklearn_degenerate_perfect_score():
    from tkh.temporal_uncertainty import source_block_ari
    report = source_block_ari({"a": 0}, {"a": 1, "b": 0}, {"a": [1]})
    assert report["point_ari"] is None
    assert report["uncertainty_status"] == "too_few_shared_nodes"


def snapshot_fixture():
    nodes = [{"id": n, "type": "method", "provenance": {"articles": [i // 2 + 1]}}
             for i, n in enumerate("abcd")]
    hierarchy = {"cutoff": 2022, "levels": [
        {"index": 0, "groups": [{"id": "C", "members": list("abcd")}]},
        {"index": 1, "groups": [{"id": "F1", "members": list("ab")},
                                  {"id": "F2", "members": list("cd")}]},
    ]}
    return {"cutoff": 2022, "nodes": nodes, "hyperedges": []}, hierarchy


def test_snapshot_wrapper_uses_old_provenance_and_reports_boundary_dependence():
    from tkh.temporal_uncertainty import snapshot_temporal_uncertainty
    old, h = snapshot_fixture()
    new, current = deepcopy(old), deepcopy(h)
    new["cutoff"] = current["cutoff"] = 2023
    for n in new["nodes"]:
        n["provenance"]["articles"].append(3)
    new["hyperedges"] = [{"id": "link", "members": ["a", "c"]}]
    report = snapshot_temporal_uncertainty(old, h, new, current, n_resamples=10)
    assert report["levels"][0]["blocks"]["count"] == 2
    assert report["levels"][0]["blocks"]["sources"] == [[1], [2]]
    assert report["levels"][0]["point_ari"] == 1
    assert report["source_policy"] == "previous-cutoff provenance only"
    assert report["dependence"]["current_shared_old_hyperedges_crossing_source_components"] == 1
    assert report["satisfies_unconditional_ci_requirement"] is False


@pytest.mark.parametrize("change", ["overlap", "missing", "future_member", "wrong_cutoff", "duplicate_node"])
def test_snapshot_wrapper_rejects_malformed_membership_instead_of_grading_it(change):
    from tkh.temporal_uncertainty import snapshot_temporal_uncertainty
    old, h = snapshot_fixture()
    new, current = deepcopy(old), deepcopy(h)
    new["cutoff"] = current["cutoff"] = 2023
    if change == "overlap":
        current["levels"][1]["groups"][0]["members"].append("c")
    elif change == "missing":
        current["levels"][1]["groups"][0]["members"].remove("b")
    elif change == "future_member":
        current["levels"][0]["groups"][0]["members"].append("future")
    elif change == "wrong_cutoff":
        current["cutoff"] = 2024
    else:
        new["nodes"].append(deepcopy(new["nodes"][0]))
    with pytest.raises(ValueError):
        snapshot_temporal_uncertainty(old, h, new, current)


def test_type_stratified_random_correspondence_preserves_type_by_cluster_counts():
    from tkh.temporal_uncertainty import alignment_significance
    before = {str(i): i // 4 for i in range(12)}
    after = dict(before)
    types = {str(i): "dataset" if i % 2 else "method" for i in range(12)}
    report = alignment_significance(before, after, types, n_permutations=25, seed=13)
    ids = sorted(before)
    a, b = np.array([before[n] for n in ids]), np.array([after[n] for n in ids])
    rng = np.random.default_rng(13)
    expected = []
    for _ in range(25):
        permuted = b.copy()
        for kind in sorted(set(types.values())):
            idx = np.array([i for i, n in enumerate(ids) if types[n] == kind])
            permuted[idx] = rng.permutation(b[idx])
        expected.append(adjusted_rand_score(a, permuted))
    assert report["null_ari"] == pytest.approx(expected)
    assert report["p_greater_equal"] == pytest.approx(1 / 26)
    assert report["interpretation"] == "fixed-partition random-correspondence test, not temporal-process uncertainty"


def test_degenerate_and_bad_correspondence_controls_do_not_become_significant():
    from tkh.temporal_uncertainty import alignment_significance
    before = dict(zip("abcd", [0, 0, 1, 1]))
    crossed = dict(zip("abcd", [0, 1, 0, 1]))
    report = alignment_significance(before, crossed, n_permutations=99)
    assert report["point_ari"] == -.5
    assert report["p_greater_equal"] == 1
    report = alignment_significance(dict.fromkeys("abcd", 0),
                                      dict.fromkeys("abcd", 4), n_permutations=99)
    assert report["p_greater_equal"] == 1


def test_fixed_seed_independent_correspondence_negative_control_is_not_systematically_positive():
    from tkh.temporal_uncertainty import alignment_significance
    rng = np.random.default_rng(41)
    before = {str(i): i // 10 for i in range(60)}
    values = list(before.values())
    significant = 0
    for trial in range(20):
        after = dict(zip(before, rng.permutation(values).tolist()))
        report = alignment_significance(before, after, n_permutations=99, seed=trial)
        significant += report["p_greater_equal"] <= .05
    # A deterministic smoke control, not a coverage/calibration theorem.
    assert significant <= 3
