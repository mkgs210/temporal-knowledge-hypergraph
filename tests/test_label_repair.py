"""Paired review uses identical evidence, hides variants and retains old results."""
import json

import pytest

from test_gemini_labels import fixture, reply


def runs(tmp_path, monkeypatch):
    from tkh.gemini_labels import run_labels
    source, prompt, _ = fixture(tmp_path)
    monkeypatch.setenv('GEMINI_API_KEY', 'fake-key')
    old, new = tmp_path/'old', tmp_path/'new'
    for path, policy in [(old, 'source_split'), (new, 'full_scope')]:
        run_labels(source, path, prompt_path=prompt, evidence_policy=policy,
                   rpm=100000, send=lambda *args: reply())
    return old, new


def test_paired_packet_uses_identical_evidence_and_records_name_visibility(tmp_path, monkeypatch):
    from tkh.label_repair import prepare_paired_review
    old, new = runs(tmp_path, monkeypatch)
    old_bytes = {p.relative_to(old): p.read_bytes() for p in old.rglob('*.json')}
    design = prepare_paired_review(old, new)
    packet = json.loads((new/'audit/diagnostic_packet.json').read_text())
    assert len(packet) == 4
    assert design['sample_groups'] == 2
    assert design['scope_names_hidden_from_repaired_generator'] is False
    assert design['is_independent_final_test'] is False
    for i in range(0, 4, 2):
        a, b = packet[i:i+2]
        assert a['pair_id'] == b['pair_id']
        assert [a['variant'], b['variant']] == ['original', 'repaired']
        assert a['held_out_members'] == b['held_out_members']
        assert a['held_out_claims'] == b['held_out_claims']
        names = {n['id'] for n in a['held_out_members'] if n['type'] != 'claim'}
        assert names == {'n0', 'n2', 'n4', 'n6'}
    assert old_bytes == {p.relative_to(old): p.read_bytes() for p in old.rglob('*.json')}
    assert prepare_paired_review(old, new) == design


def test_review_refuses_forged_generated_text(tmp_path, monkeypatch):
    from tkh.label_repair import prepare_paired_review
    old, new = runs(tmp_path, monkeypatch)
    row_file = next((new/'completed').glob('*.json'))
    row = json.loads(row_file.read_text()); row['label'] = 'Forged replacement'
    row_file.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='text|response|record'):
        prepare_paired_review(old, new)


def test_review_refuses_comparing_changed_hierarchies_as_a_prompt_repair(tmp_path, monkeypatch):
    from tkh.label_repair import prepare_paired_review
    old, new = runs(tmp_path, monkeypatch)
    path = new/'prompts.json'
    jobs = json.loads(path.read_text()); jobs[0]['hierarchy_sha256'] = 'different-membership'
    path.write_text(json.dumps(jobs))
    with pytest.raises(ValueError, match='hierarchy membership changed'):
        prepare_paired_review(old, new)


def test_judge_prompt_hides_paired_variant_and_scores_variants_separately(tmp_path, monkeypatch):
    from tkh.label_repair import prepare_paired_review
    from tkh.label_judging import run_judging
    from test_nvidia_labels import reply as api_reply
    old, new = runs(tmp_path, monkeypatch)
    # Give the test runs distinct model identities, as in the real protocol.
    prepare_paired_review(old, new)
    monkeypatch.setenv('NVIDIA_API_KEY', 'fake-key')
    def send(model, prompt, key, **kwargs):
        data = json.loads(prompt.split('EVIDENCE:\n')[1])
        assert 'variant' not in data and 'pair_id' not in data
        assert 'generator_status' not in data
        return api_reply(json.dumps({'rating':'accurate', 'coverage':'representative',
                                    'reason':'The evidence supports the topic.', 'evidence_ids':[]}))
    result = run_judging(new, tmp_path/'judge', send=send, sleep=lambda _: None, rpm=100000)
    assert result['sample_size'] == 4
    assert result['variants']['original']['ratings'] == {'accurate': 2}
    assert result['variants']['repaired']['ratings'] == {'accurate': 2}
    assert result['control_gate_passed'] is False  # Constant-answer judge must fail.
    assert result['review_design']['is_independent_final_test'] is False
