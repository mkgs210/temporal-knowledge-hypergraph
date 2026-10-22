"""Final label packet is authenticated and cannot accept a partial pilot."""
import json

import pytest

from test_gemini_labels import fixture,reply


def test_judge_selection_preserves_archived_provider_even_without_key(tmp_path,monkeypatch):
    from tkh.cpu_label_audit import resolve_judge_model
    labels=tmp_path/'labels';p=labels/'audit/api_check/protocol.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'judge_model':'nvidia/llama-3.1-nemotron-70b-instruct'}))
    monkeypatch.setattr('tkh.label_judging.load_api_key',lambda *a,**kw:pytest.fail('Archived model selection read a key'))
    assert resolve_judge_model(labels)=='nvidia/llama-3.1-nemotron-70b-instruct'


def test_gemini_is_rejected_before_any_key_or_network_access(tmp_path,monkeypatch):
    from tkh.cpu_label_audit import resolve_judge_model
    from tkh.label_judging import run_judging
    from test_label_judging import packet
    monkeypatch.setattr('tkh.label_judging.load_api_key',lambda *a,**kw:pytest.fail('Key access is forbidden'))
    with pytest.raises(ValueError,match='Gemini'):
        resolve_judge_model(tmp_path/'labels','google/gemini-2.5-flash')
    with pytest.raises(ValueError,match='Gemini'):
        run_judging(packet(tmp_path),tmp_path/'judge',model='google/gemini-2.5-flash')
    assert not (tmp_path/'judge/protocol.json').exists()


def test_complete_labels_create_a_blinded_packet_with_withheld_claim_text(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    from tkh.cpu_label_audit import prepare_audit
    run,prompt,_=fixture(tmp_path);labels=tmp_path/'labels'
    monkeypatch.setenv('GEMINI_API_KEY','fake')
    run_labels(run,labels,prompt_path=prompt,evidence_policy='full_scope',send=lambda *a:reply())
    result=prepare_audit(labels)
    packet=json.loads((labels/'audit/diagnostic_packet.json').read_text())
    assert result['sample_groups']==2 and len(packet)==2
    assert all('group_id' not in t and 'status' not in t for t in packet)
    assert any(t['held_out_claims'] for t in packet)
    # Cached evidence preparation is idempotent, not a second model call.
    assert prepare_audit(labels)==result
    p=next((labels/'completed').glob('*.json'))
    row=json.loads(p.read_text());row['gloss']='Invented result.';p.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='text|response'):
        prepare_audit(labels)


def test_partial_labels_are_rejected_before_any_final_audit_is_written(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    from tkh.cpu_label_audit import prepare_audit
    run,prompt,_=fixture(tmp_path);labels=tmp_path/'labels'
    monkeypatch.setenv('GEMINI_API_KEY','fake')
    run_labels(run,labels,prompt_path=prompt,evidence_policy='full_scope',limit=1,send=lambda *a:reply())
    with pytest.raises(ValueError,match='complete|missing'):
        prepare_audit(labels)
    assert not (labels/'audit/diagnostic_packet.json').exists()


def test_selected_pilot_cannot_be_presented_as_full_t5_coverage(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    from tkh.cpu_label_audit import prepare_audit
    run,prompt,_=fixture(tmp_path);labels=tmp_path/'labels';selection=tmp_path/'selection.json'
    selection.write_text(json.dumps([{'year':2022,'level':1,'group_id':'fine'}]))
    monkeypatch.setenv('GEMINI_API_KEY','fake')
    run_labels(run,labels,prompt_path=prompt,evidence_policy='full_scope',groups_file=selection,send=lambda *a:reply())
    with pytest.raises(ValueError,match='coverage|pilot|incomplete'):
        prepare_audit(labels)
