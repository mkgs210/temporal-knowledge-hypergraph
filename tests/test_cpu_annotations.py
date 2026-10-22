"""Cold-checkout T5 replay authenticates exact evidence, never guesses labels."""
import json

import pytest

from test_gemini_labels import fixture
from test_nvidia_labels import reply


def saved_labels(tmp_path,monkeypatch):
    from tkh.gemini_labels import run_labels
    run,prompt,_=fixture(tmp_path);source=tmp_path/'saved'
    monkeypatch.setenv('NVIDIA_API_KEY','fake')
    run_labels(run,source,provider='nvidia',model='nvidia/test',evidence_policy='full_scope',
               prompt_path=prompt,send=lambda *a:reply())
    monkeypatch.delenv('NVIDIA_API_KEY')
    return run,prompt,source


def test_replay_from_saved_responses_needs_no_api_key(tmp_path,monkeypatch):
    from tkh.cpu_annotations import replay_labels
    run,prompt,source=saved_labels(tmp_path,monkeypatch)
    def forbidden(*a,**kw):raise AssertionError('Annotation replay used credentials or a model')
    monkeypatch.setattr('tkh.gemini_labels.load_api_key',forbidden)
    out=tmp_path/'replayed'
    result=replay_labels(source,run,out,prompt_path=prompt)
    assert result['completed']==2 and (out/'replayed_from.json').is_file()
    assert (out/'snapshots/2022/hierarchy.json').is_file()


def test_replay_rejects_a_changed_group_input_instead_of_reusing_its_name(tmp_path,monkeypatch):
    from tkh.cpu_annotations import replay_labels
    run,prompt,source=saved_labels(tmp_path,monkeypatch)
    path=run/'snapshots/2022/input.json';graph=json.loads(path.read_text())
    graph['nodes'][0]['surface_form']='A different scientific record'
    path.write_text(json.dumps(graph))
    with pytest.raises(ValueError,match='evidence|input|hierarchy'):
        replay_labels(source,run,tmp_path/'bad',prompt_path=prompt)
