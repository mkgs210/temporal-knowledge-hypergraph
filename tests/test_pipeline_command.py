"""The public command must measure real child processes and stop on failure."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def runner():
    path=Path(__file__).resolve().parents[1]/'scripts/run_pipeline.py'
    spec=importlib.util.spec_from_file_location('pipeline_command',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_records_complete_wall_time_and_each_real_step(tmp_path):
    module=runner()
    result=module.run_steps(tmp_path,sys.executable,tmp_path/'out',[
        ('first',['-c','print("first output",flush=True)']),
        ('second',['-c','print("second output",flush=True)'])])
    saved=json.loads((tmp_path/'out-timing.json').read_text())
    assert result==saved and saved['exit_code']==0
    assert [s['name'] for s in saved['steps']]==['first','second']
    assert all(s['exit_code']==0 and s['elapsed_seconds']>0 for s in saved['steps'])
    assert saved['elapsed_seconds']>=sum(s['elapsed_seconds'] for s in saved['steps'])
    assert 'first output' in (tmp_path/'out.log').read_text()
    assert 'second output' in (tmp_path/'out.log').read_text()


def test_command_keeps_failure_timing_and_does_not_run_later_steps(tmp_path):
    module=runner()
    result=module.run_steps(tmp_path,sys.executable,tmp_path/'out',[
        ('failure',['-c','import sys;print("expected failure");sys.exit(7)']),
        ('must_not_run',['-c','raise AssertionError("later step ran")'])])
    assert result['exit_code']==7 and len(result['steps'])==1
    assert result['steps'][0]['name']=='failure'
    assert json.loads((tmp_path/'out-timing.json').read_text())['exit_code']==7
    assert 'expected failure' in (tmp_path/'out.log').read_text()


def test_compressed_annotations_are_authenticated_and_extracted(tmp_path):
    import hashlib
    import tarfile
    from tkh.cpu_annotations import resolve_checkpoint
    labels=tmp_path/'source/labels';labels.mkdir(parents=True)
    (labels/'protocol.json').write_text('{"test":true}')
    archive=tmp_path/'labels.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(labels,arcname='labels')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name+'.sha256').write_text(sha+'  '+archive.name+'\n')
    restored=resolve_checkpoint(archive,cache_dir=tmp_path/'cache')
    assert json.loads((restored/'protocol.json').read_text())=={'test':True}
    assert resolve_checkpoint(archive,cache_dir=tmp_path/'cache')==restored
    archive.write_bytes(archive.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):
        resolve_checkpoint(archive,cache_dir=tmp_path/'cache')
