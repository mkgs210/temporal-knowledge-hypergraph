"""Exercise persistent caching; the neural model boundary is replaced offline."""
import sqlite3

import numpy as np
import pytest


class RecordingModel:
    def __init__(self, calls, **kwargs):
        calls.append(('load', kwargs))
        self.calls = calls
    def encode(self, texts, **kwargs):
        self.calls.append(('encode', list(texts), kwargs))
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0] = 1
            vectors[i, 1] = sum(text.encode()) % 19 + 1
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def factory(calls):
    return lambda **kwargs: RecordingModel(calls, **kwargs)


def test_cold_cpu_encode_then_restart_encodes_only_new_texts(tmp_path):
    from tkh.cpu_embeddings import CPUEncoder
    calls = []
    encoder = CPUEncoder(tmp_path, model_factory=factory(calls))
    first = encoder.encode(['Alpha', 'Beta', 'Alpha'])
    assert first.shape == (3, 384)
    np.testing.assert_array_equal(first[0], first[2])
    assert calls[0][1]['device'] == 'cpu'
    assert calls[0][1]['model_name_or_path'] == 'sentence-transformers/all-MiniLM-L6-v2'
    assert len(calls[0][1]['revision']) == 40
    assert calls[1][1] == ['Alpha', 'Beta']
    assert calls[1][2]['normalize_embeddings'] is True
    calls.clear()
    restarted = CPUEncoder(tmp_path, model_factory=factory(calls))
    cached = restarted.encode(['Beta', 'Alpha'])
    assert calls == []  # Even model initialization is unnecessary for cache hits.
    np.testing.assert_array_equal(cached, first[[1, 0]])
    new = restarted.encode(['Gamma', 'Alpha'])
    assert calls[1][1] == ['Gamma']
    np.testing.assert_array_equal(new[1], first[0])


def test_changed_text_does_not_reuse_another_texts_vector(tmp_path):
    from tkh.cpu_embeddings import CPUEncoder
    calls = []
    encoder = CPUEncoder(tmp_path, model_factory=factory(calls))
    old = encoder.encode(['Original text'])
    encoder.encode(['Edited text'])
    assert [row[1] for row in calls if row[0] == 'encode'] == [['Original text'], ['Edited text']]
    np.testing.assert_array_equal(encoder.encode(['Original text']), old)


def test_cache_rejects_corrupt_vector_without_hiding_it_by_reencoding(tmp_path):
    from tkh.cpu_embeddings import CPUEncoder
    calls = []
    encoder = CPUEncoder(tmp_path, model_factory=factory(calls))
    encoder.encode(['Alpha'])
    with sqlite3.connect(encoder.path) as db:
        db.execute('UPDATE embeddings SET vector=?', (np.zeros(384, dtype=np.float32).tobytes(),))
    calls.clear()
    with pytest.raises(ValueError, match='cache|unit|vector'):
        CPUEncoder(tmp_path, model_factory=factory(calls)).encode(['Alpha'])
    assert calls == []


def test_invalid_model_output_is_not_committed_to_cache(tmp_path):
    from tkh.cpu_embeddings import CPUEncoder
    class Invalid(RecordingModel):
        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 384), dtype=np.float32)
    with pytest.raises(ValueError, match='unit|vector'):
        CPUEncoder(tmp_path, model_factory=lambda **kw: Invalid([], **kw)).encode(['Alpha'])
    calls = []
    CPUEncoder(tmp_path, model_factory=factory(calls)).encode(['Alpha'])
    assert calls[1][1] == ['Alpha']


def test_completed_batch_survives_interruption_and_is_not_reencoded(tmp_path):
    from tkh.cpu_embeddings import CPUEncoder
    class Interrupted(RecordingModel):
        def encode(self, texts, **kwargs):
            if texts == ['Beta']:
                raise RuntimeError('interrupted')
            return super().encode(texts, **kwargs)
    with pytest.raises(RuntimeError, match='interrupted'):
        CPUEncoder(tmp_path, batch_size=1,
                   model_factory=lambda **kw: Interrupted([], **kw)).encode(['Alpha', 'Beta'])
    calls = []
    CPUEncoder(tmp_path, model_factory=factory(calls)).encode(['Alpha', 'Beta'])
    assert calls[1][1] == ['Beta']
