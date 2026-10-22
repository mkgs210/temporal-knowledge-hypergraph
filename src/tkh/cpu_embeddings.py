"""Small CPU sentence embeddings, computed lazily and cached by exact text."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from tqdm import tqdm

MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
DIMENSION = 384


class CPUEncoder:
    def __init__(self, cache_dir, *, model_factory=None, batch_size=32):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Embedding batch size must be positive')
        self.config = {'model_id': MODEL_ID, 'revision': MODEL_REVISION,
            'device': 'cpu', 'dimension': DIMENSION, 'max_length': 256,
            'normalize': True, 'output_dtype': 'float32', 'prefix': '',
            'backend': 'sentence-transformers==5.2.0',
            'long_text_policy': 'Sentence Transformers truncation after 256 word-piece tokens'}
        self.batch_size = batch_size
        self.model_factory = model_factory
        self.model = None
        self.encoded_texts = 0
        self.cache_hits = 0
        key = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:20]
        self.path = Path(cache_dir)/f'{key}.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate(vectors, count):
        values = np.asarray(vectors, dtype=np.float32)
        if (values.shape != (count, DIMENSION) or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=1e-5)):
            raise ValueError('Embedding/cache vectors must be aligned finite unit 384-dimensional rows')
        return values

    def _load_model(self):
        if self.model is None:
            loader = self.model_factory
            if loader is None:
                try:
                    import torch
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    raise RuntimeError('Install CPU dependencies: python -m pip install -r requirements-cpu.lock') from None
                torch.set_num_threads(4)
                loader = SentenceTransformer
            self.model = loader(model_name_or_path=MODEL_ID, revision=MODEL_REVISION,
                                device='cpu', trust_remote_code=False)
            self.model.max_seq_length = 256
        return self.model

    def encode(self, texts):
        texts = list(texts)
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Embedding input must contain nonempty text')
        if not texts:
            return np.empty((0, DIMENSION), dtype=np.float32)
        requested = list(dict.fromkeys(texts))
        with closing(sqlite3.connect(self.path)) as db:
            db.execute('CREATE TABLE IF NOT EXISTS metadata (config TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS embeddings (text TEXT PRIMARY KEY, vector BLOB NOT NULL)')
            stored = db.execute('SELECT config FROM metadata').fetchall()
            if not stored:
                db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(self.config, sort_keys=True),))
                db.commit()
            elif len(stored) != 1 or json.loads(stored[0][0]) != self.config:
                raise ValueError('Embedding cache configuration differs')
            vectors = {}
            for text in requested:
                row = db.execute('SELECT vector FROM embeddings WHERE text=?', (text,)).fetchone()
                if row is not None:
                    try:
                        vector = np.frombuffer(row[0], dtype=np.float32)
                        self._validate(vector[None, :], 1)
                    except (ValueError, TypeError):
                        raise ValueError('Corrupt embedding cache vector; original row retained') from None
                    vectors[text] = vector.copy()
            self.cache_hits += len(vectors)
            missing = [t for t in requested if t not in vectors]
            if missing:
                model = self._load_model()
                with tqdm(total=len(missing), desc='CPU embeddings', unit='text', smoothing=0) as progress:
                    for start in range(0, len(missing), self.batch_size):
                        batch = missing[start:start+self.batch_size]
                        values = self._validate(model.encode(batch, batch_size=self.batch_size,
                            device='cpu', normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False, precision='float32'), len(batch))
                        with db:
                            for text, vector in zip(batch, values, strict=True):
                                db.execute('INSERT INTO embeddings VALUES (?, ?)', (text, vector.tobytes()))
                                vectors[text] = vector.copy()
                        self.encoded_texts += len(batch)
                        progress.update(len(batch))
            return np.stack([vectors[t] for t in texts])
