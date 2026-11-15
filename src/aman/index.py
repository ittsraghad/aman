"""KnowledgeIndex: chunks + BM25 index + dense vectors, with an on-disk vector cache."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from aman.bm25 import BM25
from aman.text import expand_query, tokenize


BATCH = 100   # chunks embedded per step (progress is saved after each)


class KnowledgeIndex:
    def __init__(self, chunks_path: Path, embedder, cache_dir: Path | None = None, build_vectors: bool = True):
        self.embedder = embedder
        self.chunks = [json.loads(l) for l in Path(chunks_path).read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self.chunks:
            raise ValueError("chunks file is empty")
        # Index title twice so exact IDs/names weigh more than body text.
        self.bm25 = BM25([tokenize(c["title"] + " " + c["title"] + " " + c["text"]) for c in self.chunks])
        self.vectors = self._load_or_build_vectors(cache_dir, build_vectors)
        self.by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self.by_doc.setdefault(c["doc_id"], []).append(i)
        self.doc_freq = self.bm25.idf  # idf table reused for relevance scoring

    def _load_or_build_vectors(self, cache_dir, build_vectors):
        texts = [c["text"] for c in self.chunks]
        if cache_dir is None:
            return self.embedder.encode_passages(texts) if build_vectors else None
        h = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:12]      # changes whenever the corpus changes
        f = Path(cache_dir) / f"vectors_{self.embedder.name}_{h}.npy"
        if f.exists():
            return np.load(f)
        if not build_vectors:
            print(f"[aman] vector cache {f.name} not found -> keyword-only search (run python -m aman.build_index)")
            return None
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        part = f.with_suffix(".partial.npy")                # resumable: progress is saved after every batch
        parts = [np.load(part)] if part.exists() else []
        done = sum(len(x) for x in parts)
        if done:
            print(f"[aman] resuming embeddings from {done}/{len(texts)}", flush=True)
        for i in range(done, len(texts), BATCH):
            parts.append(self.embedder.encode_passages(texts[i:i + BATCH]))
            np.save(part, np.vstack(parts))
            print(f"[aman] embedded {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
        vecs = np.vstack(parts) if parts else np.zeros((0, 1), np.float32)
        np.save(f, vecs)
        part.unlink(missing_ok=True)
        return vecs

    def keyword_search(self, query: str, k: int = 30):
        return self.bm25.score(expand_query(query), k)

    def vector_search(self, query: str, k: int = 30):
        if self.vectors is None:
            return []
        try:                                        # remote embedder may fail: degrade to keyword search
            q = self.embedder.encode_query(expand_query(query))
            sims = self.vectors @ q
        except Exception as e:
            print(f"[aman] vector search unavailable ({type(e).__name__}); using keyword search only")
            return []
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top if sims[i] > 0]

    def get_doc(self, doc_id: str) -> list[dict]:
        return [self.chunks[i] for i in self.by_doc.get(doc_id.upper(), [])]
