"""Turning text into vectors.

HashingEmbedder   : zero-dependency baseline (character n-gram hashing). Fast, offline,
                    good for exact-ish matches. NOT truly semantic and NOT cross-lingual.
SentenceEmbedder  : real multilingual semantic embeddings (pip install sentence-transformers).
make_embedder()   : picks SentenceEmbedder when installed, else Hashing. Override with
                    AMAN_EMBEDDER=hash|st and AMAN_EMBED_MODEL=<model name>.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

from aman.text import normalize_arabic, tokenize


class HashingEmbedder:
    name = "hash-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _idx(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(feature.encode()).digest()[:8], "little")
        return h % self.dim, 1.0 if (h >> 40) & 1 else -1.0

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            i, s = self._idx("w:" + tok)
            v[i] += 2.0 * s
            padded = f"#{tok}#"
            for n in (3, 4):
                for k in range(len(padded) - n + 1):
                    i, s = self._idx(padded[k:k + n])
                    v[i] += s
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


class SentenceEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)
        self.name = "st-" + model_name.split("/")[-1]
        self._e5 = "e5" in model_name.lower()

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        pre = "passage: " if self._e5 else ""
        return self.model.encode([pre + t for t in texts], normalize_embeddings=True,
                                 batch_size=32, show_progress_bar=len(texts) > 200).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        pre = "query: " if self._e5 else ""
        return self.model.encode([pre + text], normalize_embeddings=True)[0].astype(np.float32)


class GeminiEmbedder:
    """Remote embeddings (no local model, so it fits a 512 MB host). Multilingual and cross-lingual.
    Vectors for the corpus are built OFFLINE (python -m aman.build_index) and shipped with the app;
    only the user's question is embedded at runtime."""
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, model: str | None = None, dim: int | None = None):
        self.api_key = api_key
        self.model = model or os.getenv("AMAN_GEMINI_EMBED_MODEL", "gemini-embedding-001")
        self.dim = dim or int(os.getenv("AMAN_EMBED_DIM", "768"))
        self.name = f"gemini-{self.model}-{self.dim}"

    def _norm(self, v) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        return a / n if n else a

    def _req(self, text: str, task: str) -> dict:
        return {"model": f"models/{self.model}", "content": {"parts": [{"text": text}]},
                "taskType": task, "outputDimensionality": self.dim}

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        from aman.http import post_json
        out = []
        step = int(os.getenv("AMAN_EMBED_BATCH", "25"))
        for i in range(0, len(texts), step):
            if i:
                time.sleep(float(os.getenv("AMAN_EMBED_PAUSE", "0.3")))
            batch = texts[i:i + step]
            data = post_json(f"{self.BASE}/{self.model}:batchEmbedContents",
                             {"requests": [self._req(t, "RETRIEVAL_DOCUMENT") for t in batch]},
                             {"x-goog-api-key": self.api_key}, timeout=120, retries=8)
            out += [self._norm(e["values"]) for e in data["embeddings"]]
        return np.vstack(out) if out else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        from aman.http import post_json
        data = post_json(f"{self.BASE}/{self.model}:embedContent", self._req(text, "RETRIEVAL_QUERY"),
                         {"x-goog-api-key": self.api_key}, timeout=20, retries=1)
        return self._norm(data["embedding"]["values"])


def make_embedder():
    choice = os.getenv("AMAN_EMBEDDER", "auto")
    gkey = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if choice == "gemini" or (choice == "auto" and gkey and os.getenv("AMAN_NO_BUILD") == "1"):
        if not gkey:
            raise RuntimeError("AMAN_EMBEDDER=gemini needs GEMINI_API_KEY")
        return GeminiEmbedder(gkey)
    if choice in ("auto", "st"):
        try:
            return SentenceEmbedder(os.getenv("AMAN_EMBED_MODEL", "intfloat/multilingual-e5-small"))
        except Exception as e:  # not installed / no model cached
            if choice == "st":
                raise
            print(f"[aman] sentence-transformers unavailable ({type(e).__name__}); using hashing embedder")
    return HashingEmbedder()
