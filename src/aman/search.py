"""Hybrid search (BM25 + vectors fused with Reciprocal Rank Fusion), then reranking.

RRF: score(d) = sum over rankers of 1/(k + rank). It needs no score normalisation,
which is why it's the standard way to merge keyword and semantic rankings.
"""
from __future__ import annotations

from dataclasses import dataclass

from aman.index import KnowledgeIndex
from aman.text import ATTACK_RE, CVE_RE, expand_query, tokenize


@dataclass
class Hit:
    idx: int
    rrf: float = 0.0
    relevance: float = 0.0   # 0..1 idf-weighted query coverage (used for abstain gate)
    rerank: float = 0.0
    exact_id: bool = False


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def coverage(index: KnowledgeIndex, query: str, chunk_idx: int) -> float:
    """Fraction of the query's information (idf mass) found in the chunk."""
    q_tokens = set(tokenize(expand_query(query)))
    q_tokens = {t for t in q_tokens if t in index.doc_freq}
    if not q_tokens:
        return 0.0
    c = index.chunks[chunk_idx]
    have = set(tokenize(c["title"] + " " + c["text"]))
    total = sum(index.doc_freq[t] for t in q_tokens)
    return sum(index.doc_freq[t] for t in q_tokens if t in have) / total


class LexicalReranker:
    """Cheap reranker: idf-weighted coverage + title match bonus. Swap for a cross-encoder in prod."""
    name = "lexical"

    def score(self, index: KnowledgeIndex, query: str, idxs: list[int]) -> dict[int, float]:
        q_title = set(tokenize(expand_query(query)))
        out = {}
        for i in idxs:
            title_tokens = set(tokenize(index.chunks[i]["title"]))
            bonus = 0.25 if q_title & title_tokens else 0.0
            out[i] = coverage(index, query, i) + bonus
        return out


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)

    def score(self, index, query, idxs):
        q = expand_query(query)
        scores = self.model.predict([(q, index.chunks[i]["text"]) for i in idxs])
        return {i: float(s) for i, s in zip(idxs, scores)}


def make_reranker():
    import os
    if os.getenv("AMAN_RERANKER") == "cross":
        return CrossEncoderReranker()
    return LexicalReranker()


def hybrid_search(index: KnowledgeIndex, query: str, reranker, top_k: int = 5,
                  candidates: int = 30, max_per_doc: int = 2) -> list[Hit]:
    kw = [i for i, _ in index.keyword_search(query, candidates)]
    vec = [i for i, _ in index.vector_search(query, candidates)]
    fused = rrf_fuse([kw, vec])

    # Exact identifiers (CVE-… / T1003) must always surface first.
    ids = {m.upper() for m in CVE_RE.findall(query)} | {m.upper() for m in ATTACK_RE.findall(query)}
    exact = {i for d in ids for i in index.by_doc.get(d, [])}

    hits = {i: Hit(i, rrf=s, exact_id=i in exact) for i, s in fused.items()}
    for i in exact:
        hits.setdefault(i, Hit(i, exact_id=True))

    pool = sorted(hits.values(), key=lambda h: (-h.exact_id, -h.rrf))[: max(top_k * 3, 15)]
    rr = reranker.score(index, query, [h.idx for h in pool])
    for h in pool:
        h.rerank = rr[h.idx]
        h.relevance = coverage(index, query, h.idx)
    pool.sort(key=lambda h: (-h.exact_id, -h.rerank, -h.rrf))

    out, per_doc = [], {}
    for h in pool:
        d = index.chunks[h.idx]["doc_id"]
        if per_doc.get(d, 0) >= max_per_doc:
            continue
        per_doc[d] = per_doc.get(d, 0) + 1
        out.append(h)
        if len(out) == top_k:
            break
    return out
