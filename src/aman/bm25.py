"""BM25 keyword search implemented from scratch (no dependencies).

score(q, d) = sum over query terms t of  idf(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1-b+b*|d|/avgdl))
"""
from __future__ import annotations

import math
from collections import Counter

from aman.text import tokenize


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings.setdefault(t, []).append(i)

    def score(self, query: str, top_k: int = 30) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for t in set(tokenize(query)):
            if t not in self.idf:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avgdl or 1))
                scores[i] = scores.get(i, 0.0) + self.idf[t] * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]
