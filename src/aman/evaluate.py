"""Measure retrieval and answer quality on a labelled question set.

Retrieval : Precision@K, Recall@K, MRR (on answerable questions, document level)
Abstention: does AMAN refuse when the knowledge base has no answer (and NOT refuse when it does)?
Grounding : proxy for faithfulness - share of answer sentences that carry a valid [n] citation
            and overlap strongly with the cited passage. (A true faithfulness score needs an LLM judge.)

Usage: PYTHONPATH=src python -m aman.evaluate [--k 5] [--set data/eval/eval_set.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from aman.rag import AmanService
from aman.text import tokenize


def _unique_docs(svc: AmanService, hits) -> list[str]:
    seen, out = set(), []
    for h in hits:
        d = svc.index.chunks[h.idx]["doc_id"]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def grounding(answer: str, sources: list[dict]) -> float:
    by_n = {s["n"]: set(tokenize(s["text"])) for s in sources}
    answer = re.sub(r"([.!?])\s+((?:\[\d+\])+)", r" \2\1", answer)   # "text. [1]" -> "text [1]."
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.strip()) > 25 and not s.startswith("#")]
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        cites = [int(n) for n in re.findall(r"\[(\d+)\]", s)]
        toks = set(tokenize(re.sub(r"\[\d+\]", "", s)))
        if cites and toks and all(c in by_n for c in cites):
            if max(len(toks & by_n[c]) / len(toks) for c in cites) >= 0.6:
                ok += 1
    return ok / len(sents)


def evaluate(svc: AmanService, items: list[dict], k: int = 5) -> dict:
    p, r, rr, ground = [], [], [], []
    abst_ok = abst_total = false_abst = ans_total = 0
    failures = []
    for it in items:
        rel = set(it["relevant"])
        res = svc.ask(it["q"])
        if not rel:
            abst_total += 1
            abst_ok += res["abstained"]
            if not res["abstained"]:
                failures.append(("should abstain", it["q"]))
            continue
        ans_total += 1
        false_abst += res["abstained"]
        hits, _ = svc.retrieve(it["q"])
        docs = _unique_docs(svc, hits)[:k]
        found = [d for d in docs if d in rel]
        p.append(len(found) / k if k else 0)
        r.append(len(set(found)) / len(rel))
        first = next((i for i, d in enumerate(docs, 1) if d in rel), None)
        rr.append(1 / first if first else 0.0)
        if not res["abstained"]:
            ground.append(grounding(res["answer"], res["sources"]))
        if first is None:
            failures.append(("missed", it["q"]))
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {f"Precision@{k}": avg(p), f"Recall@{k}": avg(r), "MRR": avg(rr),
            "Abstention accuracy (unanswerable)": round(abst_ok / abst_total, 3) if abst_total else None,
            "False-abstain rate (answerable)": round(false_abst / ans_total, 3) if ans_total else None,
            "Grounding (faithfulness proxy)": avg(ground), "questions": len(items), "failures": failures}


def main():
    from aman.server import ROOT, create_service
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--set", type=Path, default=ROOT / "data/eval/eval_set.json")
    a = ap.parse_args()
    svc = create_service(db=":memory:")
    res = evaluate(svc, json.loads(a.set.read_text(encoding="utf-8")), a.k)
    for k, v in res.items():
        print(f"{k:42s} {v}")


if __name__ == "__main__":
    main()
