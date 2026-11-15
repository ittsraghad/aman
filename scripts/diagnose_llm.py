"""Run ONE question through the real pipeline and print exactly what happens (no web UI needed).

Usage (same terminal where GEMINI_API_KEY etc. are exported):
    PYTHONPATH=src python scripts/diagnose_llm.py "What is CVE-2024-3094?"
It prints the first events, the provider used, and the real error if the LLM fails.
"""
import sys

from aman.http import describe_error
from aman.llm import stream_text
from aman.server import create_service

q = " ".join(sys.argv[1:]) or "What is CVE-2024-3094?"
svc = create_service(db=":memory:")
print("embedder:", svc.index.embedder.name, "| llm:", getattr(svc.llm, "mode", "?"),
      "| models:", getattr(svc.llm, "models", "-"))
p = svc._prepare(q, [])
if p["final"]:
    print("no LLM needed:", p["final"]["intent"], p["final"]["answer"][:120])
    raise SystemExit
print("sources:", [s["doc_id"] for s in p["sources"]])
try:
    n = 0
    for piece in stream_text(svc.llm, p["question"], p["sources"], p["lang"], p["history"]):
        n += 1
        if n <= 3:
            print("chunk", n, "->", repr(piece[:80]))
    print("OK: received", n, "chunks from", getattr(svc.llm, "last_model", svc.llm.mode))
except Exception as e:
    print("FAILED ->", describe_error(e))
