"""Quick check that the embedding provider works, and that Arabic and English meet in the same space.
Usage: AMAN_EMBEDDER=gemini PYTHONPATH=src python scripts/diagnose_embed.py"""
from aman.embeddings import make_embedder

e = make_embedder()
print("embedder:", e.name)
a = e.encode_query("SQL injection")
b = e.encode_query("حقن SQL")
c = e.encode_query("how to cook pasta")
p = e.encode_passages(["SQL injection is an attack where untrusted input is inserted into a database query."])
print("dimensions:", len(a))
print("English vs Arabic (same topic):", round(float(a @ b), 3), "   <- should be clearly higher than the next line")
print("English vs unrelated question :", round(float(a @ c), 3))
print("query vs matching passage     :", round(float(a @ p[0]), 3))
