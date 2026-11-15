"""Compute the vector index OFFLINE so the deployed app only embeds user questions.

    AMAN_EMBEDDER=gemini python -m aman.build_index
Progress is saved every 100 chunks; if it stops (quota, Ctrl+C), run the same command again and it continues.
"""
from aman.embeddings import make_embedder
from aman.index import KnowledgeIndex
from aman.server import ROOT

if __name__ == "__main__":
    chunks = ROOT / "data/processed/chunks.jsonl"
    emb = make_embedder()
    print(f"[aman] embedder: {emb.name}", flush=True)
    try:
        idx = KnowledgeIndex(chunks, emb, chunks.parent, build_vectors=True)
    except KeyboardInterrupt:
        print("\n[aman] Stopped by you. Progress is saved; run the same command again to continue.")
        raise SystemExit(1)
    except Exception as e:
        print(f"\n[aman] Stopped: {e}")
        print("[aman] Progress is saved. Run the same command again (later, if it is a quota limit) and it continues.")
        raise SystemExit(1)
    print(f"[aman] DONE: indexed {len(idx.chunks)} chunks with {emb.name}; vectors shape {idx.vectors.shape}")
