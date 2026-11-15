"""Raw files -> clean Documents -> Chunks -> data/processed/chunks.jsonl

Usage:  python -m aman.build_corpus --raw data/raw --out data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aman.ingest.kev import load_kev
from aman.ingest.mitre import load_mitre
from aman.ingest.owasp import load_owasp
from aman.ingest.nvd import load_nvd
from aman.processing.chunker import chunk_document


def merge_cves(nvd_docs, kev_docs):
    """One document per CVE: NVD details (score, products) enriched with CISA KEV status; KEV-only CVEs kept."""
    by_id = {d.doc_id: d for d in nvd_docs}
    for k in kev_docs:
        d = by_id.get(k.doc_id)
        if d is None:
            by_id[k.doc_id] = k
        else:
            d.text += "\n\n" + k.text.split("\n\n", 1)[-1]
            d.metadata.update({key: v for key, v in k.metadata.items() if key.startswith("kev")})
            d.source = "NVD + CISA KEV"
    return list(by_id.values())


def build(raw: Path, out: Path, max_chars: int = 1000, overlap: int = 150, seed: bool = False) -> dict:
    docs = []
    if seed:
        from aman.seed import seed_documents
        docs += seed_documents()
    mitre = raw / "enterprise-attack.json"
    if mitre.exists():
        docs += load_mitre(mitre)
    nvd_docs = load_nvd(raw / "nvd") if (raw / "nvd").exists() else []
    kev_docs = load_kev(raw / "kev.json") if (raw / "kev.json").exists() else []
    docs += merge_cves(nvd_docs, kev_docs)
    if (raw / "owasp").exists():
        docs += load_owasp(raw / "owasp")
    if not docs:
        raise SystemExit(f"No source files found in {raw}. Run scripts/download_data.py or use --seed.")

    out.parent.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    with out.open("w", encoding="utf-8") as f:
        for d in docs:
            for c in chunk_document(d, max_chars, overlap):
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                n_chunks += 1
    return {"documents": len(docs), "chunks": n_chunks, "output": str(out)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--seed", action="store_true", help="include the built-in sample knowledge base")
    a = ap.parse_args()
    print(build(a.raw, a.out, a.max_chars, a.overlap, a.seed))


if __name__ == "__main__":
    main()
