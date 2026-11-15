"""Split Documents into Chunks.

Why chunk? Embedding models work best on focused passages, and the LLM
has a limited context window. We split on paragraph boundaries first
(keeps ideas intact), then on sentences only if a paragraph is too long.
Every chunk is prefixed with the document title so it still makes sense
when retrieved on its own.
"""
from __future__ import annotations

import re

from aman.schema import Chunk, Document

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    parts, current = [], ""
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        # a single giant sentence: hard-cut it
        while len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) + 1 > max_chars and current:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def split_text(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    if max_chars <= overlap:
        raise ValueError("max_chars must be larger than overlap")
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        pieces.extend(_split_long(para, max_chars) if len(para) > max_chars else [para])

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {piece}".strip() if tail else piece
        else:
            current = f"{current}\n\n{piece}".strip() if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: Document, max_chars: int = 1000, overlap: int = 150) -> list[Chunk]:
    header = f"{doc.title} ({doc.source})"
    out = []
    for i, body in enumerate(split_text(doc.text, max_chars, overlap)):
        out.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::{i}",
                doc_id=doc.doc_id,
                source=doc.source,
                title=doc.title,
                text=f"{header}\n{body}",
                url=doc.url,
                metadata={**doc.metadata, "chunk_index": i},
            )
        )
    return out
