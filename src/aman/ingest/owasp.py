"""Parse OWASP Markdown (Cheat Sheet Series / Top 10) into one Document per section.
Fenced code blocks are dropped (noise for retrieval); links and images are cleaned."""
from __future__ import annotations

import re
from pathlib import Path

from aman.processing.cleaning import clean_text
from aman.schema import Document

_FENCE = re.compile(r"```.*?```", re.S)
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FRONT = re.compile(r"\A---.*?---\s*", re.S)
CHEATSHEET_URL = "https://cheatsheetseries.owasp.org/cheatsheets/{name}.html"


def parse_markdown(md: str, stem: str, url: str, source: str = "OWASP Cheat Sheet Series") -> list[Document]:
    md = _FRONT.sub("", md)
    md = _IMG.sub("", _FENCE.sub("", md))
    title_m = re.search(r"^#\s+(.+)$", md, re.M)
    title = clean_text(title_m.group(1)) if title_m else stem.replace("_", " ")
    sections = re.split(r"^##\s+", md, flags=re.M)
    docs = []
    for i, sec in enumerate(sections):
        if i == 0:                       # intro before the first "##"
            heading, body = "Introduction", re.sub(r"^#\s+.+$", "", sec, flags=re.M)
        else:
            heading, _, body = sec.partition("\n")
        body = clean_text(re.sub(r"^#{3,}\s*", "", body, flags=re.M))
        if len(body) < 120:
            continue
        docs.append(Document(
            doc_id=f"OWASP-{stem}-{i}", source=source, title=f"{title}: {clean_text(heading)}",
            text=body, url=url, metadata={"type": "guide", "sheet": title}))
    return docs


def load_owasp(root: str | Path) -> list[Document]:
    docs: list[Document] = []
    for f in sorted(Path(root).rglob("*.md")):
        if f.name.lower() in ("readme.md", "index.md") or "Glossary" in f.name:
            continue
        docs += parse_markdown(f.read_text(encoding="utf-8", errors="ignore"), f.stem, CHEATSHEET_URL.format(name=f.stem))
    return docs
