"""Core data structures shared by every stage of the pipeline.

Document = a full item from a source (one CVE, one ATT&CK technique).
Chunk    = a small piece of a Document that we embed and retrieve.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Document:
    doc_id: str          # stable id, e.g. "CVE-2024-3094" or "T1003"
    source: str          # e.g. "NVD", "MITRE ATT&CK"
    title: str
    text: str
    url: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str        # "<doc_id>::<index>"
    doc_id: str
    source: str
    title: str
    text: str
    url: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
