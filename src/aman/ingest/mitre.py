"""Parse the MITRE ATT&CK STIX 2.x bundle (enterprise-attack.json)."""
from __future__ import annotations

import json
from pathlib import Path

from aman.processing.cleaning import clean_text
from aman.schema import Document


def _attack_ref(obj: dict) -> tuple[str, str]:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref["external_id"], ref.get("url", "")
    return "", ""


def parse_mitre(bundle: dict) -> list[Document]:
    docs = []
    for obj in bundle.get("objects", []):
        if obj.get("type") == "x-mitre-tactic" and not (obj.get("revoked") or obj.get("x_mitre_deprecated")):
            tid, url = _attack_ref(obj)
            if tid:
                docs.append(Document(
                    doc_id=tid, source="MITRE ATT&CK", title=f"{tid} {obj['name']} (tactic)",
                    text=clean_text(obj.get("description", "")), url=url,
                    metadata={"type": "tactic", "shortname": obj.get("x_mitre_shortname", "")}))
            continue
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue  # never serve outdated knowledge
        tid, url = _attack_ref(obj)
        if not tid:
            continue
        tactics = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
        parts = [clean_text(obj.get("description", ""))]
        detection = clean_text(obj.get("x_mitre_detection", ""))
        if detection:
            parts.append(f"Detection: {detection}")
        if tactics:
            parts.append("Tactics: " + ", ".join(tactics))
        platforms = obj.get("x_mitre_platforms", [])
        if platforms:
            parts.append("Platforms: " + ", ".join(platforms))
        docs.append(
            Document(
                doc_id=tid,
                source="MITRE ATT&CK",
                title=f"{tid} {obj['name']}",
                text="\n\n".join(p for p in parts if p),
                url=url,
                metadata={
                    "type": "technique",
                    "tactics": tactics,
                    "is_subtechnique": bool(obj.get("x_mitre_is_subtechnique")),
                },
            )
        )
    return docs


def load_mitre(path: str | Path) -> list[Document]:
    return parse_mitre(json.loads(Path(path).read_text(encoding="utf-8")))
