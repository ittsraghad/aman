"""Parse the CISA Known Exploited Vulnerabilities (KEV) catalog JSON.
KEV tells us a vulnerability is exploited in the wild and gives the official required action."""
from __future__ import annotations

import json
from pathlib import Path

from aman.processing.cleaning import clean_text
from aman.schema import Document


def parse_kev(payload: dict) -> list[Document]:
    docs = []
    for v in payload.get("vulnerabilities", []):
        cve = v.get("cveID")
        if not cve:
            continue
        action = clean_text(v.get("requiredAction", ""))
        ransomware = v.get("knownRansomwareCampaignUse", "Unknown")
        parts = [
            f"{clean_text(v.get('vulnerabilityName', ''))}. {clean_text(v.get('shortDescription', ''))}".strip(". "),
            f"CISA Known Exploited Vulnerabilities (KEV): this vulnerability is being exploited in the wild. "
            f"Affected: {v.get('vendorProject', '')} {v.get('product', '')}. "
            f"Added to the KEV catalog on {v.get('dateAdded', '')}; federal remediation due date {v.get('dueDate', '')}. "
            f"Known ransomware campaign use: {ransomware}.",
        ]
        if action:
            parts.append(f"Required action: {action}")
        docs.append(Document(
            doc_id=cve, source="CISA KEV", title=cve, text="\n\n".join(p for p in parts if p),
            url=f"https://nvd.nist.gov/vuln/detail/{cve}",
            metadata={"type": "cve", "kev": True, "kev_added": v.get("dateAdded", ""), "kev_due": v.get("dueDate", ""),
                      "kev_ransomware": ransomware, "kev_action": action}))
    return docs


def load_kev(path: str | Path) -> list[Document]:
    return parse_kev(json.loads(Path(path).read_text(encoding="utf-8")))
