"""Parse NVD CVE API 2.0 responses (one or many pages of JSON)."""
from __future__ import annotations

import json
from pathlib import Path

from aman.processing.cleaning import clean_text
from aman.schema import Document

_CVSS_KEYS = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")


def _english_description(cve: dict) -> str:
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            return d.get("value", "")
    return ""


def _best_cvss(cve: dict) -> dict:
    metrics = cve.get("metrics", {})
    for key in _CVSS_KEYS:
        if metrics.get(key):
            entry = metrics[key][0]
            data = entry.get("cvssData", {})
            return {
                "cvss_version": data.get("version", ""),
                "cvss_score": data.get("baseScore"),
                "severity": data.get("baseSeverity") or entry.get("baseSeverity", ""),
                "vector": data.get("vectorString", ""),
            }
    return {}


def _affected_products(cve: dict, limit: int = 15) -> list[str]:
    seen: list[str] = []
    for conf in cve.get("configurations", []):
        for node in conf.get("nodes", []):
            for m in node.get("cpeMatch", []):
                if not m.get("vulnerable"):
                    continue
                parts = m.get("criteria", "").split(":")  # cpe:2.3:a:vendor:product:version
                if len(parts) > 5:
                    label = f"{parts[3]} {parts[4]} {parts[5]}".replace("*", "").strip()
                    if label not in seen:
                        seen.append(label)
    return seen[:limit]


def parse_nvd(payload: dict) -> list[Document]:
    docs = []
    for item in payload.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        if not cve_id:
            continue
        if cve.get("vulnStatus") == "Rejected":
            continue
        description = clean_text(_english_description(cve))
        cvss = _best_cvss(cve)
        products = _affected_products(cve)
        refs = [r["url"] for r in cve.get("references", []) if r.get("url")][:10]

        lines = [description]
        if cvss:
            lines.append(
                f"Severity: {cvss['severity']} (CVSS {cvss['cvss_version']} base score {cvss['cvss_score']}). "
                f"Vector: {cvss['vector']}"
            )
        if products:
            lines.append("Affected products: " + "; ".join(products))
        if refs:
            lines.append("References: " + " ".join(refs))
        docs.append(
            Document(
                doc_id=cve_id,
                source="NVD",
                title=cve_id,
                text="\n\n".join(l for l in lines if l),
                url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
                metadata={"type": "cve", "published": cve.get("published", ""), **cvss},
            )
        )
    return docs


def load_nvd(path: str | Path) -> list[Document]:
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    docs: list[Document] = []
    for f in files:
        docs.extend(parse_nvd(json.loads(f.read_text(encoding="utf-8"))))
    return docs
