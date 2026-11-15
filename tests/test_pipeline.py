import json
import tempfile
import unittest
from pathlib import Path

from aman.build_corpus import build
from aman.ingest.mitre import parse_mitre
from aman.ingest.nvd import parse_nvd
from aman.processing.chunker import chunk_document, split_text
from aman.processing.cleaning import clean_text
from aman.schema import Document

MITRE_SAMPLE = {"objects": [
    {"type": "attack-pattern", "name": "OS Credential Dumping",
     "description": "Adversaries may attempt to dump credentials (Citation: Mandiant APT1) from [LSASS](https://x.y/z).",
     "external_references": [{"source_name": "mitre-attack", "external_id": "T1003",
                              "url": "https://attack.mitre.org/techniques/T1003"}],
     "kill_chain_phases": [{"phase_name": "credential-access"}],
     "x_mitre_detection": "Monitor <code>lsass.exe</code> access.", "x_mitre_platforms": ["Windows"]},
    {"type": "attack-pattern", "name": "Old", "revoked": True,
     "external_references": [{"source_name": "mitre-attack", "external_id": "T0000"}]},
    {"type": "malware", "name": "NotATechnique"},
]}

NVD_SAMPLE = {"vulnerabilities": [
    {"cve": {"id": "CVE-2024-3094", "published": "2024-03-29T17:15:21.150",
             "descriptions": [{"lang": "es", "value": "hola"},
                              {"lang": "en", "value": "Malicious code was discovered in xz upstream tarballs."}],
             "metrics": {"cvssMetricV31": [{"cvssData": {"version": "3.1", "baseScore": 10.0,
                         "baseSeverity": "CRITICAL", "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}}]},
             "configurations": [{"nodes": [{"cpeMatch": [
                 {"vulnerable": True, "criteria": "cpe:2.3:a:tukaani:xz:5.6.0:*:*:*:*:*:*:*"},
                 {"vulnerable": False, "criteria": "cpe:2.3:o:debian:debian_linux:12:*:*:*:*:*:*:*"}]}]}],
             "references": [{"url": "https://www.openwall.com/lists/oss-security/2024/03/29/4"}]}},
    {"cve": {"id": "CVE-2000-0001", "vulnStatus": "Rejected", "descriptions": []}},
]}


class CleaningTests(unittest.TestCase):
    def test_removes_citations_links_tags(self):
        out = clean_text("A (Citation: X) [link](https://a.b) <code>c</code>")
        self.assertEqual(out, "A link c")


class MitreTests(unittest.TestCase):
    def test_parse(self):
        docs = parse_mitre(MITRE_SAMPLE)
        self.assertEqual([d.doc_id for d in docs], ["T1003"])   # revoked + non-technique dropped
        d = docs[0]
        self.assertNotIn("Citation", d.text)
        self.assertIn("Tactics: credential-access", d.text)
        self.assertIn("Detection: Monitor lsass.exe access.", d.text)


class NvdTests(unittest.TestCase):
    def test_parse(self):
        docs = parse_nvd(NVD_SAMPLE)
        self.assertEqual(len(docs), 1)                           # rejected dropped
        d = docs[0]
        self.assertIn("xz upstream tarballs", d.text)
        self.assertNotIn("hola", d.text)                         # English only
        self.assertIn("CRITICAL", d.text)
        self.assertIn("tukaani xz 5.6.0", d.text)
        self.assertNotIn("debian", d.text)                       # non-vulnerable CPE ignored
        self.assertEqual(d.metadata["cvss_score"], 10.0)


class ChunkerTests(unittest.TestCase):
    def test_short_doc_single_chunk_with_header(self):
        doc = Document("CVE-1", "NVD", "CVE-1", "short text")
        chunks = chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("CVE-1 (NVD)"))
        self.assertEqual(chunks[0].chunk_id, "CVE-1::0")

    def test_long_text_respects_limit_and_overlaps(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(200))
        parts = split_text(text, max_chars=300, overlap=50)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 300 + 50 + 2 for p in parts))
        self.assertIn(parts[0][-20:].strip(), parts[1])           # overlap carried over

    def test_huge_single_sentence_is_cut(self):
        parts = split_text("x" * 2500, max_chars=1000, overlap=100)
        self.assertGreaterEqual(len(parts), 3)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            split_text("a", max_chars=100, overlap=100)


class EndToEndTests(unittest.TestCase):
    def test_build(self):
        with tempfile.TemporaryDirectory() as t:
            raw = Path(t) / "raw"; (raw / "nvd").mkdir(parents=True)
            (raw / "enterprise-attack.json").write_text(json.dumps(MITRE_SAMPLE))
            (raw / "nvd" / "a.json").write_text(json.dumps(NVD_SAMPLE))
            out = Path(t) / "out" / "chunks.jsonl"
            stats = build(raw, out)
            self.assertEqual(stats["documents"], 2)
            rows = [json.loads(l) for l in out.read_text().splitlines()]
            self.assertEqual({r["doc_id"] for r in rows}, {"T1003", "CVE-2024-3094"})


if __name__ == "__main__":
    unittest.main()
