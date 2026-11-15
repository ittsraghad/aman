# AMAN — أمان

**A bilingual (Arabic / English) cybersecurity assistant that answers from trusted sources and shows its evidence.**
مساعد ذكاء اصطناعي للأمن السيبراني يجاوب من مصادر موثوقة ويعرض لك المراجع.

## Live Demo
Try AMAN: https://aman-yg9r.onrender.com/

GitHub: https://github.com/ittsraghad/aman

AMAN is a Retrieval-Augmented Generation (RAG) system built around MITRE ATT&CK, the NVD, CISA's Known Exploited
Vulnerabilities catalog and the OWASP Cheat Sheet Series. It cites the passages it used, says "I don't know" when the
sources do not cover a question, and refuses to help build attack tools.

## Features
- **Grounded answers with citations** in Arabic or English; every `[n]` opens the exact passage and its original page.
- **Streaming chat** with follow-up memory ("How can I protect myself from this?" after a CVE question).
- **Hybrid retrieval**: BM25 keyword search (exact CVE / ATT&CK ids) + embedding search, fused with RRF, then reranked.
- **Abstention**: low-evidence questions get an honest "not enough information" plus suggestions, not a guess.
- **Guardrails**: an intent router handles greetings and refuses requests to build malware or break into accounts, before any LLM call.
- **CVE lookup** with CVSS gauge, affected products, CISA KEV status (exploited in the wild, ransomware use) and required action.
- **MITRE ATT&CK explorer** by tactic, and a **dashboard** with corpus statistics and user feedback.
- **Privacy by design**: no chat text is stored server-side; history lives in the browser.
- **Public-demo hardening**: per-IP rate limit, daily LLM-call cap, security headers, graceful fallbacks.

## Knowledge base
MITRE ATT&CK (enterprise), CISA KEV, recent critical CVEs from the NVD API and the OWASP Cheat Sheet Series:
about **3.9k documents / 7.1k passages** in the current build. See `NOTICE.md` for attribution and terms.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m aman.server            # http://127.0.0.1:8000  (offline sample corpus if no data is built)
```
With real data and Gemini:
```bash
export GEMINI_API_KEY=...        # https://aistudio.google.com
python scripts/check_providers.py                       # valid model names for your key
python scripts/download_data.py all                     # MITRE + CISA KEV + OWASP + NVD
python -m aman.build_corpus                             # raw data -> data/processed/chunks.jsonl
export AMAN_LLM=gemini AMAN_GEMINI_MODELS="gemini-3.5-flash,gemini-3.5-flash-lite"
python -m aman.server
```
Optional vector search: `AMAN_EMBEDDER=gemini python -m aman.build_index` (resumable; free-tier embedding quotas
mean large corpora are built over several days). Without the vector index AMAN runs keyword-only. All settings: `.env.example`.

## Tests and evaluation
```bash
python -m unittest discover -s tests -v     # 65 tests; LLM/embedding providers are tested with mocked HTTP
python -m aman.evaluate                     # Precision@K, Recall@K, MRR, abstention accuracy, grounding
```
The evaluation harness and its metrics are included; a labelled question set for the full corpus and an ablation study
(BM25 vs vectors vs hybrid vs reranked) are on the roadmap, and results will be published here.

## Documentation
[Architecture](docs/ARCHITECTURE.md) · [How it works (Arabic)](docs/HOW_IT_WORKS.md) · [Deployment](docs/DEPLOY.md)

## Limitations
- Answers come from a language model and can be wrong; verify against the sources shown.
- Coverage is limited to the listed sources and the date of the last data refresh.
- Arabic retrieval currently relies on a small glossary until the multilingual vector index is complete.
- The intent router is rule-based.

## Roadmap
Complete the vector index and measure its gain · labelled evaluation set and ablation study · parse ATT&CK mitigations ·
a trained intent classifier · scheduled data refresh.

## License
Code: MIT (`LICENSE`). Data: third-party terms, see `NOTICE.md`.
