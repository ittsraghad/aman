# Architecture

```mermaid
flowchart LR
    Q[User question] --> I{Intent router}
    I -->|greeting / about / abuse| C[Canned reply]
    I -->|needs context| K[Ask what "this" refers to]
    I -->|security question| R[Retrieval]
    R --> B[BM25 keyword search]
    R --> V[Vector search]
    B --> F[RRF fusion]
    V --> F
    F --> RR[Rerank]
    RR --> G{Enough evidence?}
    G -->|no| A[Abstain + suggestions]
    G -->|yes| L[LLM streams a grounded answer with citations]
```

## Request flow
1. **Intent router** (rules): greetings, "what can you do", and requests for attack tools never reach retrieval or the LLM.
2. **Follow-ups**: the browser sends the last turns; pronouns ("it", "هذا") are resolved against the last answered question.
3. **Hybrid retrieval**: BM25 (identifiers such as `CVE-2021-44228` and `T1003` are single tokens, so they match exactly) plus
   embedding similarity, fused with Reciprocal Rank Fusion, then reranked. An exact CVE/ATT&CK id always surfaces first.
4. **Abstention gate**: if the best passage covers too little of the question, AMAN says so instead of guessing.
5. **Generation**: the LLM may only use the numbered passages and must cite them. Answers are streamed (NDJSON).
6. **Failure handling**: provider failure falls back to an extractive answer; embedding failure falls back to keyword search;
   a per-IP rate limit and a daily LLM-call cap protect a public deployment.

## Privacy
No question or answer text is stored on the server. Chats live in the visitor's browser. Only anonymous counters are kept
(retrieved topics, thumbs up/down). The question text is sent to the LLM provider to produce the answer.

## Modules
| Module | Role |
|---|---|
| `ingest/` | Parsers for MITRE (STIX), NVD, CISA KEV, OWASP Markdown |
| `processing/` | Cleaning and chunking |
| `index.py`, `bm25.py`, `embeddings.py` | Keyword index, vector index (resumable build), embedding providers |
| `search.py` | RRF fusion, reranking |
| `intent.py`, `rag.py` | Routing, memory, abstention, orchestration, streaming |
| `llm.py`, `http.py` | Gemini / Claude / extractive providers, retries that obey the provider's advice |
| `server.py`, `web/` | HTTP API and single-file web UI |
| `evaluate.py` | Precision@K, Recall@K, MRR, abstention accuracy, grounding |
