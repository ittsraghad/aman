"""The AMAN brain: question -> understand -> retrieve -> (abstain?) -> generate -> cite."""
from __future__ import annotations

import os
import re

from aman.index import KnowledgeIndex
from aman.http import describe_error
from aman.intent import CANNED, SUGGESTIONS, classify
from aman.llm import ExtractiveLLM, stream_text
from aman.ratelimit import DailyCap
from aman.search import hybrid_search
from aman.store import Store
from aman.text import ATTACK_RE, CVE_RE, detect_language, expand_query, normalize_arabic, tokenize

FOLLOWUP_WORDS = {"it", "its", "this", "that", "them", "these", "those", "they", "there", "also", "more",
                  "هذا", "هذه", "هذي", "هاذا", "هاذي", "ذا", "ذلك", "تلك", "منه", "عنه", "منها", "عنها", "فيه", "فيها",
                  "به", "بها", "عليه", "عليها", "له", "لها", "كذلك"}
_FOLLOWUP_NORM = {normalize_arabic(w) for w in FOLLOWUP_WORDS}
_WORD = re.compile(r"[a-z]+|[\u0621-\u063a\u0641-\u064a]+")   # letters only: "؟" must not stick to a word


ABSTAIN = {
    "en": "I couldn't find reliable information about this in my sources, so I won't guess. I'm built for "
          "cybersecurity questions: try a CVE ID, an ATT&CK technique, or a more specific security term. For example:",
    "ar": "ما لقيت معلومات موثوقة عن هذا السؤال في مصادري، ولذلك ما راح أخمّن. أنا مخصص لأسئلة الأمن السيبراني: "
          "جرّب رقم CVE أو اسم تقنية من ATT&CK أو مصطلحًا أمنيًا أدق. مثلًا:",
}


def _known(tok: str, vocab) -> bool:
    """Token is known if the KB has it, or a longer word sharing its stem (prevent ~ prevention)."""
    return tok in vocab or (len(tok) >= 5 and any(v.startswith(tok) for v in vocab))


def is_followup(question: str, known_topics: frozenset = frozenset(), vocab: frozenset | None = None) -> bool:
    """Does the question lean on earlier context?
    yes: has a pronoun ("it", "هذا") or no content words ("Why?").
    yes: short and made only of generic words the knowledge base knows ("How to prevent?") or covered
         by the Arabic glossary ("كيف أمنعه؟").
    no : names a topic ("What is CSRF?") or contains words the KB has never seen ("how do I cook pasta")
         - those are standalone questions, and must not borrow context to get an answer."""
    if CVE_RE.search(question) or ATTACK_RE.search(question):
        return False
    words = set(_WORD.findall(normalize_arabic(question.lower())))
    if words & _FOLLOWUP_NORM:
        return True
    toks = tokenize(question)
    if not toks:
        return True
    if len(toks) >= 3 or set(toks) & known_topics:
        return False
    if expand_query(question) != question:
        return True
    return vocab is not None and all(_known(t, vocab) for t in toks)


class AmanService:
    def __init__(self, index: KnowledgeIndex, reranker, llm, store: Store, min_relevance: float = 0.34, top_k: int = 5):
        self.index, self.reranker, self.llm, self.store = index, reranker, llm, store
        self.min_relevance, self.top_k = min_relevance, top_k
        self.cap = DailyCap(int(os.getenv("AMAN_DAILY_LLM_CAP", "1000")))
        self.vocab = frozenset(index.bm25.idf)
        self.known_topics = frozenset(t for c in index.chunks for t in tokenize(c["title"]))

    def _followup(self, question: str) -> bool:
        return is_followup(question, self.known_topics, self.vocab)

    # ---- retrieval -------------------------------------------------------
    def retrieve(self, query: str):
        hits = hybrid_search(self.index, query, self.reranker, top_k=self.top_k)
        best = max((1.0 if h.exact_id else h.relevance for h in hits), default=0.0)
        return hits, best

    def _sources(self, hits) -> list[dict]:
        out = []
        for n, h in enumerate(hits, 1):
            c = self.index.chunks[h.idx]
            body = c["text"].split("\n", 1)[-1]
            out.append({"n": n, "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "title": c["title"],
                        "source": c["source"], "url": c["url"], "section": c["metadata"].get("chunk_index", 0),
                        "text": c["text"], "snippet": body[:320], "score": round(h.relevance, 3),
                        "sample": bool(c["metadata"].get("seed"))})
        return out

    # ---- conversation (stateless: the browser sends the recent history) -------------------
    @staticmethod
    def _clean_history(history) -> list[dict]:
        out = []
        for m in (history or [])[-8:] if isinstance(history, list) else []:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                item = {"role": m["role"], "content": m["content"][:1500]}
                if isinstance(m.get("search_query"), str):
                    item["search_query"] = m["search_query"][:300]
                if m.get("abstained") is True:
                    item["abstained"] = True
                out.append(item)
        return out

    def _canned(self, intent: str, lang: str) -> dict:
        return {"answer": CANNED[intent][lang], "sources": [], "language": lang, "abstained": False, "confidence": 1.0,
                "mode": "canned", "search_query": "", "intent": intent, "suggestions": SUGGESTIONS[lang]}

    def _prepare(self, question: str, history) -> dict:
        """Everything before text generation. Returns {"final": result} when no LLM is needed."""
        question = (question or "").strip()[:1000]
        if not question:
            raise ValueError("empty question")
        lang = detect_language(question)
        intent = classify(question)
        if intent != "security":
            return {"final": self._canned(intent, lang)}
        history = self._clean_history(history)

        search_query = question
        if self._followup(question):
            # topic of the conversation = the last answered (not abstained) security question
            prev = next((m["search_query"] for m in reversed(history)
                         if m.get("search_query") and not m.get("abstained")), "")
            if not prev:
                prev = next((m["content"] for m in reversed(history)
                             if m["role"] == "user" and classify(m["content"]) == "security"
                             and not self._followup(m["content"])), "")
            if prev:
                search_query = (prev + " " + question)[-300:]
            elif not (set(tokenize(question)) & self.known_topics):
                return {"final": self._canned("clarify", lang)}     # "this/it" with nothing to refer to: ask

        hits, best = self.retrieve(search_query)
        hits = [h for h in hits if h.exact_id or h.relevance >= 0.15]   # drop noise passages
        mode = getattr(self.llm, "mode", "custom")
        if not hits or best < self.min_relevance:
            return {"final": {"answer": ABSTAIN[lang], "sources": [], "language": lang, "abstained": True,
                              "confidence": round(best, 3), "mode": mode, "search_query": search_query,
                              "intent": "security", "suggestions": SUGGESTIONS[lang]}}
        llm = self.llm
        if mode != "extractive" and not self.cap.take():       # daily LLM budget used up -> offline answer
            llm, mode = ExtractiveLLM(), "extractive"
        return {"final": None, "question": question, "lang": lang, "history": history, "best": best,
                "search_query": search_query, "sources": self._sources(hits), "llm": llm, "mode": mode}

    def _result(self, p: dict, answer: str, mode: str) -> dict:
        for s in p["sources"][:3]:
            self.store.log_retrieval(s["doc_id"], s["title"])
        return {"answer": answer, "sources": p["sources"], "language": p["lang"], "abstained": False,
                "confidence": round(p["best"], 3), "mode": mode, "search_query": p["search_query"],
                "intent": "security", "suggestions": []}

    def ask(self, question: str, history=None) -> dict:
        p = self._prepare(question, history)
        if p["final"]:
            return p["final"]
        mode = p["mode"]
        try:
            answer = p["llm"].generate(p["question"], p["sources"], p["lang"], p["history"])
        except Exception as e:   # network/quota failure: degrade gracefully, never invent
            print(f"[aman] LLM failed -> {describe_error(e)} | using extractive answer")
            answer, mode = ExtractiveLLM().generate(p["question"], p["sources"], p["lang"], p["history"]), "extractive"
        return self._result(p, answer, mode)

    def ask_stream(self, question: str, history=None):
        """Generator of events: {"type":"delta","text"} ... then {"type":"done", ...full result}.
        If the provider dies mid-answer, a {"type":"replace"} event swaps in a safe extractive answer."""
        p = self._prepare(question, history)
        if p["final"]:
            yield {"type": "delta", "text": p["final"]["answer"]}
            yield {"type": "done", **p["final"]}
            return
        parts, mode = [], p["mode"]
        try:
            for piece in stream_text(p["llm"], p["question"], p["sources"], p["lang"], p["history"]):
                parts.append(piece)
                yield {"type": "delta", "text": piece}
            answer = "".join(parts).strip()
            if not answer:
                raise RuntimeError("empty answer")
        except Exception as e:
            print(f"[aman] LLM stream failed -> {describe_error(e)} | using extractive answer")
            answer = ExtractiveLLM().generate(p["question"], p["sources"], p["lang"], p["history"])
            mode = "extractive"
            yield {"type": "replace" if parts else "delta", "text": answer}
        yield {"type": "done", **self._result(p, answer, mode)}

    # ---- specialised features -------------------------------------------
    def cve(self, cve_id: str) -> dict | None:
        chunks = self.index.get_doc(cve_id)
        if not chunks or chunks[0]["metadata"].get("type") != "cve":
            return None
        c = chunks[0]
        m = c["metadata"]
        text = "\n".join(x["text"] for x in chunks)
        def line(prefix):
            mm = re.search(prefix + r"\s*(.+)", text)
            return mm.group(1).strip() if mm else ""
        desc = c["text"].split("\n", 1)[-1].split("\n\n")[0]
        guidance = [x for k, xs in self.index.by_doc.items() if k.startswith("GUIDE-")
                    for x in [self.index.chunks[i] for i in xs] if c["doc_id"] in x["title"]]
        kev = ({"added": m.get("kev_added", ""), "due": m.get("kev_due", ""), "ransomware": m.get("kev_ransomware", ""),
                "action": m.get("kev_action", "")} if m.get("kev") else None)
        mitigation = ([kev["action"]] if kev and kev["action"] else []) + [g["text"].split("\n", 1)[-1] for g in guidance]
        return {"id": c["doc_id"], "kev": kev, "description": desc, "severity": m.get("severity", ""),
                "cvss_score": m.get("cvss_score"), "cvss_version": m.get("cvss_version", ""),
                "vector": m.get("vector", ""), "published": m.get("published", ""),
                "affected": [p.strip() for p in line("Affected products:").split(";") if p.strip()],
                "references": line("References:").split(),
                "mitigation": mitigation,
                "url": c["url"], "sample": bool(m.get("seed"))}

    def search_cves(self, query: str, limit: int = 8) -> list[dict]:
        hits = hybrid_search(self.index, query + " vulnerability", self.reranker, top_k=25)
        seen, out = set(), []
        for h in hits:
            c = self.index.chunks[h.idx]
            if c["metadata"].get("type") == "cve" and c["doc_id"] not in seen:
                seen.add(c["doc_id"])
                out.append({"id": c["doc_id"], "severity": c["metadata"].get("severity", ""),
                            "cvss_score": c["metadata"].get("cvss_score"),
                            "summary": c["text"].split("\n", 1)[-1][:200]})
        return out[:limit]

    def attack(self, q: str = "", tactic: str = "") -> dict:
        techs = []
        for doc_id, idxs in self.index.by_doc.items():
            c = self.index.chunks[idxs[0]]
            if c["source"] != "MITRE ATT&CK":
                continue
            m = c["metadata"]
            if m.get("type") == "technique" and (not tactic or tactic in m.get("tactics", [])):
                techs.append({"id": doc_id, "name": c["title"].split(" ", 1)[1], "tactics": m.get("tactics", []),
                              "sub": m.get("is_subtechnique", False), "url": c["url"],
                              "text": c["text"].split("\n", 1)[-1]})
        if q:
            ql = q.lower()
            techs = [t for t in techs if ql in (t["id"] + t["name"] + t["text"]).lower()]
        tactics = sorted({t for tt in self.attack_all() for t in tt["tactics"]})
        return {"techniques": sorted(techs, key=lambda t: t["id"]), "tactics": tactics}

    def attack_all(self):
        return [{"tactics": self.index.chunks[i[0]]["metadata"].get("tactics", [])}
                for k, i in self.index.by_doc.items() if self.index.chunks[i[0]]["metadata"].get("type") == "technique"]

    def dashboard(self) -> dict:
        by_source, sev, tactics, cves = {}, {}, {}, []
        for doc_id, idxs in self.index.by_doc.items():
            c = self.index.chunks[idxs[0]]
            by_source[c["source"]] = by_source.get(c["source"], 0) + 1
            m = c["metadata"]
            if m.get("type") == "cve":
                sev[m.get("severity") or "UNKNOWN"] = sev.get(m.get("severity") or "UNKNOWN", 0) + 1
                cves.append({"id": doc_id, "severity": m.get("severity", ""), "score": m.get("cvss_score"),
                             "published": m.get("published", "")})
            for t in m.get("tactics", []) if m.get("type") == "technique" else []:
                tactics[t] = tactics.get(t, 0) + 1
        cves.sort(key=lambda x: x["published"], reverse=True)
        return {"documents": len(self.index.by_doc), "chunks": len(self.index.chunks), "by_source": by_source,
                "severity": sev, "tactics": dict(sorted(tactics.items(), key=lambda x: -x[1])),
                "latest_cves": cves[:8], "top_topics": self.store.top_topics(), "feedback": self.store.feedback_stats(),
                "kev_count": sum(1 for k, i in self.index.by_doc.items() if self.index.chunks[i[0]]["metadata"].get("kev")),
                "embedder": self.index.embedder.name, "reranker": self.reranker.name,
                "llm": getattr(self.llm, "mode", "custom"), "sample_data": any(
                    c["metadata"].get("seed") for c in self.index.chunks)}
