
"""HTTP server: JSON API + the web UI (standard library only).

Run:  PYTHONPATH=src python -m aman.server -> http://127.0.0.1:8000
Env : GEMINI_API_KEY | ANTHROPIC_API_KEY, AMAN_LLM, AMAN_EMBEDDER, AMAN_NO_BUILD, AMAN_CHUNKS, AMAN_DB,
      AMAN_RATE_PER_MIN, AMAN_DAILY_LLM_CAP, AMAN_TRUST_PROXY, AMAN_PROXY_HOPS, PORT,
      AMAN_AUTHOR / AMAN_LINKEDIN / AMAN_GITHUB (shown on the About page)
"""
from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from aman.build_corpus import build
from aman.embeddings import make_embedder
from aman.index import KnowledgeIndex
from aman.llm import make_llm
from aman.rag import AmanService
from aman.ratelimit import RateLimiter
from aman.search import make_reranker
from aman.store import Store

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"

CSP = (
    "default-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)


def create_service(chunks: Path | None = None, db: str | None = None, embedder=None, reranker=None, llm=None):
    chunks = Path(chunks or os.getenv("AMAN_CHUNKS", ROOT / "data/processed/chunks.jsonl"))
    if not chunks.exists():
        print("[aman] no corpus found -> building the built-in sample knowledge base")
        build(Path("/nonexistent"), chunks, seed=True)

    index = KnowledgeIndex(
        chunks,
        embedder or make_embedder(),
        chunks.parent,
        build_vectors=os.getenv("AMAN_NO_BUILD") != "1"
    )

    store = Store(db or os.getenv("AMAN_DB", ROOT / "data/aman.db"))
    return AmanService(index, reranker or make_reranker(), llm or make_llm(), store)


def make_handler(
    svc: AmanService,
    chat_limiter: RateLimiter | None = None,
    api_limiter: RateLimiter | None = None
):
    chat_limiter = chat_limiter or RateLimiter(int(os.getenv("AMAN_RATE_PER_MIN", "12")))
    api_limiter = api_limiter or RateLimiter(120)
    trust_proxy = os.getenv("AMAN_TRUST_PROXY") == "1"
    hops = int(os.getenv("AMAN_PROXY_HOPS", "1"))

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *a):
            pass

        def _ip(self) -> str:
            if trust_proxy:
                parts = [
                    p.strip()
                    for p in self.headers.get("X-Forwarded-For", "").split(",")
                    if p.strip()
                ]
                if len(parts) >= hops:
                    return parts[-hops]

            return self.client_address[0]

        def _send(
            self,
            code: int,
            payload,
            ctype="application/json; charset=utf-8",
            extra: dict | None = None
        ):
            body = (
                payload
                if isinstance(payload, bytes)
                else json.dumps(payload, ensure_ascii=False).encode()
            )

            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")

            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", CSP)
            elif ctype.startswith("application/json"):
                self.send_header("Cache-Control", "no-store")

            for k, v in (extra or {}).items():
                self.send_header(k, v)

            self.end_headers()
            self.wfile.write(body)

        def _limited(self, limiter: RateLimiter) -> bool:
            ok, retry = limiter.check(self._ip())

            if not ok:
                self._send(
                    429,
                    {"error": "too many requests", "retry_after": retry},
                    extra={"Retry-After": str(retry)}
                )

            return not ok

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)

            if n > 30_000:
                raise ValueError("request too large")

            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            path = u.path

            try:
                if path in ("/", "/index.html"):
                    return self._send(
                        200,
                        (WEB / "index.html").read_bytes(),
                        "text/html; charset=utf-8"
                    )

                # Open Graph preview image for LinkedIn and other platforms
                if path == "/og-image.jpg":
                    return self._send(
                        200,
                        (WEB / "og-image-cropped.jpg").read_bytes(),
                        "image/jpeg",
                        {"Cache-Control": "public, max-age=86400"}
                    )

                if path == "/logo.svg":
                    return self._send(
                        200,
                        (WEB / "logo.svg").read_bytes(),
                        "image/svg+xml",
                        {"Cache-Control": "public, max-age=86400"}
                    )

                if path == "/api/health":
                    return self._send(200, {"status": "ok"})

                if self._limited(api_limiter):
                    return

                if path == "/api/config":
                    return self._send(
                        200,
                        {
                            "author": os.getenv("AMAN_AUTHOR", ""),
                            "linkedin": os.getenv("AMAN_LINKEDIN", ""),
                            "github": os.getenv("AMAN_GITHUB", "")
                        }
                    )

                if path == "/api/dashboard":
                    return self._send(200, svc.dashboard())

                m = re.fullmatch(r"/api/cve/(CVE-\d{4}-\d{4,})", path, re.I)

                if m:
                    r = svc.cve(m.group(1).upper())
                    return self._send(
                        200 if r else 404,
                        r or {"error": "CVE not found in the knowledge base"}
                    )

                if path == "/api/cve":
                    return self._send(200, svc.search_cves(q.get("q", "")[:200]))

                if path == "/api/attack":
                    return self._send(
                        200,
                        svc.attack(q.get("q", "")[:100], q.get("tactic", "")[:60])
                    )

                if path == "/api/source":
                    cid = unquote(q.get("chunk_id", ""))

                    for c in svc.index.chunks:
                        if c["chunk_id"] == cid:
                            return self._send(200, c)

                    return self._send(404, {"error": "source not found"})

                return self._send(404, {"error": "not found"})

            except Exception as e:
                print(f"[aman] GET {path} failed: {type(e).__name__}: {e}")
                return self._send(500, {"error": "internal error"})

        def do_POST(self):
            path = urlparse(self.path).path

            try:
                if path == "/api/chat":
                    if self._limited(chat_limiter):
                        return

                    b = self._body()
                    q = str(b.get("question", ""))[:1000]

                    if not q.strip():
                        return self._send(400, {"error": "question is required"})

                    return self._send(200, svc.ask(q, b.get("history")))

                if path == "/api/chat/stream":
                    if self._limited(chat_limiter):
                        return

                    b = self._body()
                    q = str(b.get("question", ""))[:1000]

                    if not q.strip():
                        return self._send(400, {"error": "question is required"})

                    self.send_response(200)

                    for k, v in (
                        ("Content-Type", "application/x-ndjson; charset=utf-8"),
                        ("Cache-Control", "no-cache, no-store"),
                        ("X-Accel-Buffering", "no"),
                        ("X-Content-Type-Options", "nosniff")
                    ):
                        self.send_header(k, v)

                    self.end_headers()
                    self.close_connection = True

                    try:
                        for ev in svc.ask_stream(q, b.get("history")):
                            self.wfile.write(
                                (json.dumps(ev, ensure_ascii=False) + "\n").encode()
                            )
                            self.wfile.flush()

                    except (BrokenPipeError, ConnectionResetError):
                        pass

                    except Exception as e:
                        print(f"[aman] stream failed: {type(e).__name__}: {e}")

                        try:
                            self.wfile.write(b'{"type":"error"}\n')
                        except OSError:
                            pass

                    return

                if path == "/api/feedback":
                    if self._limited(api_limiter):
                        return

                    b = self._body()

                    if b.get("rating") not in ("up", "down"):
                        return self._send(
                            400,
                            {"error": "rating must be up or down"}
                        )

                    ids = (
                        [str(x)[:40] for x in b.get("doc_ids", [])][:10]
                        if isinstance(b.get("doc_ids"), list)
                        else []
                    )

                    svc.store.add_feedback(
                        b["rating"],
                        str(b.get("intent", "")),
                        str(b.get("mode", "")),
                        ids
                    )

                    return self._send(200, {"ok": True})

                return self._send(404, {"error": "not found"})

            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})

            except Exception as e:
                print(f"[aman] POST {path} failed: {type(e).__name__}: {e}")
                return self._send(500, {"error": "internal error"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))

    a = ap.parse_args()
    svc = create_service()
    d = svc.dashboard()

    print(
        f"[aman] {d['documents']} documents / {d['chunks']} chunks | "
        f"embedder={d['embedder']} reranker={d['reranker']} llm={d['llm']}"
    )

    print(f"[aman] open http://{a.host}:{a.port}")

    ThreadingHTTPServer(
        (a.host, a.port),
        make_handler(svc)
    ).serve_forever()


if __name__ == "__main__":
    main()
