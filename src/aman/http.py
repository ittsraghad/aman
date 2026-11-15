"""Tiny JSON-over-HTTPS helper (standard library) with retry on rate limits."""
from __future__ import annotations

import io
import json
import re
import time
import urllib.error
import urllib.request


_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')


class QuotaExhausted(RuntimeError):
    """The daily quota is used up: waiting a few seconds will not help."""


def post_json(url: str, body: dict, headers: dict | None = None, timeout: int = 60, retries: int = 0) -> dict:
    data = json.dumps(body).encode()
    h = {"content-type": "application/json", **(headers or {})}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data, h), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or retries == 0:
                raise
            try:
                txt = e.read().decode("utf-8", "ignore")
            except Exception:
                txt = ""
            m = _RETRY_DELAY.search(txt)
            delay = float(m.group(1)) + 1 if m else min(2 ** attempt * 2, 30)    # obey the server's own advice
            if e.code == 429 and (delay > 150 or (not m and ("PerDay" in txt or "per day" in txt.lower()))):
                raise QuotaExhausted("daily quota reached: " + " ".join(txt.split())[:1500]) from None
            if attempt >= retries:
                raise urllib.error.HTTPError(url, e.code, e.reason, e.headers, io.BytesIO(txt.encode())) from None
            print(f"[aman] API said HTTP {e.code}: {' '.join(txt.split())[:220]!r}\n"
                  f"[aman] waiting {delay:.0f}s, then retrying ({attempt + 1}/{retries}) - this is normal on free quotas",
                  flush=True)
            time.sleep(min(delay, 150))


def post_sse(url: str, body: dict, headers: dict | None = None, timeout: int = 90):
    """POST and yield each Server-Sent-Event JSON payload as it arrives (used for streamed LLM answers)."""
    data = json.dumps(body).encode()
    h = {"content-type": "application/json", **(headers or {})}
    with urllib.request.urlopen(urllib.request.Request(url, data, h), timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        yield json.loads(payload)
                    except ValueError:
                        continue


def describe_error(e: Exception) -> str:
    """Human-readable error for logs: HTTP status + the provider's message (never includes request headers/keys)."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", "ignore")[:300].replace("\n", " ")
        except Exception:
            body = ""
        return f"HTTP {e.code} {e.reason}: {body}"
    return f"{type(e).__name__}: {e}"
