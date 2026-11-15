"""Protection for a public demo: per-client rate limit and a global daily cap on LLM calls."""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import date


class RateLimiter:
    """Sliding window: at most `limit` hits per `window` seconds per key."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        now = time.time() if now is None else now
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False, max(1, int(self.window - (now - q[0])) + 1)
            q.append(now)
            if len(self.hits) > 5000:                       # bound memory
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] >= self.window]:
                    del self.hits[k]
            return True, 0


class DailyCap:
    """At most `cap` LLM calls per calendar day (protects free-tier quota / your wallet)."""

    def __init__(self, cap: int):
        self.cap, self.day, self.n = cap, None, 0
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            today = date.today()
            if self.day != today:
                self.day, self.n = today, 0
            if self.n >= self.cap:
                return False
            self.n += 1
            return True
