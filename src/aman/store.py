"""SQLite for anonymous usage statistics only. NO user questions or answers are stored on the server:
chat history lives in each visitor's browser. (On free hosts the disk is ephemeral, so stats reset on restart.)"""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS retrievals(id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, title TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS feedback(id INTEGER PRIMARY KEY AUTOINCREMENT, rating TEXT, intent TEXT, mode TEXT,
    doc_ids TEXT, ts REAL);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._keep = None
        if self.path == ":memory:":
            self._keep = sqlite3.connect("file:aman_mem?mode=memory&cache=shared", uri=True, check_same_thread=False)
            self.path = "file:aman_mem?mode=memory&cache=shared"
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c:
            c.executescript(SCHEMA)

    def _conn(self):
        return sqlite3.connect(self.path, uri=self.path.startswith("file:"), timeout=10)

    def _run(self, sql, args=(), fetch=False):
        with closing(self._conn()) as c:
            cur = c.execute(sql, args)
            rows = cur.fetchall() if fetch else None
            c.commit()
            return rows

    def log_retrieval(self, doc_id: str, title: str):
        self._run("INSERT INTO retrievals(doc_id,title,ts) VALUES(?,?,?)", (doc_id, title, time.time()))

    def top_topics(self, limit: int = 8) -> list[dict]:
        rows = self._run("SELECT doc_id, title, COUNT(*) n FROM retrievals GROUP BY doc_id ORDER BY n DESC LIMIT ?",
                         (limit,), True)
        return [{"doc_id": d, "title": t, "count": n} for d, t, n in rows]

    def add_feedback(self, rating: str, intent: str = "", mode: str = "", doc_ids: list[str] | None = None):
        self._run("INSERT INTO feedback(rating,intent,mode,doc_ids,ts) VALUES(?,?,?,?,?)",
                  (rating, intent[:20], mode[:20], ",".join((doc_ids or [])[:10]), time.time()))

    def feedback_stats(self) -> dict:
        rows = dict(self._run("SELECT rating, COUNT(*) FROM feedback GROUP BY rating", fetch=True))
        return {"up": rows.get("up", 0), "down": rows.get("down", 0)}
