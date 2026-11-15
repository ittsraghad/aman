"""OPTIONAL FastAPI front for AmanService (NOT exercised by the test-suite; install requirements-full.txt).
Run: PYTHONPATH=src uvicorn aman.api_fastapi:app --port 8000
The default stdlib server (aman.server) exposes the same endpoints and is the tested one."""
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from aman.server import WEB, create_service

svc = create_service()
app = FastAPI(title="AMAN")


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[dict] | None = None


class FeedbackIn(BaseModel):
    rating: str = Field(pattern="^(up|down)$")
    intent: str = ""
    mode: str = ""
    doc_ids: list[str] = []


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.post("/api/chat")
def chat(body: ChatIn):
    return svc.ask(body.question, body.history)


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    svc.store.add_feedback(body.rating, body.intent, body.mode, body.doc_ids)
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard():
    return svc.dashboard()


@app.get("/api/cve/{cve_id}")
def cve(cve_id: str):
    r = svc.cve(cve_id.upper())
    if not r:
        raise HTTPException(404, "CVE not found in the knowledge base")
    return r


@app.get("/api/cve")
def cve_search(q: str = ""):
    return svc.search_cves(q)


@app.get("/api/attack")
def attack(q: str = "", tactic: str = ""):
    return svc.attack(q, tactic)
