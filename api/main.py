
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.triage import run_triage
from api.logging_db import log_request, init_db

app = FastAPI(title="Support Ticket Triage API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


class TicketRequest(BaseModel):
    text: str


class TriageResponse(BaseModel):
    category: str
    urgency: str
    draft_response: str
    retrieved_sources: list[str]
    latency_ms: float
    error: str | None = None


@app.post("/triage", response_model=TriageResponse)
def triage_ticket(request: TicketRequest):
    result = run_triage(request.text)
    log_request(request.text, result)
    return TriageResponse(
        category=result.category,
        urgency=result.urgency,
        draft_response=result.draft_response,
        retrieved_sources=result.retrieved_sources,
        latency_ms=result.latency_ms,
        error=result.error,
    )


@app.get("/health")
def health():
    return {"status": "ok"}