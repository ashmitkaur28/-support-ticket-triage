"""
Simple SQLite logging for every triage request. This is what powers your
observability dashboard and your resume's cost/latency numbers.
"""
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "logs.db"

# Groq pricing as of writing — check console.groq.com/docs/models for
# current numbers before reporting these anywhere, pricing changes.
# $/million tokens.
PRICING = {
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            ticket_text TEXT,
            category TEXT,
            urgency TEXT,
            success INTEGER,
            error TEXT,
            latency_ms REAL,
            classify_input_tokens INTEGER,
            classify_output_tokens INTEGER,
            draft_input_tokens INTEGER,
            draft_output_tokens INTEGER,
            cost_usd REAL
        )
    """)
    conn.commit()
    conn.close()


def compute_cost(classify_tokens: dict, draft_tokens: dict) -> float:
    cost = 0.0
    if classify_tokens:
        p = PRICING["openai/gpt-oss-20b"]
        cost += classify_tokens.get("input_tokens", 0) / 1e6 * p["input"]
        cost += classify_tokens.get("output_tokens", 0) / 1e6 * p["output"]
    if draft_tokens:
        p = PRICING["openai/gpt-oss-120b"]
        cost += draft_tokens.get("input_tokens", 0) / 1e6 * p["input"]
        cost += draft_tokens.get("output_tokens", 0) / 1e6 * p["output"]
    return cost


def log_request(ticket_text: str, result) -> None:
    init_db()
    cost = compute_cost(result.classify_tokens, result.draft_tokens)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO requests (
            timestamp, ticket_text, category, urgency, success, error,
            latency_ms, classify_input_tokens, classify_output_tokens,
            draft_input_tokens, draft_output_tokens, cost_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            time.time(),
            ticket_text,
            result.category,
            result.urgency,
            0 if result.error else 1,
            result.error,
            result.latency_ms,
            result.classify_tokens.get("input_tokens", 0),
            result.classify_tokens.get("output_tokens", 0),
            result.draft_tokens.get("input_tokens", 0),
            result.draft_tokens.get("output_tokens", 0),
            cost,
        ),
    )
    conn.commit()
    conn.close()