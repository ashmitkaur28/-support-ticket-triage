"""
Core pipeline: guardrail check -> classify -> retrieve -> draft response.
Now powered by Groq's free API instead of a paid provider.

Usage:
    python -m core.triage --ticket "My order never arrived and support won't reply"
"""
import argparse
import json
import time
from dataclasses import dataclass, field

from groq import Groq
from dotenv import load_dotenv

from core import guardrails
from core.retrieval import retrieve

load_dotenv()
client = Groq()  # reads GROQ_API_KEY from .env automatically

CLASSIFY_MODEL = "openai/gpt-oss-20b"     # fast/cheap model for classification
DRAFT_MODEL = "openai/gpt-oss-120b"     # stronger model for response drafting

CATEGORIES = ["Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance", "Sales and Pre-Sales", "Human Resources", "General Inquiry"]
URGENCIES = ["low", "medium", "high"]


@dataclass
class TriageResult:
    category: str
    urgency: str
    draft_response: str
    retrieved_sources: list = field(default_factory=list)
    classify_tokens: dict = field(default_factory=dict)
    draft_tokens: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None
    guardrail_rule: str | None = None


def classify(ticket_text: str) -> tuple[str, str, dict]:
    prompt = f"""Classify this support ticket. Respond ONLY with JSON, no
markdown fences, no preamble. Format:
{{"category": "<one of {CATEGORIES}>", "urgency": "<one of {URGENCIES}>"}}

Ticket:
{ticket_text}"""

    resp = client.chat.completions.create(
        model=CLASSIFY_MODEL,
        max_tokens=500,
        temperature=0.2,
        reasoning_effort="medium",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    if not text or not text.strip():
        raise ValueError("Model returned empty content.")
    text = text.strip().removeprefix("```json").removesuffix("```").strip()
    parsed = json.loads(text)
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    return parsed["category"], parsed["urgency"], usage

def draft_response(ticket_text: str, category: str, context_chunks: list[str]) -> tuple[str, dict]:
    context_block = "\n\n".join(context_chunks) if context_chunks else "No relevant policy found."

    prompt = f"""You are a support agent drafting a reply to a customer.
Use ONLY the policy context below to answer — don't invent policy details
that aren't there. Be warm but concise. Do not promise anything the policy
doesn't cover.

Category: {category}

Relevant policy context:
{context_block}

Customer ticket:
{ticket_text}

Write the reply now."""

    resp = client.chat.completions.create(
        model=DRAFT_MODEL,
        max_tokens=800,
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content
    if not text or not text.strip():
        raise ValueError("Model returned empty content (likely ran out of tokens on reasoning).")
    text = text.strip()
    usage = {
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    return text, usage


def run_triage(ticket_text: str) -> TriageResult:
    start = time.time()
    try:
        guardrails.check_input(ticket_text)

        category, urgency, classify_usage = classify(ticket_text)

        hits = retrieve(ticket_text, k=3)
        context_chunks = [h["text"] for h in hits]
        sources = list({h["source"] for h in hits})

        response_text, draft_usage = draft_response(ticket_text, category, context_chunks)

        guardrails.check_output(response_text)

        return TriageResult(
            category=category,
            urgency=urgency,
            draft_response=response_text,
            retrieved_sources=sources,
            classify_tokens=classify_usage,
            draft_tokens=draft_usage,
            latency_ms=(time.time() - start) * 1000,
        )
    except guardrails.GuardrailViolation as e:
        return TriageResult(
            category="", urgency="", draft_response="",
            error=f"Guardrail violation: {e}",
            guardrail_rule=e.rule,
            latency_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return TriageResult(
            category="", urgency="", draft_response="",
            error=f"Unexpected error: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket", type=str, required=True)
    args = parser.parse_args()

    result = run_triage(args.ticket)
    if result.error:
        print(f"ERROR: {result.error}")
        if result.guardrail_rule:
            print(f"Guardrail rule triggered: {result.guardrail_rule}")
    else:
        print(f"Category: {result.category}")
        print(f"Urgency: {result.urgency}")
        print(f"Sources used: {result.retrieved_sources}")
        print(f"Latency: {result.latency_ms:.0f}ms")
        print(f"\nDraft response:\n{result.draft_response}")