"""
Generates a small set of synthetic support tickets for testing/eval.
Run once to create data/tickets.json.

Usage: python data/generate_synthetic_tickets.py
"""
import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

CATEGORIES = ["billing", "shipping", "technical_issue", "account_access", "product_question"]
URGENCIES = ["low", "medium", "high"]

PROMPT = """Generate 20 realistic customer support tickets for a mid-size
e-commerce company. Vary them across these categories: {categories}
and urgency levels: {urgencies}.

Make some short and casual, some longer and frustrated, some calm and
polite. Include realistic details (order numbers, product names, dates)
but keep them fictional.

Respond ONLY with a JSON array, no markdown fences, no preamble. Each
object must have exactly these fields:
- "id": a string like "t001"
- "text": the ticket text as the customer wrote it
- "true_category": one of {categories}
- "true_urgency": one of {urgencies}
"""


def generate_batch(n_batches: int = 3):
    all_tickets = []
    for i in range(n_batches):
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": PROMPT.format(categories=CATEGORIES, urgencies=URGENCIES)
            }]
        )
        text = msg.content[0].text.strip()
        text = text.removeprefix("```json").removesuffix("```").strip()
        batch = json.loads(text)
        # re-number ids to avoid collisions across batches
        for j, t in enumerate(batch):
            t["id"] = f"t{i:02d}{j:02d}"
        all_tickets.extend(batch)
        print(f"Batch {i+1}/{n_batches}: generated {len(batch)} tickets")
    return all_tickets


if __name__ == "__main__":
    tickets = generate_batch()
    out_path = Path(__file__).parent / "tickets.json"
    out_path.write_text(json.dumps(tickets, indent=2))
    print(f"Saved {len(tickets)} tickets to {out_path}")
