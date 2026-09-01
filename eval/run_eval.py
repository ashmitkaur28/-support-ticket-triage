
import json
from pathlib import Path

from api.logging_db import compute_cost
from core.triage import run_triage

EVAL_PATH = Path(__file__).parent / "eval_set.json"


def run_eval():
    tickets = json.loads(EVAL_PATH.read_text())

    correct_category = 0
    correct_urgency = 0
    correct_both = 0
    total_cost = 0.0
    total_latency = 0.0
    errors = 0

    print(f"Running eval on {len(tickets)} tickets...\n")

    for t in tickets:
        result = run_triage(t["text"])

        if result.error:
            errors += 1
            print(f"[{t['id']}] ERROR: {result.error}")
            continue

        cat_match = result.category == t["true_category"]
        urg_match = result.urgency == t["true_urgency"]
        correct_category += cat_match
        correct_urgency += urg_match
        correct_both += cat_match and urg_match
        total_cost += compute_cost(result.classify_tokens, result.draft_tokens)
        total_latency += result.latency_ms

        status = "✓" if cat_match and urg_match else "✗"
        print(
            f"[{t['id']}] {status} predicted=({result.category}, {result.urgency}) "
            f"true=({t['true_category']}, {t['true_urgency']})"
        )

    n = len(tickets) - errors
    if n == 0:
        print("\nAll requests errored — nothing to report.")
        return

    print("\n--- Eval Results ---")
    print(f"Tickets evaluated: {n} (errors: {errors})")
    print(f"Category accuracy: {correct_category / n * 100:.1f}%")
    print(f"Urgency accuracy:  {correct_urgency / n * 100:.1f}%")
    print(f"Both correct:      {correct_both / n * 100:.1f}%")
    print(f"Avg latency:       {total_latency / n:.0f} ms")
    print(f"Avg cost/ticket:   ${total_cost / n:.5f}")
    print(f"Total eval cost:   ${total_cost:.4f}")


if __name__ == "__main__":
    run_eval()
