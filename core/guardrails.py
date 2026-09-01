

MAX_TICKET_LENGTH = 4000
MIN_TICKET_LENGTH = 3

INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all prior instructions",
    "ignore the above",
    "disregard the above",
    "disregard previous",
    "you are now",
    "new instructions:",
    "system prompt",
    "system:",
    "act as",
    "pretend you are",
    "reveal your instructions",
    "print your prompt",
]


class GuardrailViolation(Exception):
    def __init__(self, message: str, rule: str):
        super().__init__(message)
        self.rule = rule  


def check_input(ticket_text: str) -> None:
    if not ticket_text or len(ticket_text.strip()) < MIN_TICKET_LENGTH:
        raise GuardrailViolation("Ticket text is empty or too short.", rule="min_length")

    if len(ticket_text) > MAX_TICKET_LENGTH:
        raise GuardrailViolation(
            f"Ticket exceeds {MAX_TICKET_LENGTH} characters.", rule="max_length"
        )

    lowered = ticket_text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lowered:
            raise GuardrailViolation(
                f"Potential prompt-injection pattern detected: '{pattern}'",
                rule="injection_pattern",
            )

    
    for ch in set(ticket_text):
        if ticket_text.count(ch) > 200 and ch.isalnum():
            raise GuardrailViolation("Ticket appears to be spam/flood content.", rule="flood")


def check_output(response_text: str) -> None:
    if not response_text or len(response_text.strip()) < 5:
        raise GuardrailViolation("Generated response is empty or too short.", rule="empty_output")

    lowered = response_text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in lowered:
            raise GuardrailViolation(
                "Output appears to echo an injection pattern — the model may "
                "have been manipulated despite input filtering.",
                rule="output_echo",
            )