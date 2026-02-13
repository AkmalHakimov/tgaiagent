from __future__ import annotations

import re
from dataclasses import dataclass

HIGH_RISK_PATTERNS = [
    re.compile(r"\b(?:password|otp|2fa|seed phrase|private key)\b", re.IGNORECASE),
    re.compile(r"\b(?:hack|ddos|malware|exploit)\b", re.IGNORECASE),
]

QUESTION_HINTS = [
    "?",
    "how ",
    "what ",
    "why ",
    "can you",
    "could you",
    "please help",
    "explain",
]


@dataclass(slots=True)
class PolicyDecision:
    allowed: bool
    should_reply: bool
    reason: str


def is_high_risk(text: str) -> bool:
    return any(p.search(text) for p in HIGH_RISK_PATTERNS)


def looks_like_question(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    return any(hint in normalized for hint in QUESTION_HINTS)


def enforce_policy(text: str, max_chars: int) -> PolicyDecision:
    if not text.strip():
        return PolicyDecision(allowed=False, should_reply=False, reason="empty_message")

    if len(text) > 5000:
        return PolicyDecision(allowed=False, should_reply=False, reason="message_too_long")

    if is_high_risk(text):
        return PolicyDecision(allowed=False, should_reply=False, reason="high_risk_content")

    if not looks_like_question(text):
        return PolicyDecision(allowed=True, should_reply=False, reason="not_a_question")

    if max_chars < 100:
        return PolicyDecision(allowed=False, should_reply=False, reason="invalid_max_reply_chars")

    return PolicyDecision(allowed=True, should_reply=True, reason="ok")


def clip_reply(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
