from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class IncomingMessage:
    message_id: int
    chat_id: int
    user_id: int
    sender_name: str
    text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class PlannedAction:
    should_reply: bool
    intent: str
    confidence: float
    reply_style: str
    tool_calls: list[dict[str, Any]]
    rationale: str


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    output: str
