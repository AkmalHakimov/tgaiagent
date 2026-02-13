from __future__ import annotations

from .llm import LLMClient
from .prompts import PLANNER_SYSTEM_PROMPT, build_planner_user_prompt
from .types import PlannedAction


class Planner:
    def __init__(self, llm: LLMClient, allowed_tools: set[str], agent_name: str) -> None:
        self._llm = llm
        self._allowed_tools = allowed_tools
        self._agent_name = agent_name

    async def plan(self, sender_name: str, text: str, context_lines: list[str]) -> PlannedAction:
        fallback = {
            "should_reply": True,
            "intent": "general_question",
            "confidence": 0.40,
            "reply_style": "clear_direct",
            "tool_calls": [],
            "rationale": "fallback planner path",
        }
        payload = await self._llm.generate_json(
            PLANNER_SYSTEM_PROMPT,
            build_planner_user_prompt(
                agent_name=self._agent_name,
                sender_name=sender_name,
                text=text,
                context_lines=context_lines,
            ),
            fallback=fallback,
        )

        tool_calls = []
        for call in payload.get("tool_calls", []):
            name = str(call.get("name", "")).strip()
            args = call.get("args", {})
            if name in self._allowed_tools and isinstance(args, dict):
                tool_calls.append({"name": name, "args": args})

        return PlannedAction(
            should_reply=bool(payload.get("should_reply", True)),
            intent=str(payload.get("intent", "general_question")),
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.5)))),
            reply_style=str(payload.get("reply_style", "clear_direct")),
            tool_calls=tool_calls,
            rationale=str(payload.get("rationale", "")),
        )
