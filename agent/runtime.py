from __future__ import annotations

import asyncio
import logging
import re

from .llm import LLMClient
from .memory import MemoryStore
from .planner import Planner
from .policy import clip_reply, enforce_policy
from .prompts import (
    build_response_system_prompt,
    build_response_user_prompt,
)
from .tools import ToolRegistry
from .types import IncomingMessage, ToolResult

logger = logging.getLogger(__name__)

NAME_RE = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\- ]{1,40})\b", re.IGNORECASE)
CITY_RE = re.compile(r"\bi live in\s+([A-Za-z][A-Za-z\- ]{1,40})\b", re.IGNORECASE)


class AgentRuntime:
    def __init__(
        self,
        *,
        memory: MemoryStore,
        llm: LLMClient,
        planner: Planner,
        tools: ToolRegistry,
        agent_name: str,
        max_context_messages: int,
        max_reply_chars: int,
    ) -> None:
        self._memory = memory
        self._llm = llm
        self._planner = planner
        self._tools = tools
        self._agent_name = agent_name
        self._max_context_messages = max_context_messages
        self._max_reply_chars = max_reply_chars
        self._queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(maxsize=200)

    async def enqueue(self, incoming: IncomingMessage) -> None:
        try:
            self._queue.put_nowait(incoming)
        except asyncio.QueueFull:
            logger.warning("queue full, dropping message %s", incoming.message_id)

    async def worker(self, send_reply_cb) -> None:
        while True:
            incoming = await self._queue.get()
            try:
                await self._process_one(incoming, send_reply_cb)
            except Exception:
                logger.exception("failed processing message id=%s", incoming.message_id)
            finally:
                self._queue.task_done()

    async def _process_one(self, incoming: IncomingMessage, send_reply_cb) -> None:
        if await self._memory.is_processed(incoming.chat_id, incoming.message_id):
            return

        await self._memory.mark_processed(incoming.chat_id, incoming.message_id)
        await self._memory.add_message(
            chat_id=incoming.chat_id,
            user_id=incoming.user_id,
            role="user",
            text=incoming.text,
            meta={"sender_name": incoming.sender_name, "message_id": incoming.message_id},
        )
        await self._extract_profile_facts(incoming)

        policy = enforce_policy(incoming.text, max_chars=self._max_reply_chars)
        if not policy.allowed:
            if policy.reason == "high_risk_content":
                await send_reply_cb(
                    incoming.chat_id,
                    "I can't help with requests involving hacking, stolen credentials, or harmful actions.",
                )
            return

        if not policy.should_reply:
            return

        recent = await self._memory.get_recent_messages(incoming.chat_id, self._max_context_messages)
        context_lines = [f"{m.role}: {m.text}" for m in recent]
        facts = await self._memory.get_profile_facts(incoming.user_id, limit=8)

        plan = await self._planner.plan(
            sender_name=incoming.sender_name,
            text=incoming.text,
            context_lines=context_lines,
        )
        if not plan.should_reply or plan.confidence < 0.25:
            return

        tool_results = await self._run_tools(plan.tool_calls, incoming.user_id)
        tool_output_lines = [f"{r.name}: {r.output}" for r in tool_results]

        response = await self._llm.generate_text(
            system_prompt=build_response_system_prompt(self._agent_name),
            user_prompt=build_response_user_prompt(
                message_text=incoming.text,
                context_lines=context_lines,
                tool_outputs=tool_output_lines,
                profile_facts=facts,
                intent=plan.intent,
                style=plan.reply_style,
            ),
            temperature=0.3,
        )
        response = clip_reply(response.strip(), self._max_reply_chars)
        if not response:
            return

        await send_reply_cb(incoming.chat_id, response)
        await self._memory.add_message(
            chat_id=incoming.chat_id,
            user_id=incoming.user_id,
            role="assistant",
            text=response,
            meta={
                "intent": plan.intent,
                "confidence": plan.confidence,
                "rationale": plan.rationale,
                "tools": [r.name for r in tool_results],
            },
        )

    async def _run_tools(self, tool_calls: list[dict], user_id: int) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in tool_calls:
            args = dict(call.get("args", {}))
            if call.get("name") == "recall_user_profile":
                args["user_id"] = user_id
            result = await self._tools.execute(str(call.get("name", "")), args)
            results.append(result)
        return results

    async def _extract_profile_facts(self, incoming: IncomingMessage) -> None:
        text = incoming.text.strip()
        if not text:
            return
        name_match = NAME_RE.search(text)
        if name_match:
            await self._memory.add_profile_fact(
                incoming.user_id,
                "name",
                name_match.group(1).strip(),
                confidence=0.85,
            )
        city_match = CITY_RE.search(text)
        if city_match:
            await self._memory.add_profile_fact(
                incoming.user_id,
                "city",
                city_match.group(1).strip(),
                confidence=0.80,
            )
