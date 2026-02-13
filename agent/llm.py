from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=6),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = await self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        payload = response.model_dump()
        try:
            outputs = payload.get("output", [])
            chunks: list[str] = []
            for out in outputs:
                for item in out.get("content", []):
                    if item.get("type") == "output_text":
                        chunks.append(item.get("text", ""))
            if chunks:
                return "\n".join(chunks).strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("response parse fallback failed: %s", exc)

        raise RuntimeError("LLM did not return text")

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        raw = await self.generate_text(system_prompt, user_prompt, temperature=0.0)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                snippet = raw[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
        logger.warning("failed to parse JSON from planner output; using fallback")
        return fallback
