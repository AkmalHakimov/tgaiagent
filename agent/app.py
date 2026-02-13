from __future__ import annotations

import asyncio
import logging

from .config import get_settings
from .llm import LLMClient
from .logging_setup import setup_logging
from .memory import MemoryStore
from .planner import Planner
from .runtime import AgentRuntime
from .telegram_gateway import TelegramGateway
from .tools import ToolRegistry

logger = logging.getLogger(__name__)


async def _run() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    memory = MemoryStore(settings.db_path)
    await memory.init()

    llm = LLMClient(api_key=settings.openai_api_key, model=settings.openai_model)
    tools = ToolRegistry(memory)
    planner = Planner(llm=llm, allowed_tools=tools.allowed_tool_names, agent_name=settings.agent_name)
    runtime = AgentRuntime(
        memory=memory,
        llm=llm,
        planner=planner,
        tools=tools,
        agent_name=settings.agent_name,
        max_context_messages=settings.max_context_messages,
        max_reply_chars=settings.max_reply_chars,
    )

    gateway = TelegramGateway(
        api_id=settings.tg_api_id,
        api_hash=settings.tg_api_hash,
        session_name=settings.session_name,
    )
    gateway.register_handler(runtime.enqueue)

    await gateway.start()

    workers = [asyncio.create_task(runtime.worker(gateway.send_reply)) for _ in range(3)]

    try:
        logger.info("Agent is running")
        await gateway.run()
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await gateway.close()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
