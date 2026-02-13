from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from telethon import TelegramClient, events

from .types import IncomingMessage

logger = logging.getLogger(__name__)


class TelegramGateway:
    def __init__(self, api_id: int, api_hash: str, session_name: str) -> None:
        self._client = TelegramClient(session_name, api_id, api_hash)
        self._on_message: Callable[[IncomingMessage], Awaitable[None]] | None = None
        self._lock = asyncio.Lock()

    def register_handler(self, callback: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        self._on_message = callback

    async def start(self) -> None:
        await self._client.start()
        me = await self._client.get_me()
        logger.info("Telegram client connected as %s (%s)", me.username or me.first_name, me.id)

        @self._client.on(events.NewMessage(incoming=True))
        async def _listener(event: events.NewMessage.Event) -> None:
            if not event.is_private:
                return
            if event.message is None:
                return
            text = event.raw_text or ""
            sender = await event.get_sender()
            sender_name = " ".join(filter(None, [getattr(sender, "first_name", ""), getattr(sender, "last_name", "")])).strip()
            incoming = IncomingMessage(
                message_id=event.message.id,
                chat_id=event.chat_id,
                user_id=sender.id if sender else event.sender_id,
                sender_name=sender_name or (getattr(sender, "username", "") or "Unknown"),
                text=text,
            )
            if self._on_message is not None:
                await self._on_message(incoming)

    async def run(self) -> None:
        await self._client.run_until_disconnected()

    async def send_reply(self, chat_id: int, text: str) -> None:
        async with self._lock:
            await self._client.send_message(entity=chat_id, message=text)

    async def close(self) -> None:
        await self._client.disconnect()
