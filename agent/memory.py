from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


@dataclass(slots=True)
class StoredMessage:
    role: str
    text: str
    created_at: str


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS processed_messages (
                    chat_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(chat_id, message_id)
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat_created
                ON messages(chat_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS user_profile_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    fact_key TEXT NOT NULL,
                    fact_value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_profile_user
                ON user_profile_facts(user_id, created_at DESC);
                """
            )
            await db.commit()

    async def is_processed(self, chat_id: int, message_id: int) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT 1 FROM processed_messages WHERE chat_id=? AND message_id=?",
                (chat_id, message_id),
            ) as cur:
                row = await cur.fetchone()
                return row is not None

    async def mark_processed(self, chat_id: int, message_id: int) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO processed_messages(chat_id, message_id, created_at)
                VALUES(?, ?, ?)
                """,
                (chat_id, message_id, now),
            )
            await db.commit()

    async def add_message(
        self,
        *,
        chat_id: int,
        user_id: int,
        role: str,
        text: str,
        meta: dict,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO messages(chat_id, user_id, role, text, meta_json, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (chat_id, user_id, role, text, json.dumps(meta), now),
            )
            await db.commit()

    async def get_recent_messages(self, chat_id: int, limit: int = 20) -> list[StoredMessage]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT role, text, created_at
                FROM messages
                WHERE chat_id=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (chat_id, limit),
            ) as cur:
                rows = await cur.fetchall()

        return [StoredMessage(role=r[0], text=r[1], created_at=r[2]) for r in reversed(rows)]

    async def add_profile_fact(self, user_id: int, key: str, value: str, confidence: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO user_profile_facts(user_id, fact_key, fact_value, confidence, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (user_id, key, value, float(confidence), now),
            )
            await db.commit()

    async def get_profile_facts(self, user_id: int, limit: int = 10) -> list[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT fact_key, fact_value, confidence
                FROM user_profile_facts
                WHERE user_id=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ) as cur:
                rows = await cur.fetchall()

        return [f"{k}: {v} (conf={c:.2f})" for (k, v, c) in rows]
