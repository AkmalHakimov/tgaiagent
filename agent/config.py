from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tg_api_id: int = Field(alias="TG_API_ID")
    tg_api_hash: str = Field(alias="TG_API_HASH")

    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    agent_name: str = Field(default="Orion", alias="AGENT_NAME")
    db_path: Path = Field(default=Path("./data/agent.db"), alias="DB_PATH")
    session_name: str = Field(default="./data/telegram.session", alias="SESSION_NAME")

    max_context_messages: int = Field(default=25, alias="MAX_CONTEXT_MESSAGES")
    max_reply_chars: int = Field(default=1600, alias="MAX_REPLY_CHARS")
    enable_voice_notes: bool = Field(default=False, alias="ENABLE_VOICE_NOTES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    Path(settings.session_name).parent.mkdir(parents=True, exist_ok=True)
    return settings
