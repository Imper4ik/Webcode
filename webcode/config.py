"""Application configuration models and helpers."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List


def _lower(value: str | None) -> str:
    return (value or "").strip().lower()


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class HintConfig:
    provider: str
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str

    def provider_chain(self) -> List[str]:
        """Return an ordered list of providers to try."""
        provider = self.provider
        if provider in {"openai", "gemini", "offline"}:
            return [provider]

        chain: List[str] = []
        if self.openai_api_key:
            chain.append("openai")
        if self.gemini_api_key:
            chain.append("gemini")
        chain.append("offline")
        return chain


@dataclass(slots=True)
class Judge0Config:
    url: str
    api_key: str
    host_header: str
    language_id: int


@dataclass(slots=True)
class AppConfig:
    data_dir: str
    users_json: str
    topics_json: str
    debug_log: bool
    hint: HintConfig
    judge0: Judge0Config

    @classmethod
    def from_env(cls) -> "AppConfig":
        data_dir = os.getenv("DATA_DIR", "data")
        return cls(
            data_dir=data_dir,
            users_json=os.path.join(data_dir, "users.json"),
            topics_json=os.path.join(data_dir, "topics.json"),
            debug_log=os.getenv("DEBUG_LOG", "0") == "1",
            hint=HintConfig(
                provider=_lower(os.getenv("HINT_PROVIDER", "")),
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
                gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            ),
            judge0=Judge0Config(
                url=os.getenv("JUDGE0_URL", "https://judge0-ce.p.rapidapi.com"),
                api_key=os.getenv("JUDGE0_KEY", ""),
                host_header=os.getenv("JUDGE0_HOST_HEADER", "judge0-ce.p.rapidapi.com"),
                language_id=_int(os.getenv("JUDGE0_LANGUAGE_ID"), 71),
            ),
        )


__all__ = ["AppConfig", "HintConfig", "Judge0Config"]
