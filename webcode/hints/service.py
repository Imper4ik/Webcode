"""Hint service that orchestrates provider selection and caching."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

from ..config import AppConfig
from .providers import ProviderError, gemini_generate, openai_chat_with_retry


@dataclass(slots=True)
class CacheItem:
    """A cached hint along with its expiration timestamp."""

    hint: str
    until: float


@dataclass(slots=True)
class HintService:
    """Provide rate-limited hints with provider fallbacks and caching."""

    config: AppConfig
    cache_ttl: int = 60
    _cache: Dict[str, CacheItem] = field(default_factory=dict)

    def get_hint(self, description: str, code: str) -> tuple[str, bool]:
        """Return a hint for *code* and whether it was served from cache."""

        cache_key = f"{description}\n\n{code}"
        cached = self._cache.get(cache_key)
        if cached and cached.until > time.time():
            return cached.hint, True
        if cached:
            self._cache.pop(cache_key, None)

        hint = self._fetch_hint(description, code)
        self._cache[cache_key] = CacheItem(hint=hint, until=time.time() + self.cache_ttl)
        return hint, False

    def _fetch_hint(self, description: str, code: str) -> str:
        system_msg = {
            "role": "system",
            "content": "Ты помогаешь студенту. Дай краткую подсказку (1–3 предложения), не раскрывая готового решения.",
        }
        user_msg = {"role": "user", "content": f"Задание: {description}\n\nМой код:\n{code}"}

        last_error = ""
        for provider in self.config.hint.provider_chain():
            try:
                if provider == "openai":
                    data = openai_chat_with_retry([system_msg, user_msg], self.config.hint, debug=self.config.debug_log)
                    return data["choices"][0]["message"]["content"].strip()
                if provider == "gemini":
                    prompt = (
                        f"Задание: {description}\n\nМой код:\n{code}\n\n" "Дай подсказку кратко (1–3 предложения), без полного решения."
                    )
                    return gemini_generate(prompt, self.config.hint, debug=self.config.debug_log)
                return "Подсказки офлайн: проверь, что возвращаемое значение соответствует тестам и граничным случаям."
            except ProviderError as error:  # pragma: no cover - network failure path
                if self.config.debug_log:
                    print(f"[Hint] provider {provider} error: {error}")
                last_error = str(error)
        raise RuntimeError(last_error or "No hint provider available")


__all__ = ["HintService"]
