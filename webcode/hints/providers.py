"""Hint provider integrations."""
from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterable

import requests

from ..config import HintConfig


class ProviderError(RuntimeError):
    """Raised when a provider is unavailable."""


def _normalize_model(name: str | None) -> str:
    name = (name or "").strip()
    return name[7:] if name.startswith("models/") else name


def openai_chat_with_retry(messages: Iterable[Dict[str, Any]], config: HintConfig, *, max_retries: int = 5,
                           timeout: int = 30, debug: bool = False) -> Dict[str, Any]:
    if not config.openai_api_key:
        raise ProviderError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": config.openai_model, "messages": list(messages), "temperature": 0.2}

    attempt = 0
    while True:
        attempt += 1
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt > max_retries:
                    raise ProviderError(response.text)
                delay = _retry_delay(response, attempt)
                if debug:
                    print(f"[OpenAI] {response.status_code}, retry in {delay:.2f}s ({attempt}/{max_retries})")
                time.sleep(delay)
                continue
            raise ProviderError(response.text)
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            if attempt > max_retries:
                raise ProviderError(str(exc)) from exc
            delay = _exponential_backoff(attempt)
            if debug:
                print(f"[OpenAI] network error {exc}, retry in {delay:.2f}s ({attempt}/{max_retries})")
            time.sleep(delay)


def _retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return _exponential_backoff(attempt)


def _exponential_backoff(attempt: int) -> float:
    base = min(8, 0.5 * (2 ** (attempt - 1)))
    return base + random.uniform(0, 0.333 * base)


def gemini_generate(prompt: str, config: HintConfig, *, timeout: int = 30, debug: bool = False) -> str:
    if not config.gemini_api_key:
        raise ProviderError("GEMINI_API_KEY is not set")

    primary = _normalize_model(config.gemini_model or "gemini-2.5-flash")
    fallbacks = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
    ]

    seen: set[str] = set()
    models: list[str] = []
    for name in [primary, *fallbacks]:
        normal = _normalize_model(name)
        if normal and normal not in seen:
            seen.add(normal)
            models.append(normal)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    last_error: str | None = None

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={config.gemini_api_key}"
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    raise ProviderError(f"No candidates ({model})")
                parts = candidates[0].get("content", {}).get("parts") or []
                if not parts:
                    raise ProviderError(f"No parts ({model})")
                return (parts[0].get("text") or "").strip()
            last_error = f"{model}: {response.status_code} {response.text[:200]}"
            if debug:
                print("[Gemini] FAIL:", last_error)
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            last_error = f"{model}: {exc}"
            if debug:
                print("[Gemini] NET ERR:", last_error)

    raise ProviderError(last_error or "Gemini not available")


__all__ = ["ProviderError", "openai_chat_with_retry", "gemini_generate"]
