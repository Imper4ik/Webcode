"""Simple JSON storage helpers."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict

from .config import AppConfig


@dataclass(slots=True)
class JsonStorage:
    users_json: str
    topics_json: str

    def read(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def write(self, path: str, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def read_users(self) -> Dict[str, Any]:
        return self.read(self.users_json)

    def read_topics(self) -> Dict[str, Any]:
        return self.read(self.topics_json)

    def write_users(self, payload: Dict[str, Any]) -> None:
        self.write(self.users_json, payload)


DEFAULT_TOPICS = {
    "Python": {
        "Intro": {"tasks": ["task1"]},
        "Math": {"tasks": ["task2"]},
    }
}

DEFAULT_USERS = {"user1": {"completed_tasks": []}}


def ensure_data(config: AppConfig) -> JsonStorage:
    os.makedirs(config.data_dir, exist_ok=True)
    if not os.path.exists(config.topics_json):
        with open(config.topics_json, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_TOPICS, fh, ensure_ascii=False, indent=2)
    if not os.path.exists(config.users_json):
        with open(config.users_json, "w", encoding="utf-8") as fh:
            json.dump(DEFAULT_USERS, fh, ensure_ascii=False, indent=2)
    return JsonStorage(users_json=config.users_json, topics_json=config.topics_json)


__all__ = ["JsonStorage", "ensure_data", "DEFAULT_TOPICS", "DEFAULT_USERS"]
