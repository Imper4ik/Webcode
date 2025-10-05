"""Judge0 API helper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .config import AppConfig


@dataclass(slots=True)
class Judge0Client:
    config: AppConfig

    def headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.judge0.api_key and self.config.judge0.host_header:
            headers.update({
                "x-rapidapi-key": self.config.judge0.api_key,
                "x-rapidapi-host": self.config.judge0.host_header,
            })
        return headers

    def run(self, source_code: str, stdin: str = "", expected_output: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.config.judge0.url}/submissions?base64_encoded=false&wait=true"
        payload = {
            "language_id": self.config.judge0.language_id,
            "source_code": source_code,
            "stdin": stdin,
        }
        response = requests.post(url, headers=self.headers(), json=payload, timeout=60)
        if self.config.debug_log:
            print("[J0] Status:", response.status_code)
            print("[J0] Body:", response.text[:500])
        response.raise_for_status()
        return response.json()


__all__ = ["Judge0Client"]
