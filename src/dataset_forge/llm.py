from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from dataset_forge.models import Message, ModelRouter


class LlmCallError(RuntimeError):
    """Raised when a live model call fails and no deterministic fallback was requested."""


@dataclass(slots=True)
class LlmClient:
    router: ModelRouter
    live: bool = False
    timeout_seconds: int = 45

    def complete(self, messages: list[Message], *, role: str, temperature: float = 0.7) -> str | None:
        if not self.live:
            return None
        model = self.router.model_for_role(role)
        api_key = os.environ.get(self.router.api_key_env)
        if not api_key:
            raise LlmCallError(
                f"Live LLM mode requested, but {self.router.api_key_env} is not set. "
                "Disable live mode or provide an OpenCode Go API key."
            )
        payload = {
            "model": model.replace("opencode-go/", ""),
            "messages": messages,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            self.router.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise LlmCallError(f"OpenCode Go HTTP error {error.code}: {body}") from error
        except OSError as error:
            raise LlmCallError(f"OpenCode Go request failed: {error}") from error
        try:
            parsed = json.loads(raw)
            return str(parsed["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise LlmCallError(f"OpenCode Go returned an unexpected response shape: {raw[:500]}") from error
