from __future__ import annotations

import json
import urllib.request

from rag_eval.config import LLMConfig
from rag_eval.generation.openrouter import LLMClient


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self.payload


def test_llm_client_uses_configured_base_url_and_key(monkeypatch):
    seen: dict[str, str | float | None] = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = LLMConfig(
        api_key="groq-key",
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        provider="groq",
    )

    result = LLMClient(config).complete("system", "user")

    assert result == "ok"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["authorization"] == "Bearer groq-key"
    assert seen["timeout"] == config.timeout
