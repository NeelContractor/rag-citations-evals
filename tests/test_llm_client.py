from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from rag_eval.config import LLMConfig
from rag_eval.generation import openrouter as openrouter_module
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
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        seen["user_agent"] = request.get_header("User-agent")
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = LLMConfig(
        api_key="groq-key",
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        provider="groq",
    )

    result = LLMClient(config).complete("system", "user")

    assert result == "ok"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["authorization"] == "Bearer groq-key"
    assert seen["user_agent"] == "rag-eval/1.0"
    assert seen["payload"]["response_format"] == {"type": "json_object"}
    assert seen["payload"]["reasoning_effort"] == "low"
    assert seen["timeout"] == config.timeout


def test_llm_client_honors_retry_after(monkeypatch):
    calls = 0
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                io.BytesIO(b'{"error":"rate limited"}'),
            )
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openrouter_module.time, "sleep", sleeps.append)
    config = LLMConfig(
        api_key="groq-key",
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        provider="groq",
    )

    result = LLMClient(config).complete("system", "user")

    assert result == "ok"
    assert calls == 2
    assert sleeps == [0.5]
