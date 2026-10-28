from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from rag_eval.config import GenerationConfig, LLMConfig


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        generation: GenerationConfig | None = None,
    ) -> None:
        self.config = config
        self.generation = generation or GenerationConfig()

    @property
    def model(self) -> str:
        return self.config.model

    def complete(self, system: str, user: str) -> str:
        provider = self.config.provider_label
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.generation.temperature,
            "max_tokens": self.generation.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.provider == "groq" and "gpt-oss" in self.config.model.casefold():
            payload["reasoning_effort"] = "low"
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"{self.config.app_name}/1.0",
                "X-Title": self.config.app_name,
            },
            method="POST",
        )

        last_error: Exception | None = None
        attempts = 5
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = json.load(response)
                return self._extract_content(body)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                last_error = LLMError(f"{provider} returned HTTP {error.code}: {detail}")
                retryable = (
                    error.code == 429 or error.code >= 500 or "json_validate_failed" in detail
                )
                if not retryable:
                    raise last_error from error
                delay = _http_retry_delay(error, detail, attempt)
            except urllib.error.URLError as error:
                last_error = LLMError(f"Could not reach {provider}: {error.reason}")
                delay = min(2**attempt, 8)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                raise LLMError(f"Unexpected {provider} response: {error}") from error

            if attempt < attempts - 1:
                time.sleep(delay)

        raise last_error or LLMError(f"{provider} request failed")

    def _extract_content(self, body: dict) -> str:
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in {"text", None}
            )
        raise LLMError(f"{self.config.provider_label} response did not contain text content")


def _http_retry_delay(error: urllib.error.HTTPError, detail: str, attempt: int) -> float:
    if error.code == 429:
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after:
            try:
                return min(max(float(retry_after) + 0.5, 0.5), 60.0)
            except ValueError:
                pass
        match = re.search(r"try again in ([\d.]+)s", detail, flags=re.IGNORECASE)
        if match:
            return min(max(float(match.group(1)) + 0.5, 0.5), 60.0)
        return min(2 ** (attempt + 1), 30)
    return min(2**attempt, 8)


OpenRouterClient = LLMClient
