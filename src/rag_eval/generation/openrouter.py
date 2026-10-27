from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from rag_eval.config import GenerationConfig, OpenRouterConfig


class LLMError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        config: OpenRouterConfig,
        generation: GenerationConfig | None = None,
    ) -> None:
        self.config = config
        self.generation = generation or GenerationConfig()

    @property
    def model(self) -> str:
        return self.config.model

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.generation.temperature,
            "max_tokens": self.generation.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "X-Title": self.config.app_name,
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    body = json.load(response)
                return self._extract_content(body)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                last_error = LLMError(f"OpenRouter returned HTTP {error.code}: {detail}")
                if error.code != 429 and error.code < 500:
                    raise last_error from error
            except urllib.error.URLError as error:
                last_error = LLMError(f"Could not reach OpenRouter: {error.reason}")
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
                raise LLMError(f"Unexpected OpenRouter response: {error}") from error

            if attempt < 2:
                time.sleep(1 + attempt)

        raise last_error or LLMError("OpenRouter request failed")

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
        raise LLMError("OpenRouter response did not contain text content")
