from __future__ import annotations

import json
import re

from pydantic import ValidationError

from rag_eval.generation.citations import LLMAnswer

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class OutputParseError(ValueError):
    pass


def extract_json_object(raw: str) -> dict:
    text = raw.strip()
    text = _FENCE.sub("", text).strip()
    if text.startswith("["):
        raise OutputParseError("Model output JSON must be an object")

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise OutputParseError("Model output did not contain a JSON object")

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise OutputParseError(f"Model output was not valid JSON: {error}") from error
    if not isinstance(data, dict):
        raise OutputParseError("Model output JSON must be an object")
    return data


def parse_llm_answer(raw: str) -> LLMAnswer:
    data = extract_json_object(raw)

    try:
        return LLMAnswer.model_validate(data)
    except ValidationError as error:
        raise OutputParseError(
            f"Model output did not match the expected schema: {error}"
        ) from error
