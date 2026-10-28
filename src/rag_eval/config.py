from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
Provider = Literal["openrouter", "groq"]


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "heading"] = "fixed"
    chunk_size: int = Field(default=800, ge=50)
    chunk_overlap: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = Field(default=32, ge=1)
    cache_dir: Path = Path(".cache/fastembed")


class StoreConfig(BaseModel):
    db_path: Path = Path(".lancedb")
    table_name: str = "chunks"


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    hybrid: bool = False
    rerank: bool = False
    rrf_k: int = Field(default=60, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    rerank_model: str = "answerdotai/answerai-colbert-small-v1"

    @model_validator(mode="after")
    def validate_candidates(self) -> RetrievalConfig:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


class GenerationConfig(BaseModel):
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=800, ge=64)
    max_retries: int = Field(default=1, ge=0, le=3)
    require_citations: bool = True


class LLMConfig(BaseModel):
    api_key: str = Field(repr=False)
    model: str = DEFAULT_OPENROUTER_MODEL
    base_url: str = OPENROUTER_BASE_URL
    timeout: float = Field(default=90.0, gt=0)
    app_name: str = "rag-eval"
    provider: Provider = "openrouter"

    @property
    def provider_label(self) -> str:
        return "Groq" if self.provider == "groq" else "OpenRouter"

    @classmethod
    def from_env(cls, env_path: Path = Path(".env")) -> LLMConfig:
        load_env(env_path, override=True)
        provider = _detect_provider()
        api_key = _provider_api_key(provider)
        if not api_key:
            raise ValueError(
                "No LLM API key is set. Set OPENROUTER_API_KEY for OpenRouter, "
                "GROQ_API_KEY for Groq, or LLM_API_KEY for a custom endpoint."
            )
        return cls(
            api_key=api_key,
            model=_provider_model(provider),
            base_url=_provider_base_url(provider),
            provider=provider,
        )


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def _first_env(*names: str) -> str:
    for name in names:
        value = _env(name)
        if value:
            return value
    return ""


def _detect_provider() -> Provider:
    configured = _env("LLM_PROVIDER").casefold()
    if configured:
        if configured not in {"openrouter", "groq"}:
            raise ValueError("LLM_PROVIDER must be either 'openrouter' or 'groq'.")
        return configured

    base_url = _first_env("LLM_BASE_URL", "GROQ_BASE_URL", "OPENROUTER_BASE_URL")
    if "groq.com" in base_url.casefold():
        return "groq"
    if _env("GROQ_API_KEY") and not _env("OPENROUTER_API_KEY"):
        return "groq"
    return "openrouter"


def _provider_api_key(provider: Provider) -> str:
    if provider == "groq":
        return _first_env("LLM_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")
    return _first_env("LLM_API_KEY", "OPENROUTER_API_KEY")


def _provider_model(provider: Provider) -> str:
    if provider == "groq":
        return _first_env("LLM_MODEL", "GROQ_MODEL", "OPENROUTER_MODEL") or DEFAULT_GROQ_MODEL
    return _first_env("LLM_MODEL", "OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL


def _provider_base_url(provider: Provider) -> str:
    configured = _first_env("LLM_BASE_URL")
    if configured:
        return configured
    if provider == "groq":
        groq_url = _first_env("GROQ_BASE_URL")
        if groq_url:
            return groq_url
        legacy_url = _first_env("OPENROUTER_BASE_URL")
        if "groq.com" in legacy_url.casefold():
            return legacy_url
        return GROQ_BASE_URL
    return _first_env("OPENROUTER_BASE_URL") or OPENROUTER_BASE_URL


OpenRouterConfig = LLMConfig


def load_env(path: Path, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
