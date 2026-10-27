from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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


class OpenRouterConfig(BaseModel):
    api_key: str = Field(repr=False)
    model: str = "recursive/flint-v1.0:free"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float = Field(default=90.0, gt=0)
    app_name: str = "rag-eval"

    @classmethod
    def from_env(cls, env_path: Path = Path(".env")) -> OpenRouterConfig:
        load_env(env_path)
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set. Add it to .env or the environment.")
        model = os.getenv("OPENROUTER_MODEL", cls.model_fields["model"].default)
        return cls(api_key=api_key, model=model)


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
