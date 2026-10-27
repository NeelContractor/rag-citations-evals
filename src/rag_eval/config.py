from __future__ import annotations

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
