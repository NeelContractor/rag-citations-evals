from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class RawDocument:
    doc: str
    page: int | None
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc: str
    page: int | None = None
    chunk_index: int = Field(ge=0)
    heading: str | None = None
    text: str

    @property
    def embedding_text(self) -> str:
        if self.heading:
            return f"{self.heading}\n\n{self.text}"
        return self.text
