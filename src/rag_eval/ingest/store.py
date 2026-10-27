from __future__ import annotations

from pathlib import Path

import lancedb

from rag_eval.config import StoreConfig
from rag_eval.models import Chunk


class ChunkStore:
    def __init__(self, config: StoreConfig) -> None:
        self.config = config

    def replace(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            raise ValueError("cannot write an empty chunk set")

        records = [
            {
                "chunk_id": chunk.chunk_id,
                "doc": chunk.doc,
                "page": chunk.page if chunk.page is not None else 0,
                "chunk_index": chunk.chunk_index,
                "heading": chunk.heading or "",
                "text": chunk.text,
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.config.db_path.mkdir(parents=True, exist_ok=True)
        database = lancedb.connect(str(self.config.db_path))
        database.create_table(self.config.table_name, data=records, mode="overwrite")

    def count(self, table_name: str | None = None) -> int:
        database = lancedb.connect(str(self.config.db_path))
        table = database.open_table(table_name or self.config.table_name)
        return table.count_rows()

    def exists(self, table_name: str | None = None) -> bool:
        database = lancedb.connect(str(self.config.db_path))
        name = table_name or self.config.table_name
        return name in database.table_names()

    @property
    def table_path(self) -> Path:
        return self.config.db_path / f"{self.config.table_name}.lance"
