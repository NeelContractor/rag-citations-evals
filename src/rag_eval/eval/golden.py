from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import lancedb
from pydantic import BaseModel, Field, ValidationError, model_validator

from rag_eval.generation.citations import normalize_for_matching, normalize_whitespace
from rag_eval.retrieval.types import RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")
MATCH_THRESHOLD = 0.8


class Evidence(BaseModel):
    doc: str
    page: int | None = None
    quote: str = Field(min_length=3)


class GoldenItem(BaseModel):
    id: str
    question: str = Field(min_length=3)
    reference_answer: str = ""
    answerable: bool = True
    evidence: list[Evidence] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def answerable_items_need_evidence(self) -> GoldenItem:
        if self.answerable and not self.evidence:
            raise ValueError("answerable golden items must include at least one evidence span")
        return self


def load_golden(path: Path) -> list[GoldenItem]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found: {path}")
    items: list[GoldenItem] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = GoldenItem.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(f"{path}:{line_number}: invalid golden item\n{error}") from error
        if item.id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {item.id!r}")
        seen_ids.add(item.id)
        items.append(item)
    if not items:
        raise ValueError(f"Golden set is empty: {path}")
    return items


def save_golden(path: Path, items: Iterable[GoldenItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json() for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_chunks(db_path: Path, table_name: str) -> list[RetrievedChunk]:
    if not db_path.exists():
        raise FileNotFoundError(f"Vector store not found at {db_path}. Run `make ingest` first.")
    database = lancedb.connect(str(db_path))
    if table_name not in database.table_names():
        raise FileNotFoundError(
            f"Table {table_name!r} not found in {db_path}. Run `make ingest` first."
        )
    rows = (
        database.open_table(table_name)
        .to_arrow()
        .select(["chunk_id", "doc", "page", "heading", "text"])
        .to_pylist()
    )
    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"],
            doc=row["doc"],
            page=row["page"] or None,
            heading=row["heading"] or None,
            text=row["text"],
            score=0.0,
        )
        for row in rows
    ]


def map_evidence_to_chunks(evidence: Evidence, chunks: list[RetrievedChunk]) -> set[str]:
    candidates = [chunk for chunk in chunks if chunk.doc == evidence.doc]
    if evidence.page is not None:
        candidates = [chunk for chunk in candidates if chunk.page == evidence.page]
    if not candidates:
        return set()

    quote = normalize_for_matching(evidence.quote).casefold()
    if not quote:
        return set()

    exact = {
        chunk.chunk_id
        for chunk in candidates
        if quote in normalize_for_matching(chunk.text).casefold()
    }
    if exact:
        return exact

    quote_tokens = set(_TOKEN.findall(quote))
    if not quote_tokens:
        return set()

    scored: list[tuple[float, str]] = []
    for chunk in candidates:
        chunk_tokens = set(_TOKEN.findall(normalize_whitespace(chunk.text).casefold()))
        coverage = len(quote_tokens & chunk_tokens) / len(quote_tokens)
        scored.append((coverage, chunk.chunk_id))

    best = max(coverage for coverage, _ in scored)
    if best < MATCH_THRESHOLD:
        return set()
    return {chunk_id for coverage, chunk_id in scored if coverage == best}


def map_gold_chunks(item: GoldenItem, chunks: list[RetrievedChunk]) -> set[str]:
    if not item.answerable:
        return set()
    mapped: set[str] = set()
    for evidence in item.evidence:
        mapped.update(map_evidence_to_chunks(evidence, chunks))
    return mapped
