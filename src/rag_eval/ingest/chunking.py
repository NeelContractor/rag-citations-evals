from __future__ import annotations

import hashlib
import re

from rag_eval.config import ChunkingConfig
from rag_eval.models import Chunk, RawDocument

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def chunk_document(
    raw: RawDocument,
    config: ChunkingConfig,
    start_index: int = 0,
) -> list[Chunk]:
    if config.strategy == "heading" and raw.doc.lower().endswith((".md", ".markdown")):
        return _chunk_heading_aware(raw, config, start_index)
    return _chunk_fixed(raw.text, raw, config, start_index, heading=None)


def _chunk_fixed(
    text: str,
    raw: RawDocument,
    config: ChunkingConfig,
    start_index: int,
    heading: str | None,
) -> list[Chunk]:
    normalized = _normalize(text)
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = start_index
    size = config.chunk_size
    overlap = config.chunk_overlap

    while start < len(normalized):
        end = min(start + size, len(normalized))
        if end < len(normalized):
            boundary = _last_whitespace(normalized, start + size // 2, end)
            if boundary > start:
                end = boundary

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(raw.doc, raw.page, index, piece),
                    doc=raw.doc,
                    page=raw.page,
                    chunk_index=index,
                    heading=heading,
                    text=piece,
                )
            )
            index += 1

        if end >= len(normalized):
            break

        candidate = max(end - overlap, start + 1)
        probe = candidate
        while probe < len(normalized) and not normalized[probe].isspace():
            probe += 1
        align_to_word = probe < len(normalized) and probe - candidate <= size // 2
        next_start = probe if align_to_word else candidate
        while next_start < len(normalized) and normalized[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def _chunk_heading_aware(
    raw: RawDocument,
    config: ChunkingConfig,
    start_index: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = start_index
    for heading_path, body in _split_sections(raw.text):
        heading = " > ".join(heading_path) if heading_path else None
        section_chunks = _chunk_fixed(body, raw, config, index, heading=heading)
        chunks.extend(section_chunks)
        index += len(section_chunks)
    return chunks


def _split_sections(markdown: str) -> list[tuple[list[str], str]]:
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False
    fence = ""

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append(([title for _, title in stack], text))

    for line in markdown.replace("\r\n", "\n").split("\n"):
        stripped = line.lstrip()
        fence_match = _FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence = marker
            elif marker == fence:
                in_fence = False
                fence = ""
            body.append(line)
            continue

        heading_match = None if in_fence else _HEADING.match(stripped)
        if heading_match:
            flush()
            body = []
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)

    flush()
    return sections


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _last_whitespace(text: str, lower: int, upper: int) -> int:
    for position in range(upper, max(lower, 0), -1):
        if text[position - 1].isspace():
            return position
    return upper


def _chunk_id(doc: str, page: int | None, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{doc}:{page}:{index}:{text}".encode()).hexdigest()[:12]
    page_part = "p0" if page is None else f"p{page}"
    return f"{_slug(doc)}-{page_part}-{index:04d}-{digest}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "doc"
