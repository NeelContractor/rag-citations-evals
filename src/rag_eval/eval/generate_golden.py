from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from rag_eval.config import GenerationConfig, LLMConfig
from rag_eval.eval.golden import Evidence, GoldenItem, load_chunks, save_golden
from rag_eval.generation.citations import REFUSAL_ANSWER, normalize_whitespace, quote_in_chunk
from rag_eval.generation.openrouter import LLMClient, LLMError
from rag_eval.generation.output import OutputParseError, extract_json_object
from rag_eval.retrieval.types import RetrievedChunk

LLM_SYSTEM_PROMPT = """You create golden evaluation data for a RAG system.

For every source chunk, write one question that can be answered using only that chunk,
a concise reference answer, and one supporting quote copied verbatim from the chunk.
Return a single JSON object with an "items" array. Each item must contain:
- chunk_id
- question
- reference_answer
- evidence_quote

Rules:
- Produce exactly one item for every chunk_id in the user message.
- Use only facts stated in that chunk. Do not use outside knowledge.
- The evidence quote must be an exact substring of the chunk, at least 15 characters long.
- Questions must be self-contained and unambiguous.
- Do not wrap the JSON in markdown fences.
"""

UNANSWERABLE_QUESTIONS = [
    "What is the capital of France?",
    "Who won the 1998 FIFA World Cup?",
    "What is the boiling point of mercury?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for tungsten?",
    "Who wrote Pride and Prejudice?",
    "Which planet has the most moons?",
    "What is the population of Tokyo in 2026?",
    "What is the runtime complexity of quicksort in the worst case?",
    "Who invented the telephone?",
]

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def select_chunks(
    chunks: list[RetrievedChunk],
    count: int,
    seed: int = 13,
) -> list[RetrievedChunk]:
    usable = [chunk for chunk in chunks if len(chunk.text) >= 80]
    selected = usable if len(usable) <= count else random.Random(seed).sample(usable, count)
    return sorted(selected, key=lambda chunk: (chunk.doc, chunk.page or 0, chunk.chunk_id))


def first_sentence(text: str) -> str:
    for sentence in _SENTENCE.split(text.strip()):
        sentence = sentence.strip()
        if len(sentence) >= 40:
            return sentence
    collapsed = normalize_whitespace(text)
    if len(collapsed) <= 240:
        return collapsed
    return collapsed[:240].rsplit(" ", 1)[0]


def heuristic_item(chunk: RetrievedChunk) -> GoldenItem:
    quote = first_sentence(chunk.text)
    topic = ""
    if chunk.heading:
        topic = chunk.heading.split(" > ")[-1]
    if not topic:
        topic = " ".join(chunk.text.split()[:8])
    topic = topic.rstrip("?.!")[:80]

    return GoldenItem(
        id="",
        question=f"In {chunk.doc}, what does the source say about {topic}?",
        reference_answer=quote,
        answerable=True,
        evidence=[Evidence(doc=chunk.doc, page=chunk.page, quote=quote)],
        source_chunk_ids=[chunk.chunk_id],
        notes="Heuristic draft; review the question and evidence.",
    )


def heuristic_items(chunks: list[RetrievedChunk], count: int, seed: int = 13) -> list[GoldenItem]:
    return [heuristic_item(chunk) for chunk in select_chunks(chunks, count, seed)]


def _batch_prompt(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        location = chunk.doc if chunk.page is None else f"{chunk.doc}, page {chunk.page}"
        heading = chunk.heading or ""
        blocks.append(
            f"[{number}] chunk_id: {chunk.chunk_id}\n"
            f"Source: {location}\n"
            f"Heading: {heading}\n\n"
            f"{chunk.text}"
        )
    return "Chunks:\n\n" + "\n\n---\n\n".join(blocks)


def _items_from_response(
    raw: str,
    chunks: list[RetrievedChunk],
) -> dict[str, GoldenItem]:
    data = extract_json_object(raw)
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise OutputParseError("Expected an 'items' array")

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    items: dict[str, GoldenItem] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        chunk_id = raw_item.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        question = str(raw_item.get("question", "")).strip()
        reference_answer = str(raw_item.get("reference_answer", "")).strip()
        quote = str(raw_item.get("evidence_quote", "")).strip()
        chunk = by_id.get(chunk_id)
        if chunk is None or not question or not reference_answer:
            continue
        if len(normalize_whitespace(quote)) < 15 or not quote_in_chunk(quote, chunk.text):
            continue
        items[chunk_id] = GoldenItem(
            id="",
            question=question,
            reference_answer=reference_answer,
            answerable=True,
            evidence=[Evidence(doc=chunk.doc, page=chunk.page, quote=quote)],
            source_chunk_ids=[chunk.chunk_id],
            notes="LLM draft; review the question and evidence.",
        )
    return items


def generate_with_llm(
    chunks: list[RetrievedChunk],
    llm: LLMClient,
    batch_size: int = 4,
    log=print,
) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        generated: dict[str, GoldenItem] = {}
        for attempt in range(2):
            try:
                raw = llm.complete(LLM_SYSTEM_PROMPT, _batch_prompt(batch))
                generated = _items_from_response(raw, batch)
                break
            except (LLMError, OutputParseError) as error:
                if attempt == 1:
                    log(f"LLM generation failed for batch {start // batch_size + 1}: {error}")

        for chunk in batch:
            items.append(generated.get(chunk.chunk_id) or heuristic_item(chunk))
    return items


def unanswerable_items(count: int) -> list[GoldenItem]:
    return [
        GoldenItem(
            id="",
            question=question,
            reference_answer=REFUSAL_ANSWER,
            answerable=False,
            evidence=[],
            source_chunk_ids=[],
            notes="Unanswerable draft; verify that the corpus does not cover this.",
        )
        for question in UNANSWERABLE_QUESTIONS[:count]
    ]


def build_golden(
    chunks: list[RetrievedChunk],
    count: int,
    unanswerable_count: int,
    llm: LLMClient | None = None,
    batch_size: int = 4,
    seed: int = 13,
    log=print,
) -> list[GoldenItem]:
    selected = select_chunks(chunks, count, seed)
    if llm is None:
        answerable = [heuristic_item(chunk) for chunk in selected]
    else:
        answerable = generate_with_llm(selected, llm, batch_size=batch_size, log=log)

    if len(answerable) < count:
        log(
            f"Warning: only {len(answerable)} usable chunks were available for "
            f"{count} requested answerable items."
        )

    items = answerable + unanswerable_items(unanswerable_count)
    deduped: list[GoldenItem] = []
    seen_questions: set[str] = set()
    for item in items:
        key = normalize_whitespace(item.question).casefold()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        deduped.append(item)

    return [
        item.model_copy(update={"id": f"q{index:03d}"})
        for index, item in enumerate(deduped, start=1)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a draft golden set from indexed chunks")
    parser.add_argument("--db", type=Path, default=Path(".lancedb"))
    parser.add_argument("--table", default="chunks")
    parser.add_argument("--out", type=Path, default=Path("data/evals/golden_draft.jsonl"))
    parser.add_argument("--count", type=int, default=30, help="answerable questions to draft")
    parser.add_argument("--unanswerable", type=int, default=5)
    parser.add_argument("--mode", choices=["auto", "llm", "heuristic"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists. Use --force to overwrite.")

    try:
        chunks = load_chunks(args.db, args.table)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    if not chunks:
        raise SystemExit("The index is empty. Run `make ingest` first.")

    llm: LLMClient | None = None
    if args.mode != "heuristic":
        try:
            config = LLMConfig.from_env()
        except ValueError as error:
            if args.mode == "llm":
                raise SystemExit(str(error)) from error
            print("No LLM API key found; falling back to heuristic mode.")
        else:
            if args.model:
                config = config.model_copy(update={"model": args.model})
            llm = LLMClient(
                config,
                GenerationConfig(temperature=0.2, max_tokens=1400),
            )

    items = build_golden(
        chunks,
        count=args.count,
        unanswerable_count=args.unanswerable,
        llm=llm,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    save_golden(args.out, items)

    answerable_count = sum(1 for item in items if item.answerable)
    print(
        f"Wrote {len(items)} items ({answerable_count} answerable, "
        f"{len(items) - answerable_count} unanswerable) to {args.out}"
    )
    print("Review the draft, then copy it to data/evals/golden.jsonl before running `make eval`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
