from __future__ import annotations

from rag_eval.generation.citations import REFUSAL_ANSWER
from rag_eval.retrieval.types import RetrievedChunk

JSON_SHAPE = (
    '{"answer": "your answer", '
    '"citations": [{"chunk_id": "the source chunk_id", "quote": "exact text"}]}'
)

SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided by the user.

Rules:
1. Use only information explicitly stated in the sources. Do not use outside knowledge.
2. Every factual sentence in your answer must be supported by at least one citation.
3. Return a single JSON object with exactly this shape:
   {JSON_SHAPE}
4. Each quote must be copied verbatim from the chunk you cite. Do not paraphrase quotes.
5. Cite the chunk_id field shown in each source block, not the source number.
6. If the sources do not contain enough information to answer, return:
   {{"answer": "{REFUSAL_ANSWER}", "citations": []}}
7. Do not wrap the JSON in markdown fences and do not add commentary outside the JSON.
"""

RETRY_SUFFIX = """
Your previous response could not be parsed. Return only the required JSON object,
with no markdown fences or extra text.
"""


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        location = chunk.doc if chunk.page is None else f"{chunk.doc}, page {chunk.page}"
        heading = f"\nHeading: {chunk.heading}" if chunk.heading else ""
        blocks.append(
            f"[{number}] chunk_id: {chunk.chunk_id}\nSource: {location}{heading}\n\n{chunk.text}"
        )

    return f"Question: {question}\n\nSources:\n\n" + "\n\n---\n\n".join(blocks)
