from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from rag_eval.generation.citations import is_refusal, normalize_whitespace
from rag_eval.generation.output import OutputParseError, extract_json_object
from rag_eval.retrieval.types import RetrievedChunk

JUDGE_SYSTEM_PROMPT = """You are a strict evaluation judge for a RAG system.

Decide whether every claim in the answer is supported by the provided sources.
Return one JSON object with exactly these keys:
{"faithful": true, "score": 1.0, "unsupported_claims": []}

Rules:
- Score 1.0 means every claim is directly supported by the sources.
- Score 0.0 means the answer is unsupported or contradicted.
- Use intermediate scores for answers that are only partly supported.
- List any unsupported claims in unsupported_claims.
- A refusal makes no factual claims, so score it 1.0.
- Do not use outside knowledge.
"""


class FaithfulnessJudgement(BaseModel):
    faithful: bool
    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


class JudgeError(RuntimeError):
    pass


class FaithfulnessJudge:
    def __init__(self, llm, max_context_chars: int = 6000) -> None:
        self.llm = llm
        self.max_context_chars = max_context_chars

    def judge(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> FaithfulnessJudgement:
        if is_refusal(answer):
            return FaithfulnessJudgement(faithful=True, score=1.0)

        prompt = self._build_prompt(question, answer, chunks)
        raw = self.llm.complete(JUDGE_SYSTEM_PROMPT, prompt)
        try:
            data = extract_json_object(raw)
        except OutputParseError as error:
            raise JudgeError(str(error)) from error

        try:
            return FaithfulnessJudgement.model_validate(data)
        except ValidationError as error:
            raise JudgeError(f"Judge output did not match the expected schema: {error}") from error

    def _build_prompt(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        blocks: list[str] = []
        used = 0
        for chunk in chunks:
            block = f"chunk_id: {chunk.chunk_id}\n{normalize_whitespace(chunk.text)}"
            if used + len(block) > self.max_context_chars:
                remaining = max(self.max_context_chars - used, 0)
                if remaining:
                    blocks.append(block[:remaining])
                break
            blocks.append(block)
            used += len(block)

        return f"Question: {question}\n\nAnswer: {answer}\n\nSources:\n\n" + "\n\n---\n\n".join(
            blocks
        )
