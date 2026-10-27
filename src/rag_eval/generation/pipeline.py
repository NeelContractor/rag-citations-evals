from __future__ import annotations

from pydantic import BaseModel

from rag_eval.config import GenerationConfig
from rag_eval.generation.citations import (
    REFUSAL_ANSWER,
    InvalidCitation,
    VerifiedCitation,
    verify_citations,
)
from rag_eval.generation.output import OutputParseError, parse_llm_answer
from rag_eval.generation.prompts import RETRY_SUFFIX, SYSTEM_PROMPT, build_user_prompt
from rag_eval.retrieval.types import RetrievedChunk


class GenerationError(RuntimeError):
    pass


class RagResponse(BaseModel):
    question: str
    answer: str
    citations: list[VerifiedCitation]
    invalid_citations: list[InvalidCitation]
    refused: bool = False
    refusal_reason: str | None = None
    retrieved: list[RetrievedChunk]
    model: str


class RagPipeline:
    def __init__(
        self,
        retriever,
        llm,
        config: GenerationConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or GenerationConfig()

    def ask(self, question: str, top_k: int | None = None) -> RagResponse:
        if top_k is None:
            chunks = self.retriever.search(question)
        else:
            chunks = self.retriever.search(question, top_k=top_k)
        return self.answer_with_chunks(question, chunks)

    def answer_with_chunks(self, question: str, chunks: list[RetrievedChunk]) -> RagResponse:
        if not chunks:
            return self._refusal(question, [], "no_context")

        user_prompt = build_user_prompt(question, chunks)
        last_error: OutputParseError | None = None

        for attempt in range(self.config.max_retries + 1):
            prompt = user_prompt if attempt == 0 else user_prompt + RETRY_SUFFIX
            raw = self.llm.complete(SYSTEM_PROMPT, prompt)
            try:
                parsed = parse_llm_answer(raw)
            except OutputParseError as error:
                last_error = error
                continue

            verified = verify_citations(
                parsed,
                chunks,
                require_citations=self.config.require_citations,
            )
            return RagResponse(
                question=question,
                answer=verified.answer,
                citations=verified.citations,
                invalid_citations=verified.invalid_citations,
                refused=verified.refused,
                refusal_reason=verified.refusal_reason,
                retrieved=chunks,
                model=self.llm.model,
            )

        raise GenerationError(f"Model did not return valid JSON: {last_error}")

    def _refusal(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        reason: str,
    ) -> RagResponse:
        return RagResponse(
            question=question,
            answer=REFUSAL_ANSWER,
            citations=[],
            invalid_citations=[],
            refused=True,
            refusal_reason=reason,
            retrieved=chunks,
            model=self.llm.model,
        )
