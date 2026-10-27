from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(
        self,
        documents: Sequence[tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids = [chunk_id for chunk_id, _ in documents]
        self.tokens = [tokenize(text) for _, text in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequencies: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequencies.update(set(tokens))
        total = len(self.tokens)
        self.idf = {
            term: math.log((total - frequency + 0.5) / (frequency + 0.5) + 1.0)
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self.tokens:
            return []

        scores: list[tuple[str, float]] = []
        for chunk_id, frequencies, length in zip(
            self.chunk_ids, self.term_frequencies, self.lengths, strict=True
        ):
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / self.average_length
                )
                score += self.idf[term] * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scores.append((chunk_id, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:limit]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> dict[str, float]:
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
