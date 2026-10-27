from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(gold_ids: Iterable[str], retrieved_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return len(gold & set(retrieved_ids)) / len(gold)


def reciprocal_rank(gold_ids: Iterable[str], retrieved_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def citation_precision(
    cited_ids: Iterable[str],
    invalid_ids: Iterable[str],
    gold_ids: Iterable[str],
) -> float:
    cited = list(cited_ids)
    invalid = list(invalid_ids)
    total = len(cited) + len(invalid)
    if total == 0:
        return 0.0
    gold = set(gold_ids)
    correct = sum(1 for chunk_id in cited if chunk_id in gold)
    return correct / total


def citation_recall(cited_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    return len(gold & set(cited_ids)) / len(gold)


def citation_validity(cited_ids: Iterable[str], invalid_ids: Iterable[str]) -> float:
    cited = list(cited_ids)
    invalid = list(invalid_ids)
    total = len(cited) + len(invalid)
    if total == 0:
        return 1.0
    return len(cited) / total


def refusal_rate(refusals: Iterable[bool]) -> float:
    values = list(refusals)
    if not values:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return sum(usable) / len(usable)


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows
    ]
    return "\n".join([header_line, separator, *body])
