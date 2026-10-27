from __future__ import annotations

import html
import re


def highlight_quote_html(text: str, quote: str) -> str:
    escaped_text = html.escape(text)
    escaped_quote = html.escape(quote).strip()
    if not escaped_quote:
        return escaped_text

    parts = [re.escape(part) for part in escaped_quote.split()]
    parts = [part.replace("-", r"-\s*") for part in parts]
    pattern = re.compile(r"\s+".join(parts), re.IGNORECASE)
    match = pattern.search(escaped_text)
    if match is None:
        return escaped_text

    return (
        escaped_text[: match.start()]
        + "<mark>"
        + match.group(0)
        + "</mark>"
        + escaped_text[match.end() :]
    )
