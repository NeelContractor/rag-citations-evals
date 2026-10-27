from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag_eval.models import RawDocument

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}


def discover_documents(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Documents directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Documents path is not a directory: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_document(path: Path, root: Path) -> list[RawDocument]:
    relative = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[RawDocument] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(RawDocument(doc=relative, page=page_number, text=text))
        return pages

    if suffix in {".md", ".markdown"}:
        text = path.read_text(encoding="utf-8")
        return [RawDocument(doc=relative, page=None, text=text)] if text.strip() else []

    raise ValueError(f"Unsupported file type: {path}")
