"""
Extract text from PDFs and split into datapoints for embedding.

The paper proposal mentions LangChain's PyPDFLoader and text splitters; this
module uses `pypdf` plus a small paragraph splitter to avoid that
dependency by default. Install with: pip install -e ".[corpus]"
"""

from __future__ import annotations

import re
from pathlib import Path

from topic_boundaries.config import DEFAULT_CHUNK_MAX_CHARS, DEFAULT_CHUNK_OVERLAP
from topic_boundaries.documents import Datapoint


def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install pypdf "
            'or pip install -e ".[corpus]"'
        ) from e

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n\n".join(parts)


_WS = re.compile(r"[ \t\r\f\v]+")


def normalize_whitespace(text: str) -> str:
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Keep paragraphs separate; split oversized paragraphs into overlapping windows."""
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap < max_chars.")
    text = normalize_whitespace(text)
    chunks: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        for start in range(0, len(paragraph), max_chars - overlap):
            end = start + max_chars
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(paragraph):
                break
    return chunks


def pdf_to_datapoints(
    path: Path,
    *,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    doc_id_prefix: str | None = None,
) -> list[Datapoint]:
    raw = read_pdf_text(path)
    prefix = doc_id_prefix or path.stem
    chunks = chunk_text(raw, max_chars=max_chars, overlap=overlap)
    out: list[Datapoint] = []
    for i, body in enumerate(chunks):
        out.append(
            Datapoint(
                doc_id=f"{prefix}#chunk{i}",
                body=body,
                meta={"source_pdf": str(path.resolve()), "chunk_index": i},
            )
        )
    return out
