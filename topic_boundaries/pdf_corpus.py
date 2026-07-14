"""
Extract text from PDFs and split into datapoints for embedding.

The paper proposal mentions LangChain's PyPDFLoader and text splitters; this
module uses `pypdf` plus a small paragraph/chunk merge step to avoid that
dependency by default. Install with: pip install -e ".[corpus]"
"""

from __future__ import annotations

import re
from pathlib import Path

from topic_boundaries.documents import Datapoint


def _require_pypdf():
    try:
        from pypdf import PdfReader  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PDF support requires pypdf. Install with: pip install pypdf "
            'or pip install -e ".[corpus]"'
        ) from e


def read_pdf_text(path: Path) -> str:
    _require_pypdf()
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n\n".join(parts)


_WS = re.compile(r"[ \t\r\f\v]+")


def normalize_whitespace(text: str) -> str:
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def chunk_text(text: str, *, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Greedy chunking on paragraphs with optional overlap between chunks."""
    text = normalize_whitespace(text)
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        candidate = p if not buf else f"{buf}\n\n{p}"
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
        if len(p) <= max_chars:
            buf = p
            continue
        # oversized paragraph: hard-split
        start = 0
        while start < len(p):
            end = min(start + max_chars, len(p))
            piece = p[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(p):
                break  # reached the end; overlap step would emit duplicate tails
            start = max(end - overlap, start + 1)
        buf = ""
    if buf:
        chunks.append(buf)
    return chunks


def pdf_to_datapoints(
    path: Path,
    *,
    max_chars: int = 1500,
    overlap: int = 200,
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
