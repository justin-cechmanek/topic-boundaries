"""Data sources: anything that produces a list[Datapoint].

Built-ins cover the common cases (JSONL, PDF, arXiv, CSV, a directory of text
files). A custom loader only needs a load() method.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from topic_boundaries.config import DEFAULT_CHUNK_MAX_CHARS, DEFAULT_CHUNK_OVERLAP
from topic_boundaries.documents import Datapoint, datapoint_from_record
from topic_boundaries.pdf_corpus import chunk_text, pdf_to_datapoints


def _parse_vector(value: Any) -> list[float]:
    """Parse a cell into a float list: JSON array, or comma/space separated."""
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    s = str(value).strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return [float(x) for x in s.replace(",", " ").split()]


def _record_from_row(
    row: dict[str, Any],
    *,
    id_col: str,
    body_col: str | None,
    meta_cols: Iterable[str] | None,
    vector_col: str | None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"doc_id": row[id_col]}
    if body_col is not None:
        rec["body"] = "" if row.get(body_col) is None else str(row[body_col])
    if vector_col is not None and row.get(vector_col) not in (None, ""):
        rec["vector"] = _parse_vector(row[vector_col])
    for m in meta_cols or []:
        rec[m] = row.get(m)
    return rec


class JsonlSource:
    """One JSON object per line (arXiv/collect_datasets schema or generic)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[Datapoint]:
        from topic_boundaries.documents import load_jsonl

        return load_jsonl(self.path)


class PdfSource:
    """One PDF, a directory of PDFs (glob), or an explicit list of PDFs.

    Datapoints from every file are concatenated; each keeps its own
    ``source_pdf`` meta (set by ``pdf_to_datapoints``). For a directory the
    doc_id prefix is the file's path relative to the directory so same-named
    PDFs in different subdirs don't collide; a single file / list uses the stem,
    with numeric suffixes for repeated stems in a list.
    """

    def __init__(
        self,
        path: str | Path | Iterable[str | Path],
        *,
        glob: str = "**/*.pdf",
        max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.path = path
        self.glob = glob
        self.max_chars = max_chars
        self.overlap = overlap

    def _files(self) -> list[tuple[Path, str]]:
        """(file, doc_id_prefix) pairs to load, in stable order."""
        if isinstance(self.path, (list, tuple)):  # explicit list of paths
            files: list[tuple[Path, str]] = []
            used: set[str] = set()
            for item in self.path:
                file = Path(item)
                prefix = file.stem
                suffix = 2
                while prefix in used:
                    prefix = f"{file.stem}-{suffix}"
                    suffix += 1
                used.add(prefix)
                files.append((file, prefix))
            return files
        p = Path(self.path)
        if p.is_dir():
            return [
                (f, str(f.relative_to(p).with_suffix("")))
                for f in sorted(p.glob(self.glob))
                if f.is_file()
            ]
        return [(p, p.stem)]

    def load(self) -> list[Datapoint]:
        out: list[Datapoint] = []
        for file, prefix in self._files():
            out.extend(
                pdf_to_datapoints(
                    file,
                    max_chars=self.max_chars,
                    overlap=self.overlap,
                    doc_id_prefix=prefix,
                )
            )
        return out


class ArxivSource:
    """Search arXiv and load up to ``max_results`` abstract records."""

    def __init__(self, query: str, *, max_results: int = 100):
        self.query = query
        self.max_results = max_results

    def load(self) -> list[Datapoint]:
        from topic_boundaries.collect_datasets import harvest

        return [datapoint_from_record(r) for r in harvest(self.query, self.max_results)]


class ArxivPaperSource:
    """A single arXiv paper by id or URL (not a search query).

    Default: one Datapoint from the abstract + metadata. With ``fetch_pdf``,
    downloads the PDF and chunks it via ``pdf_to_datapoints`` instead.
    """

    def __init__(
        self,
        id_or_url: str,
        *,
        fetch_pdf: bool = False,
        max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.id_or_url = id_or_url
        self.fetch_pdf = fetch_pdf
        self.max_chars = max_chars
        self.overlap = overlap

    def load(self) -> list[Datapoint]:
        import tempfile

        from topic_boundaries.collect_datasets import (
            download_pdf,
            fetch_paper,
            normalize_arxiv_id,
        )

        arxiv_id = normalize_arxiv_id(self.id_or_url)
        if not self.fetch_pdf:
            return [datapoint_from_record(fetch_paper(self.id_or_url))]
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / f"{arxiv_id.replace('/', '_')}.pdf"
            pdf = download_pdf(self.id_or_url, dest)
            return pdf_to_datapoints(
                pdf,
                max_chars=self.max_chars,
                overlap=self.overlap,
                doc_id_prefix=arxiv_id,
            )


class RecordsSource:
    """Any iterable of dicts mapped to Datapoints.

    The id/body/meta/vector column mapping is the same one CsvSource uses; this
    just drops the CSV parsing so in-memory dicts or a parsed JSON array work
    too. Values may be native (lists, numbers) — not only strings.
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        *,
        id_col: str,
        body_col: str | None = None,
        meta_cols: Iterable[str] | None = None,
        vector_col: str | None = None,
    ):
        self.records = records
        self.id_col = id_col
        self.body_col = body_col
        self.meta_cols = list(meta_cols) if meta_cols else None
        self.vector_col = vector_col

    def load(self) -> list[Datapoint]:
        return [
            datapoint_from_record(
                _record_from_row(
                    row,
                    id_col=self.id_col,
                    body_col=self.body_col,
                    meta_cols=self.meta_cols,
                    vector_col=self.vector_col,
                )
            )
            for row in self.records
        ]


class CsvSource:
    """CSV rows mapped to Datapoints (stdlib csv; no pandas)."""

    def __init__(
        self,
        path: str | Path,
        *,
        id_col: str,
        body_col: str | None = None,
        meta_cols: Iterable[str] | None = None,
        vector_col: str | None = None,
    ):
        self.path = Path(path)
        self.id_col = id_col
        self.body_col = body_col
        self.meta_cols = list(meta_cols) if meta_cols else None
        self.vector_col = vector_col

    def load(self) -> list[Datapoint]:
        with self.path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return RecordsSource(
            rows,
            id_col=self.id_col,
            body_col=self.body_col,
            meta_cols=self.meta_cols,
            vector_col=self.vector_col,
        ).load()


class JsonSource:
    """A JSON file holding an array of objects, mapped like CsvSource.

    (For line-delimited JSON with the arXiv/generic schema, use JsonlSource.)
    """

    def __init__(
        self,
        path: str | Path,
        *,
        id_col: str,
        body_col: str | None = None,
        meta_cols: Iterable[str] | None = None,
        vector_col: str | None = None,
    ):
        self.path = Path(path)
        self.id_col = id_col
        self.body_col = body_col
        self.meta_cols = list(meta_cols) if meta_cols else None
        self.vector_col = vector_col

    def load(self) -> list[Datapoint]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{self.path} must contain a JSON array of objects.")
        return RecordsSource(
            data,
            id_col=self.id_col,
            body_col=self.body_col,
            meta_cols=self.meta_cols,
            vector_col=self.vector_col,
        ).load()


# Strip Markdown syntax that adds embedding noise: ATX heading markers (keep the
# heading text) and code-fence marker lines (keep the code text between them).
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_MD_FENCE = re.compile(r"^\s*(?:```|~~~).*$", re.MULTILINE)


def _read_text(file: Path) -> str:
    text = file.read_text(encoding="utf-8", errors="replace")
    if file.suffix.lower() in (".md", ".markdown"):
        text = _MD_FENCE.sub("", text)
        text = _MD_HEADING.sub("", text)
    return text


def _text_datapoints(
    file: Path, prefix: str, *, max_chars: int, overlap: int
) -> list[Datapoint]:
    """Chunk one .txt/.md file into Datapoints (shared by dir + single-file loaders)."""
    return [
        Datapoint(
            doc_id=f"{prefix}#chunk{i}",
            body=chunk,
            meta={"source_file": str(file.resolve()), "chunk_index": i},
        )
        for i, chunk in enumerate(
            chunk_text(_read_text(file), max_chars=max_chars, overlap=overlap)
        )
    ]


class TextDirSource:
    """A directory of .txt/.md files; each file is chunked like the PDF loader.

    ``glob`` accepts one pattern or several; the default covers both text and
    Markdown. Markdown headings/code fences are normalized before chunking.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        glob: str | Iterable[str] = ("**/*.txt", "**/*.md"),
        max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.path = Path(path)
        self.glob = glob
        self.max_chars = max_chars
        self.overlap = overlap

    def load(self) -> list[Datapoint]:
        patterns = [self.glob] if isinstance(self.glob, str) else list(self.glob)
        files = sorted({f for pat in patterns for f in self.path.glob(pat)})
        out: list[Datapoint] = []
        for file in files:
            if not file.is_file():
                continue
            rel = file.relative_to(self.path)
            out.extend(
                _text_datapoints(
                    file, str(rel), max_chars=self.max_chars, overlap=self.overlap
                )
            )
        return out


class TextFileSource:
    """A single .txt/.md file, chunked like TextDirSource (which is dir-only)."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.path = Path(path)
        self.max_chars = max_chars
        self.overlap = overlap

    def load(self) -> list[Datapoint]:
        return _text_datapoints(
            self.path, self.path.stem, max_chars=self.max_chars, overlap=self.overlap
        )
