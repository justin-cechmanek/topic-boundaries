"""Data sources: anything that produces a list[Datapoint].

Built-ins cover the common cases (JSONL, PDF, arXiv, CSV, DataFrame, a directory
of text files). Implement the DataSource protocol for anything exotic, or call
``register_source`` to expose it to the CLI.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from topic_boundaries.documents import Datapoint, datapoint_from_record
from topic_boundaries.pdf_corpus import chunk_text, pdf_to_datapoints


@runtime_checkable
class DataSource(Protocol):
    def load(self) -> list[Datapoint]:
        ...


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
    def __init__(self, path: str | Path, *, max_chars: int = 1500, overlap: int = 200):
        self.path = Path(path)
        self.max_chars = max_chars
        self.overlap = overlap

    def load(self) -> list[Datapoint]:
        return pdf_to_datapoints(self.path, max_chars=self.max_chars, overlap=self.overlap)


class ArxivSource:
    def __init__(self, query: str, *, max_results: int = 100):
        self.query = query
        self.max_results = max_results

    def load(self) -> list[Datapoint]:
        from topic_boundaries.collect_datasets import harvest

        return [datapoint_from_record(r) for r in harvest(self.query, self.max_results)]


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
            for row in rows
        ]


class DataFrameSource:
    """A pandas (or duck-typed) DataFrame mapped to Datapoints; no pandas dep here."""

    def __init__(
        self,
        df: Any,
        *,
        id_col: str,
        body_col: str | None = None,
        meta_cols: Iterable[str] | None = None,
        vector_col: str | None = None,
    ):
        self.df = df
        self.id_col = id_col
        self.body_col = body_col
        self.meta_cols = list(meta_cols) if meta_cols else None
        self.vector_col = vector_col

    def load(self) -> list[Datapoint]:
        out: list[Datapoint] = []
        for _, row in self.df.iterrows():
            out.append(
                datapoint_from_record(
                    _record_from_row(
                        row,
                        id_col=self.id_col,
                        body_col=self.body_col,
                        meta_cols=self.meta_cols,
                        vector_col=self.vector_col,
                    )
                )
            )
        return out


class TextDirSource:
    """A directory of text files; each file is chunked like the PDF loader."""

    def __init__(
        self,
        path: str | Path,
        *,
        glob: str = "**/*.txt",
        max_chars: int = 1500,
        overlap: int = 200,
    ):
        self.path = Path(path)
        self.glob = glob
        self.max_chars = max_chars
        self.overlap = overlap

    def load(self) -> list[Datapoint]:
        out: list[Datapoint] = []
        for file in sorted(self.path.glob(self.glob)):
            if not file.is_file():
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
            rel = file.relative_to(self.path)
            for i, chunk in enumerate(
                chunk_text(text, max_chars=self.max_chars, overlap=self.overlap)
            ):
                out.append(
                    Datapoint(
                        doc_id=f"{rel}#chunk{i}",
                        body=chunk,
                        meta={"source_file": str(file.resolve()), "chunk_index": i},
                    )
                )
        return out


# Programmatic registry of built-in sources, for discovery and plug-ins:
# `SOURCES[name](...)` / `register_source("myfmt", MySource)`. The CLI uses its
# own typed flags (--csv, --text-dir, ...) rather than this map.
SOURCES: dict[str, Callable[..., DataSource]] = {
    "jsonl": JsonlSource,
    "pdf": PdfSource,
    "arxiv": ArxivSource,
    "csv": CsvSource,
    "textdir": TextDirSource,
}


def register_source(name: str, factory: Callable[..., DataSource]) -> None:
    """Register a DataSource factory under `name` (programmatic/plug-in use)."""
    SOURCES[name] = factory
