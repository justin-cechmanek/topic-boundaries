from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Record keys that carry a precomputed embedding, in preference order.
_VECTOR_KEYS = ("vector", "embedding")


@dataclass
class Datapoint:
    doc_id: str
    body: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # Optional precomputed embedding. When set, PrecomputedEmbedder uses it and
    # no text embedding happens; `body` may be empty for vector-only data.
    vector: np.ndarray | None = None


def load_jsonl(path: Path) -> list[Datapoint]:
    rows: list[Datapoint] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(datapoint_from_record(json.loads(line)))
    return rows


def _extract_vector(rec: dict[str, Any]) -> np.ndarray | None:
    for k in _VECTOR_KEYS:
        v = rec.get(k)
        if v is not None:
            return np.asarray(v, dtype=np.float32)
    return None


def datapoint_from_record(rec: dict[str, Any]) -> Datapoint:
    """Normalize a JSON record into a Datapoint.

    Prefer ``abstract`` when present (arXiv / ``collect_datasets`` schema) so
    vectors match paper abstracts; fall back to ``body`` for generic JSONL. A
    record with a precomputed ``vector``/``embedding`` needs no text at all.
    """
    vector = _extract_vector(rec)
    if "abstract" in rec:
        body = str(rec["abstract"])
        doc_id = str(
            rec.get("arxiv_link")
            or rec.get("arxive_link")
            or rec.get("title")
            or rec.get("doc_id")
            or ""
        )
    elif "body" in rec:
        body = str(rec["body"])
        doc_id = str(
            rec.get("doc_id")
            or rec.get("id")
            or rec.get("arxiv_link")
            or rec.get("arxive_link")
            or ""
        )
    elif vector is not None:
        body = ""  # vector-only record: nothing to embed
        doc_id = str(rec.get("doc_id") or rec.get("id") or rec.get("title") or "")
    else:
        raise ValueError(
            "Each record needs an 'abstract', 'body', or precomputed "
            "'vector'/'embedding' field."
        )
    if not doc_id:
        raise ValueError("Could not infer doc_id; add doc_id or arxiv_link or title.")
    meta = {k: v for k, v in rec.items() if k not in ("body", "abstract", *_VECTOR_KEYS)}
    return Datapoint(doc_id=doc_id, body=body, meta=meta, vector=vector)
