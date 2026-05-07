from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Datapoint:
    doc_id: str
    body: str
    meta: dict[str, Any]


def load_jsonl(path: Path) -> list[Datapoint]:
    rows: list[Datapoint] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(datapoint_from_record(json.loads(line)))
    return rows


def datapoint_from_record(rec: dict[str, Any]) -> Datapoint:
    """Normalize JSON lines for embedding.

    Prefer ``abstract`` when present (arXiv / ``collect_datasets`` schema) so vectors
    match paper abstracts; fall back to ``body`` for generic JSONL.
    """
    if "abstract" in rec:
        body = str(rec["abstract"])
        doc_id = str(rec.get("arxive_link") or rec.get("title") or rec.get("doc_id") or "")
    elif "body" in rec:
        body = str(rec["body"])
        doc_id = str(rec.get("doc_id") or rec.get("id") or rec.get("arxive_link") or "")
    else:
        raise ValueError(
            "Each record needs an 'abstract' or 'body' field for embedding text."
        )
    if not doc_id:
        raise ValueError("Could not infer doc_id; add doc_id or arxiv_link or title.")
    meta = {k: v for k, v in rec.items() if k not in ("body", "abstract")}
    return Datapoint(doc_id=doc_id, body=body, meta=meta)
