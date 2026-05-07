from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from redisvl.index import SearchIndex


@dataclass
class IndexedCorpus:
    index: SearchIndex
    vector_dim: int
    n_clusters: int


def open_index(schema_path_str: str | None, redis_url: str, vector_dim: int) -> SearchIndex:
    from pathlib import Path

    from src.schema_builder import load_schema_for_dims

    schema_path = Path(__file__).resolve().parent / "schema.yml"
    if schema_path_str:
        schema_path = Path(schema_path_str)
    schema = load_schema_for_dims(schema_path, vector_dim)
    idx = SearchIndex(schema, redis_url=redis_url)
    idx.connect()
    return idx


def records_for_redis(
    doc_ids: list[str],
    bodies: list[str],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(len(doc_ids)):
        out.append(
            {
                "doc_id": doc_ids[i],
                "body": bodies[i],
                "cluster_id": str(int(cluster_labels[i])),
                "embedding": embeddings[i].astype(np.float32).tolist(),
            }
        )
    return out


def create_and_load(
    index: SearchIndex,
    records: list[dict[str, Any]],
    *,
    overwrite: bool,
    drop_keys: bool,
    id_field: str = "doc_id",
    batch_size: int = 100,
) -> None:
    index.create(overwrite=overwrite, drop=drop_keys)
    index.load(records, id_field=id_field, batch_size=batch_size)
