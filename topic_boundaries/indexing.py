from __future__ import annotations

from typing import Any

import numpy as np
from redisvl.index import SearchIndex

from topic_boundaries.documents import Datapoint


def open_index(schema_path_str: str | None, redis_url: str, vector_dim: int) -> SearchIndex:
    from pathlib import Path

    from topic_boundaries.schema_builder import load_schema_for_dims

    schema_path = Path(__file__).resolve().parent / "schema.yml"
    if schema_path_str:
        schema_path = Path(schema_path_str)
    schema = load_schema_for_dims(schema_path, vector_dim)
    return SearchIndex(schema, redis_url=redis_url)


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
                # redis-py rejects list values in HSET; float32 blob for RediSearch vector
                "embedding": embeddings[i].astype(np.float32).tobytes(),
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


def index_clustered_datapoints(
    datapoints: list[Datapoint],
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    redis_url: str,
    overwrite_index: bool,
    schema_path: str | None = None,
) -> SearchIndex:
    """Load aligned datapoints, vectors, and labels into a RedisVL index.

    Returns a connected ``SearchIndex``. ``schema_path`` selects a custom
    schema; its ``embedding`` field dimensions are set from ``vectors``.
    ``overwrite_index=True`` replaces the index and its stored documents.
    Invalid IDs, vectors, or labels raise ``ValueError`` before connecting.
    """
    if (
        vectors.ndim != 2
        or labels.ndim != 1
        or not len(datapoints)
        or len(datapoints) != len(vectors)
        or len(datapoints) != len(labels)
        or vectors.shape[1] == 0
    ):
        raise ValueError("Expected nonempty, aligned datapoints, 2-D vectors, and 1-D labels.")
    doc_ids = [d.doc_id for d in datapoints]
    if (
        any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in doc_ids)
        or len(set(doc_ids)) != len(doc_ids)
    ):
        raise ValueError("Expected unique, nonempty doc_ids before indexing.")
    if not (
        np.issubdtype(vectors.dtype, np.floating)
        or np.issubdtype(vectors.dtype, np.integer)
    ) or not np.isfinite(vectors).all():
        raise ValueError("Expected finite numeric vectors before indexing.")
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError("Expected nonnegative integer cluster labels before indexing.")
    index = open_index(schema_path, redis_url, int(vectors.shape[1]))
    records = records_for_redis(
        doc_ids,
        [d.body for d in datapoints],
        vectors,
        labels,
    )
    create_and_load(index, records, overwrite=overwrite_index, drop_keys=overwrite_index)
    return index
