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
    idx = SearchIndex(schema, redis_url=redis_url)
    # connect() does not use the constructor's redis_url unless passed explicitly
    idx.connect(redis_url=redis_url)
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
    """Create a Redis index and load datapoints with their vectors and cluster labels."""
    if (
        vectors.ndim != 2
        or labels.ndim != 1
        or not len(datapoints)
        or len(datapoints) != len(vectors)
        or len(datapoints) != len(labels)
    ):
        raise ValueError("Expected nonempty, aligned datapoints, 2-D vectors, and 1-D labels.")
    index = open_index(schema_path, redis_url, int(vectors.shape[1]))
    records = records_for_redis(
        [d.doc_id for d in datapoints],
        [d.body for d in datapoints],
        vectors,
        labels,
    )
    create_and_load(index, records, overwrite=overwrite_index, drop_keys=overwrite_index)
    return index
