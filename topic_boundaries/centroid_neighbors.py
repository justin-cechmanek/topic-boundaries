from __future__ import annotations

from redisvl.query import VectorQuery
from redisvl.query.filter import Tag


def nearest_to_centroid(
    index,
    centroid: list[float] | list,
    cluster_id: int,
    vector_field: str = "embedding",
    k: int = 10,
) -> list[dict]:
    """Documents closest to a centroid within the same cluster (topic core).

    Server-side top-``k``: ``num_results=k`` lets the HNSW index return the
    nearest neighbors directly instead of pulling the whole cluster to the
    client and slicing. An empty cluster yields no rows.
    """
    flt = Tag("cluster_id") == str(cluster_id)
    q = VectorQuery(
        vector=list(centroid),
        vector_field_name=vector_field,
        num_results=k,
        return_fields=["doc_id", "body", "cluster_id"],
        filter_expression=flt,
    )
    return index.query(q)
