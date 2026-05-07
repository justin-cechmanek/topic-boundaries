from __future__ import annotations

from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from src.max_distance_sort.boundaries import _cluster_doc_count


def nearest_to_centroid(
    index,
    centroid: list[float] | list,
    cluster_id: int,
    vector_field: str = "embedding",
    k: int = 10,
) -> list[dict]:
    """Documents closest to a centroid within the same cluster (topic core).

    Uses the same filtered exhaustive scoring as other methods (``ADHOC_BF``)
    and ``num_results`` equal to the cluster size so the top-``k`` neighbors are
    correct under the index metric.
    """
    n_docs = _cluster_doc_count(index, cluster_id)
    if n_docs == 0:
        return []
    flt = Tag("cluster_id") == str(cluster_id)
    q = VectorQuery(
        vector=list(centroid),
        vector_field_name=vector_field,
        num_results=n_docs,
        return_fields=["doc_id", "body", "cluster_id"],
        filter_expression=flt,
        hybrid_policy="ADHOC_BF",
    )
    rows = index.query(q)
    return rows[:k]
