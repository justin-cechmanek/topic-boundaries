from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from redisvl.query import CountQuery, VectorQuery
from redisvl.query.filter import Tag


@dataclass
class BoundaryHit:
    cluster_id: int
    doc_id: str
    body: str
    distance: float
    rank_in_cluster: int


def _cluster_doc_count(index, cluster_id: int) -> int:
    q = CountQuery(filter_expression=Tag("cluster_id") == str(cluster_id))
    total = index.query(q)
    return int(total) if not isinstance(total, list) else len(total)


def _parse_vector_distance(row: dict) -> float:
    raw = row.get("vector_distance")
    if raw is None:
        return float("nan")
    return float(raw)


def furthest_from_centroid(
    index,
    cluster_id: int,
    centroid: np.ndarray,
    top_n: int | None = None,
) -> list[BoundaryHit]:
    """Rank within-cluster points by the index's distance metric.

    Bounded cosine queries use nearest neighbors of the negated centroid;
    other metrics and full rankings use an exhaustive filtered scan.
    Redis's squared L2 score is converted to Euclidean distance.
    """
    centroid_arr = np.asarray(centroid, dtype=np.float32)
    metric = index.schema.fields["embedding"].attrs.distance_metric.value
    cosine_top_n = top_n is not None and metric == "COSINE"

    if cosine_top_n:
        q = VectorQuery(
            vector=(-centroid_arr).tolist(),
            vector_field_name="embedding",
            num_results=top_n,
            return_fields=["doc_id", "body"],
            filter_expression=Tag("cluster_id") == str(cluster_id),
        )
        rows = index.query(q)  # already farthest-first (nearest to -centroid)
    else:
        n_docs = _cluster_doc_count(index, cluster_id)
        if n_docs == 0:
            return []
        q = VectorQuery(
            vector=centroid_arr.tolist(),
            vector_field_name="embedding",
            num_results=n_docs,
            return_fields=["doc_id", "body"],
            filter_expression=Tag("cluster_id") == str(cluster_id),
            hybrid_policy="ADHOC_BF",
        )
        # Nearest/lowest distance first from the server → farthest first for boundary ranking.
        rows = list(reversed(index.query(q)))
        if top_n is not None:
            rows = rows[:top_n]
    hits = []
    for rank, row in enumerate(rows):
        distance = _parse_vector_distance(row)
        if cosine_top_n:
            distance = 2.0 - distance
        elif metric == "L2":
            distance = sqrt(distance)
        hits.append(
            BoundaryHit(
                cluster_id=cluster_id,
                doc_id=str(row["doc_id"]),
                body=str(row.get("body", "")),
                distance=distance,
                rank_in_cluster=rank,
            )
        )
    return hits


def boundary_rankings_for_all_clusters(
    index,
    centroids: np.ndarray,
    n_clusters: int,
    top_n_per_cluster: int | None = None,
) -> list[BoundaryHit]:
    all_hits: list[BoundaryHit] = []
    for c in range(n_clusters):
        all_hits.extend(
            furthest_from_centroid(
                index,
                c,
                centroids[c],
                top_n=top_n_per_cluster,
            )
        )
    return all_hits
