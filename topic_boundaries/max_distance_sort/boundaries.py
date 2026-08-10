from __future__ import annotations

from dataclasses import dataclass

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
    """
    Within-cluster points ranked by greatest distance to the centroid.

    Distances match the index metric (cosine). With ``top_n`` set, we run a
    server-side "farthest" query: querying the HNSW index with the *negated*
    centroid returns the nearest neighbors of ``-centroid``, which are exactly
    the points farthest from ``centroid`` under cosine — so the top-``top_n``
    come back without pulling the whole cluster. The reported distance is the
    cosine distance to ``centroid``, recovered as ``2 - d(-centroid)``.
    (Cosine-only: ``cos(x, -c) = -cos(x, c)``.)

    With ``top_n=None`` the full cluster is ranked, which needs an exhaustive
    filtered scan (``ADHOC_BF``, nearest-first then reversed).
    """
    centroid_arr = np.asarray(centroid, dtype=np.float32)

    if top_n is not None:
        q = VectorQuery(
            vector=(-centroid_arr).tolist(),
            vector_field_name="embedding",
            num_results=top_n,
            return_fields=["doc_id", "body"],
            filter_expression=Tag("cluster_id") == str(cluster_id),
        )
        rows = index.query(q)  # already farthest-first (nearest to -centroid)
        return [
            BoundaryHit(
                cluster_id=cluster_id,
                doc_id=str(row["doc_id"]),
                body=str(row.get("body", "")),
                distance=2.0 - _parse_vector_distance(row),
                rank_in_cluster=rank,
            )
            for rank, row in enumerate(rows)
        ]

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
    return [
        BoundaryHit(
            cluster_id=cluster_id,
            doc_id=str(row["doc_id"]),
            body=str(row.get("body", "")),
            distance=_parse_vector_distance(row),
            rank_in_cluster=rank,
        )
        for rank, row in enumerate(rows)
    ]


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
