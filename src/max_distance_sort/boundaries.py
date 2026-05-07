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

    Uses RedisVL ``VectorQuery`` so distances match the index metric (cosine).
    Results are nearest-first; we reverse to rank farthest first. Requires a
    cluster document count so ``num_results`` covers the whole cluster
    (with ``hybrid_policy='ADHOC_BF'`` so filtered vectors are scored exhaustively).
    """
    n_docs = _cluster_doc_count(index, cluster_id)
    if n_docs == 0:
        return []

    centroid_list = np.asarray(centroid, dtype=np.float32).tolist()
    flt = Tag("cluster_id") == str(cluster_id)
    q = VectorQuery(
        vector=centroid_list,
        vector_field_name="embedding",
        num_results=n_docs,
        return_fields=["doc_id", "body"],
        filter_expression=flt,
        hybrid_policy="ADHOC_BF",
    )
    rows = index.query(q)
    # Nearest/lowest distance first from the server → farthest first for boundary ranking.
    rows = list(reversed(rows))
    if top_n is not None:
        rows = rows[:top_n]

    out: list[BoundaryHit] = []
    for rank, row in enumerate(rows):
        out.append(
            BoundaryHit(
                cluster_id=cluster_id,
                doc_id=str(row["doc_id"]),
                body=str(row.get("body", "")),
                distance=_parse_vector_distance(row),
                rank_in_cluster=rank,
            )
        )
    return out


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
