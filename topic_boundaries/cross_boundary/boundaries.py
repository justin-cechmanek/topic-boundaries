from __future__ import annotations

from dataclasses import dataclass

from redisvl.query import VectorQuery
from redisvl.query.filter import Tag


@dataclass
class CrossBoundaryHit:
    source_cluster_id: int
    neighbor_doc_id: str
    neighbor_body: str
    neighbor_cluster_id: str
    distance: float


def nearest_outside_cluster(
    index,
    source_cluster_id: int,
    centroid: list[float],
    k: int,
    vector_field: str = "embedding",
) -> list[CrossBoundaryHit]:
    """
    Nearest vectors to `centroid` whose cluster tag differs from `source_cluster_id`.
    """
    flt = Tag("cluster_id") != str(source_cluster_id)
    q = VectorQuery(
        vector=list(centroid),
        vector_field_name=vector_field,
        num_results=k,
        return_fields=["doc_id", "body", "cluster_id"],
        filter_expression=flt,
        hybrid_policy="ADHOC_BF",
    )
    rows = index.query(q)
    hits: list[CrossBoundaryHit] = []
    for row in rows:
        dist_raw = row.get("vector_distance")
        dist = float(dist_raw) if dist_raw is not None else float("nan")
        hits.append(
            CrossBoundaryHit(
                source_cluster_id=source_cluster_id,
                neighbor_doc_id=str(row["doc_id"]),
                neighbor_body=str(row.get("body", "")),
                neighbor_cluster_id=str(row.get("cluster_id", "")),
                distance=dist,
            )
        )
    return hits


def cross_boundary_hits_for_all_clusters(
    index,
    centroids,
    n_clusters: int,
    k_per_centroid: int,
) -> list[CrossBoundaryHit]:
    all_hits: list[CrossBoundaryHit] = []
    for c in range(n_clusters):
        centroid = centroids[c]
        centroid_list = centroid.tolist() if hasattr(centroid, "tolist") else list(centroid)
        all_hits.extend(
            nearest_outside_cluster(index, c, centroid_list, k=k_per_centroid)
        )
    return all_hits
