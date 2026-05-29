from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.documents import Datapoint


@dataclass
class VoronoiBoundaryHit:
    cluster_id: int
    doc_id: str
    body: str
    voronoi_ratio: float
    nearest_other_cluster_id: int
    rank_in_cluster: int


def voronoi_boundary_ratio(
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every point compute its Voronoi boundary proximity ratio.

    ratio = dist_to_own_centroid / dist_to_nearest_other_centroid

    A ratio close to 1.0 means the point is near the Voronoi boundary between
    its cluster and a neighbouring cluster. A ratio close to 0.0 means it sits
    deep inside its own cluster.

    Returns:
        ratios:               shape (n_samples,) float32
        nearest_other:        shape (n_samples,) int64  — index of the closest
                              centroid that is NOT the point's own cluster
    """
    # Pairwise squared distances: (n_samples, n_clusters)
    # ||v - c||^2 = ||v||^2 - 2 v·c + ||c||^2
    sq_norms_v = np.einsum("ij,ij->i", vectors, vectors)[:, None]   # (n, 1)
    sq_norms_c = np.einsum("ij,ij->i", centroids, centroids)[None, :]  # (1, k)
    dot = vectors @ centroids.T                                        # (n, k)
    dists_sq = np.maximum(sq_norms_v - 2.0 * dot + sq_norms_c, 0.0)
    dists = np.sqrt(dists_sq).astype(np.float32)

    dist_own = dists[np.arange(len(labels)), labels]

    # Mask own cluster so argmin finds the nearest *other* centroid.
    masked = dists.copy()
    masked[np.arange(len(labels)), labels] = np.inf
    nearest_other = np.argmin(masked, axis=1).astype(np.int64)
    dist_other = dists[np.arange(len(labels)), nearest_other]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(dist_other > 0.0, dist_own / dist_other, 0.0).astype(np.float32)

    return ratios, nearest_other


def boundary_rankings_for_all_clusters(
    datapoints: list[Datapoint],
    vectors: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
    top_n_per_cluster: int | None = None,
) -> list[VoronoiBoundaryHit]:
    """
    Rank points within each cluster by Voronoi boundary proximity (ratio closest to 1.0).

    Args:
        datapoints:         list of Datapoint objects (must have .doc_id and .body)
        vectors:            embedding matrix, shape (n_samples, dim)
        labels:             cluster assignment per point, shape (n_samples,)
        centroids:          cluster centroids, shape (n_clusters, dim)
        n_clusters:         number of clusters
        top_n_per_cluster:  if set, return at most this many hits per cluster
    """
    ratios, nearest_other = voronoi_boundary_ratio(vectors, labels, centroids)

    all_hits: list[VoronoiBoundaryHit] = []
    for c in range(n_clusters):
        mask = np.flatnonzero(labels == c)
        if mask.size == 0:
            continue

        cluster_ratios = ratios[mask]
        # Descending sort: ratio nearest 1.0 (boundary) first.
        order = np.argsort(-cluster_ratios)
        sorted_idx = mask[order]

        if top_n_per_cluster is not None:
            sorted_idx = sorted_idx[:top_n_per_cluster]

        for rank, global_idx in enumerate(sorted_idx):
            dp = datapoints[global_idx]
            all_hits.append(
                VoronoiBoundaryHit(
                    cluster_id=c,
                    doc_id=str(dp.doc_id),
                    body=str(dp.body),
                    voronoi_ratio=float(ratios[global_idx]),
                    nearest_other_cluster_id=int(nearest_other[global_idx]),
                    rank_in_cluster=rank,
                )
            )

    return all_hits
