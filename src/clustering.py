from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_embeddings(
    vectors: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 42,
    n_init: int | str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        labels: shape (n_samples,) cluster index per row
        centroids: shape (n_clusters, dim) cluster centers
    """
    km = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
    )
    labels = km.fit_predict(vectors)
    return labels.astype(np.int64), km.cluster_centers_.astype(np.float32)


def cluster_counts(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_clusters)
    return counts
