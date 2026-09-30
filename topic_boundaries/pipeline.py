from __future__ import annotations

import numpy as np

from topic_boundaries.clustering import cluster_embeddings
from topic_boundaries.config import DEFAULT_EMBED_BATCH_SIZE
from topic_boundaries.documents import Datapoint
from topic_boundaries.embedding import Embedder, HFTextEmbedder


def _resolve_embedder(
    embedder: Embedder | None, embedding_model: str | None, batch_size: int
) -> Embedder:
    if embedder is not None:
        return embedder
    if embedding_model is None:
        raise ValueError("Provide either `embedder` or `embedding_model`.")
    return HFTextEmbedder(embedding_model, batch_size=batch_size)


def embed_and_cluster_datapoints(
    datapoints: list[Datapoint],
    *,
    n_clusters: int,
    embedding_model: str | None = None,
    embedder: Embedder | None = None,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    kmeans_random_state: int = 42,
    kmeans_n_init: int | str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vectors, labels, centroids)`` without connecting to Redis.

    Supply ``embedder`` (for example, ``PrecomputedEmbedder``) or an
    ``embedding_model`` for text. The arrays have shapes ``(n, d)``, ``(n,)``,
    and ``(n_clusters, d)``. Raises ``ValueError`` when neither embedder is
    supplied and ``RuntimeError`` if K-means leaves a cluster empty.
    """
    embedder = _resolve_embedder(embedder, embedding_model, embed_batch_size)
    vectors = np.asarray(embedder.embed(datapoints), dtype=np.float32)
    labels, centroids = cluster_embeddings(
        vectors,
        n_clusters,
        random_state=kmeans_random_state,
        n_init=kmeans_n_init,
    )
    counts = np.bincount(labels, minlength=n_clusters)
    if (counts == 0).any():
        raise RuntimeError(
            "Some clusters are empty; decrease n_clusters or increase dataset size."
        )
    return vectors, labels, centroids
