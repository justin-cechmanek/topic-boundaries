from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topic_boundaries.clustering import cluster_counts, cluster_embeddings
from topic_boundaries.documents import Datapoint
from topic_boundaries.embedding import Embedder, HFTextEmbedder
from topic_boundaries.indexing import IndexedCorpus, create_and_load, open_index, records_for_redis


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
    embed_batch_size: int = 64,
    kmeans_random_state: int = 42,
    kmeans_n_init: int | str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Embed datapoints and run KMeans without touching Redis (for analysis / viz).

    Supply your own `embedder` (e.g. PrecomputedEmbedder) or an `embedding_model`
    for the default text embedder.
    """
    embedder = _resolve_embedder(embedder, embedding_model, embed_batch_size)
    vectors = np.asarray(embedder.embed(datapoints), dtype=np.float32)
    labels, centroids = cluster_embeddings(
        vectors,
        n_clusters,
        random_state=kmeans_random_state,
        n_init=kmeans_n_init,
    )
    dim = embedder.dims if embedder.dims is not None else int(vectors.shape[1])
    return vectors, labels, centroids, int(dim)


@dataclass
class PipelineState:
    datapoints: list[Datapoint]
    vectors: np.ndarray
    labels: np.ndarray
    centroids: np.ndarray
    indexed: IndexedCorpus


def run_pipeline(
    datapoints: list[Datapoint],
    *,
    redis_url: str,
    n_clusters: int,
    embedding_model: str | None = None,
    embedder: Embedder | None = None,
    schema_path: str | None,
    overwrite_index: bool,
    embed_batch_size: int = 64,
    kmeans_random_state: int = 42,
    kmeans_n_init: int | str = "auto",
) -> PipelineState:
    ids = [d.doc_id for d in datapoints]
    vectors, labels, centroids, vector_dim = embed_and_cluster_datapoints(
        datapoints,
        n_clusters=n_clusters,
        embedding_model=embedding_model,
        embedder=embedder,
        embed_batch_size=embed_batch_size,
        kmeans_random_state=kmeans_random_state,
        kmeans_n_init=kmeans_n_init,
    )

    counts = cluster_counts(labels, n_clusters)
    if (counts == 0).any():
        raise RuntimeError(
            "Some clusters are empty; decrease n_clusters or increase dataset size."
        )

    index = open_index(schema_path, redis_url, vector_dim)
    records = records_for_redis(ids, [d.body for d in datapoints], vectors, labels)
    create_and_load(
        index,
        records,
        overwrite=overwrite_index,
        drop_keys=overwrite_index,
    )

    indexed = IndexedCorpus(
        index=index,
        vector_dim=vector_dim,
        n_clusters=n_clusters,
    )
    return PipelineState(
        datapoints=datapoints,
        vectors=vectors,
        labels=labels,
        centroids=centroids,
        indexed=indexed,
    )
