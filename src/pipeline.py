from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from redisvl.utils.vectorize import HFTextVectorizer

from src.clustering import cluster_counts, cluster_embeddings
from src.documents import Datapoint
from src.indexing import IndexedCorpus, create_and_load, open_index, records_for_redis


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
    embedding_model: str,
    schema_path: str | None,
    overwrite_index: bool,
    embed_batch_size: int = 64,
    kmeans_random_state: int = 42,
    kmeans_n_init: int | str = "auto",
) -> PipelineState:
    # Datapoint.body holds the string passed to the vectorizer (abstract for arXiv JSONL).
    texts = [d.body for d in datapoints]
    ids = [d.doc_id for d in datapoints]

    vectorizer = HFTextVectorizer(model=embedding_model)
    show_progress = len(texts) > embed_batch_size
    vectors = np.asarray(
        vectorizer.embed_many(
            contents=texts,
            batch_size=embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ),
        dtype=np.float32,
    )
    labels, centroids = cluster_embeddings(
        vectors,
        n_clusters,
        random_state=kmeans_random_state,
        n_init=kmeans_n_init,
    )

    counts = cluster_counts(labels, n_clusters)
    if (counts == 0).any():
        raise RuntimeError(
            "Some clusters are empty; decrease n_clusters or increase dataset size."
        )

    if vectorizer.dims is None:
        raise RuntimeError("Vectorizer has no dims; cannot build Redis schema.")
    vector_dim = vectorizer.dims
    index = open_index(schema_path, redis_url, vector_dim)
    records = records_for_redis(ids, texts, vectors, labels)
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
