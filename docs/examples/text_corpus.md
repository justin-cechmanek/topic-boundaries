# Rank a text corpus

Install `pip install -e ".[dev]"` and start the local Redis Stack. The first
run may download the configured sentence-transformers model. This example
uses the illustrative 20-record corpus and prints three farthest documents in
each cluster.

```python
from topic_boundaries import (
    JsonlSource,
    Settings,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
    max_distance_rankings,
)

settings = Settings.from_config()
datapoints = JsonlSource("datasets/sample.jsonl").load()
vectors, labels, centroids = embed_and_cluster_datapoints(
    datapoints,
    n_clusters=5,
    embedding_model=settings.embedding_model,
    kmeans_random_state=settings.kmeans_random_state,
    kmeans_n_init=settings.kmeans_n_init,
)
index = index_clustered_datapoints(
    datapoints,
    vectors,
    labels,
    redis_url=settings.redis_url,
    schema_path=str(settings.schema_path),
    overwrite_index=True,
)
hits = max_distance_rankings(index, centroids, n_clusters=5, top_n_per_cluster=3)
for hit in hits:
    print(hit.cluster_id, hit.rank_in_cluster, hit.doc_id, hit.distance)
```

`labels[i]` is the cluster assigned to `datapoints[i]`. A `BoundaryHit` gives
the document ID, text, distance, and rank within its cluster. The default
schema uses cosine distance. `overwrite_index=True` replaces the configured
index and its documents, so use a dedicated schema and index name for data
you need to keep.
