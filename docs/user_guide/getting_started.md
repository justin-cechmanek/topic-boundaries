# Getting Started

This walkthrough uses the 20 records in `datasets/sample.jsonl`. Run it from the repository root after [installing the package and starting Redis](installation.md).

## Load, cluster, and index records

```python
from topic_boundaries import (
    JsonlSource,
    Settings,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
)

settings = Settings.from_config()
datapoints = JsonlSource("datasets/sample.jsonl").load()
vectors, labels, centroids = embed_and_cluster_datapoints(
    datapoints,
    n_clusters=5,
    embedding_model=settings.embedding_model,
)
index = index_clustered_datapoints(
    datapoints,
    vectors,
    labels,
    redis_url=settings.redis_url,
    overwrite_index=True,
)
```

`labels[i]` is the cluster for `datapoints[i]`; `centroids` contains one vector per cluster. `overwrite_index=True` recreates the configured Redis index. Use a dedicated index name if you need to preserve an existing index.

## Rank boundary documents

```python
from topic_boundaries import max_distance_rankings

hits = max_distance_rankings(
    index,
    centroids,
    n_clusters=5,
    top_n_per_cluster=3,
)
for hit in hits:
    print(hit.cluster_id, hit.rank_in_cluster, hit.doc_id, hit.distance)
```

The result contains up to three hits per cluster, in farthest-first order within each cluster. A `BoundaryHit` also includes the document body.

## Try another method

The Voronoi method uses vectors already in memory:

```python
from topic_boundaries import voronoi_rankings

hits = voronoi_rankings(
    datapoints,
    vectors,
    labels,
    centroids,
    n_clusters=5,
    top_n_per_cluster=3,
)
```

Its `voronoi_ratio` measures proximity to a K-means boundary. See [Boundary Methods](../concepts/boundary_methods.md) to compare the techniques.
