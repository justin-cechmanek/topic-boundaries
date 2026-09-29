# API Reference

This page lists the main public entry points. Import them from `topic_boundaries` unless a module is shown.

## Data and embedding

| Name | Purpose |
| --- | --- |
| `Datapoint` | One document or vector, with `doc_id`, `body`, `meta`, and `vector` fields |
| `JsonlSource(path).load()` | Read line-delimited JSON into datapoints |
| `CsvSource(path, id_col=..., ...).load()` | Map CSV columns into datapoints |
| `HFTextEmbedder(model, batch_size=64)` | Embed text with the configured model |
| `PrecomputedEmbedder()` | Pass through existing vectors |

Other loaders are listed in [Data Sources](../user_guide/data_sources.md). `Embedder` is a structural protocol with an `embed(datapoints)` method and a `dims` property.

## Pipeline

`run_pipeline(datapoints, *, redis_url, n_clusters, embedding_model=None, embedder=None, schema_path, overwrite_index, ...)` embeds or passes through vectors, runs K-means, loads the Redis index, and returns a `PipelineState`.

| `PipelineState` field | Value |
| --- | --- |
| `datapoints` | Input list |
| `vectors` | Float32 matrix, one row per datapoint |
| `labels` | Cluster label for each row |
| `centroids` | One centroid vector per cluster |
| `indexed.index` | Connected RedisVL `SearchIndex` |
| `indexed.vector_dim`, `indexed.n_clusters` | Index dimensions and cluster count |

`embed_and_cluster_datapoints(...)` returns `(vectors, labels, centroids, vector_dim)` without connecting to Redis.

## Boundary functions

| Function | Input and result |
| --- | --- |
| `max_distance_rankings(index, centroids, n_clusters, top_n_per_cluster=None)` | Farthest-first `BoundaryHit` list |
| `cross_boundary_hits_for_all_clusters(index, centroids, n_clusters, k_per_centroid)` | Nearby other-cluster `CrossBoundaryHit` list |
| `boundary_doc_indices_per_cluster(labels, vectors, n_clusters, **kwargs)` | Convex hull indices by cluster |
| `voronoi_rankings(datapoints, vectors, labels, centroids, n_clusters, top_n_per_cluster=None)` | Voronoi `VoronoiBoundaryHit` list |
| `nearest_to_centroid(index, centroid, cluster_id, vector_field="embedding", k=10)` | Redis rows nearest the cluster centroid |

The hit classes preserve the score and identity relevant to each method. See [Boundary Methods](../concepts/boundary_methods.md) for score meanings and tradeoffs.
