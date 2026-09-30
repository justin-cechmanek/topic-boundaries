# API Reference

The [Python reference](reference.rst) renders signatures and docstrings for
every name exported by `topic_boundaries`. [Getting Started](../user_guide/getting_started.md)
shows how to use the main functions together.

## Data flow

1. A source's `load()` returns `list[Datapoint]`. Each datapoint needs a unique
   `doc_id` before indexing, plus text in `body` or a precomputed `vector`.
2. `embed_and_cluster_datapoints` returns `(vectors, labels, centroids)` with
   shapes `(n, d)`, `(n,)`, and `(n_clusters, d)`. Supply `embedding_model` for
   text or `embedder=PrecomputedEmbedder()` for existing vectors.
3. `index_clustered_datapoints` loads aligned datapoints, vectors, and labels
   into Redis and returns a connected `SearchIndex`. It requires an `embedding`
   vector field and `cluster_id` tag in the schema. `overwrite_index=True`
   replaces the configured index and its stored documents.
4. Pass the returned arrays or index to a boundary method.

K-means and Voronoi use Euclidean geometry regardless of the Redis metric.
The default Redis schema uses cosine distance. A custom schema can use L2;
the indexing function replaces its vector dimensions with the input width.

## Results and requirements

| Method | Redis? | Score or result |
| --- | --- | --- |
| `max_distance_rankings` | Yes | Cosine distance or Euclidean distance for L2; farthest first |
| `cross_boundary_hits_for_all_clusters` | Yes | Cosine distance or Euclidean distance for L2; nearest first |
| `boundary_doc_indices_per_cluster` | No | Global datapoint indices on a PCA-space hull |
| `voronoi_rankings` | No | Own-centroid distance / nearest-other-centroid distance; near 1 means near the K-means boundary |
| `nearest_to_centroid` | Yes | Raw Redis query rows and configured vector score |

`max_distance_rankings` returns `BoundaryHit`, cross-boundary search returns
`CrossBoundaryHit`, and `voronoi_rankings` returns `VoronoiBoundaryHit`.
Bounded cosine max-distance queries use approximate HNSW retrieval. Unbounded
rankings and L2 max-distance rankings scan each cluster. Convex hull falls back
to two first-PC extremes for degenerate clusters; Voronoi needs at least two
clusters. See [Boundary Methods](../concepts/boundary_methods.md) for tradeoffs.

Indexing rejects duplicate or blank IDs, malformed vectors, and invalid
labels with `ValueError` before Redis is contacted. `PrecomputedEmbedder`
requires every datapoint to have a vector of the same width.

```{toctree}
:maxdepth: 1

reference
```
