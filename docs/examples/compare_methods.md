# Compare boundary methods

Each method asks a different geometric question. This script runs all four on
one set of clusters. A method's top result is a candidate to inspect, not a
claim that it is the most novel document.

```python
from topic_boundaries import (
    JsonlSource,
    Settings,
    boundary_doc_indices_per_cluster,
    cross_boundary_hits_for_all_clusters,
    embed_and_cluster_datapoints,
    index_clustered_datapoints,
    max_distance_rankings,
    voronoi_rankings,
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
    datapoints, vectors, labels,
    redis_url=settings.redis_url,
    schema_path=str(settings.schema_path),
    overwrite_index=True,
)

farthest = max_distance_rankings(index, centroids, 5, top_n_per_cluster=2)
cross = cross_boundary_hits_for_all_clusters(index, centroids, 5, k_per_centroid=2)
hull = boundary_doc_indices_per_cluster(labels, vectors, 5)
voronoi = voronoi_rankings(datapoints, vectors, labels, centroids, 5, top_n_per_cluster=2)

print("farthest:", [(h.cluster_id, h.doc_id) for h in farthest[:2]])
print("cross:", [(h.source_cluster_id, h.neighbor_doc_id) for h in cross[:2]])
print("hull:", [(c, [datapoints[i].doc_id for i in indices[:2]]) for c, indices in hull.items()])
print("voronoi:", [(h.cluster_id, h.doc_id) for h in voronoi[:2]])
```

Max-distance and cross-boundary query Redis; hull and Voronoi use arrays in
memory. The hull returns global datapoint indices rather than hit objects.
Scores differ in meaning across methods; compare ranks within a method, not
their raw numbers across methods.
