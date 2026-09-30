# Architecture

## Pipeline

1. A source returns a list of `Datapoint` objects. Each has a `doc_id` and either text in `body` or a precomputed `vector`.
2. `HFTextEmbedder` turns text into normalized float32 vectors. `PrecomputedEmbedder` passes through supplied vectors.
3. K-means assigns each vector a cluster label and computes cluster centroids.
4. `index_clustered_datapoints` writes the document ID, body, cluster tag, and vector to a RedisVL search index.
5. A boundary method uses those arrays or the index to identify documents of interest.

`embed_and_cluster_datapoints` returns the vectors, labels, and centroids. Pass the datapoints, vectors, and labels to `index_clustered_datapoints` when a boundary method needs Redis; it returns the search index.

## Index schema

The bundled `topic_boundaries/schema.yml` defines an HNSW vector index with a `cluster_id` tag and cosine distance. Indexing adjusts the vector field's dimensions to match the vectors. Pass `schema_path` to `index_clustered_datapoints` to load a different schema, including one with Euclidean distance.

The distance metric affects the meaning of a score. For example, `max_distance_rankings` reads the configured metric: bounded cosine queries can search from the negated centroid, while Euclidean queries rank all matching cluster vectors and return Euclidean distances.

## What the methods share

All methods start from the same cluster labels and centroids. Max distance sort and cross-boundary search query Redis; convex hull and Voronoi methods work from the in-memory vector matrix. The methods return different forms of evidence, so their ranks are best compared within a method rather than treated as one universal score.
