# Architecture

## Pipeline

1. A source returns a list of `Datapoint` objects. Each has a `doc_id` and either text in `body` or a precomputed `vector`.
2. `HFTextEmbedder` turns text into normalized float32 vectors. `PrecomputedEmbedder` passes through supplied vectors.
3. K-means assigns each vector a cluster label and computes cluster centroids.
4. The pipeline writes the document ID, body, cluster tag, and vector to a RedisVL search index.
5. A boundary method uses those arrays or the index to identify documents of interest.

`run_pipeline` returns a `PipelineState` containing the datapoints, vectors, labels, centroids, and an `IndexedCorpus` with the Redis index.

## Index schema

The bundled `topic_boundaries/schema.yml` defines an HNSW vector index with a `cluster_id` tag and cosine distance. The pipeline adjusts the vector field's dimensions to match the embedder. Pass `schema_path` to `run_pipeline` to load a different schema. Redis uses `L2` as the schema value for Euclidean distance.

The distance metric affects the meaning of a score. For example, `max_distance_rankings` reads the configured metric: bounded cosine queries can search from the negated centroid, while Euclidean queries rank all matching cluster vectors and return Euclidean distances.

## What the methods share

All methods start from the same cluster labels and centroids. Max distance sort and cross-boundary search query Redis; convex hull and Voronoi methods work from the in-memory vector matrix. The methods return different forms of evidence, so their ranks are best compared within a method rather than treated as one universal score.
