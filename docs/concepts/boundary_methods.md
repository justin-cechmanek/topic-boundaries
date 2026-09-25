# Boundary Methods

## Max Distance Sort

Ranks each cluster's documents from farthest to nearest relative to that cluster's centroid. Use it for a straightforward list of peripheral documents.

With a bounded `top_n_per_cluster` and a cosine index, the method queries nearest neighbors of the negated centroid. Cosine distance obeys `d(c, x) = 2 - d(-c, x)` for nonzero vectors. HNSW retrieval can be approximate. With L2 or another metric, or when no limit is set, the method scans all matching cluster vectors. L2 scores are returned as Euclidean distances.

## Cross Boundary

For each centroid, finds nearby documents tagged as belonging to *other* clusters. Use it to inspect relationships between topics. A hit identifies its source cluster and the neighboring document's cluster. The returned score is Redis's vector score for the configured index metric.

## Convex Hull

Finds vertices of a convex hull around each cluster after PCA compression. Use it when you want a geometric outline of a cluster. The default maximum hull dimension is 12. For degenerate clusters where Qhull cannot form a hull, the method logs a warning and returns the two extremes along the first principal component.

## Voronoi Boundary

Compares each point's Euclidean distance to its own centroid with its distance to the nearest other centroid. A ratio near 1 suggests the point lies near a K-means decision boundary. This method needs at least two clusters and runs on the in-memory vectors, without another Redis query.

## Centroid Neighbors

Finds documents closest to a centroid within its cluster. These are useful for describing a topic's core, alongside the boundary methods that find its edges.

See [API Reference](../api/index.md) for the callable names and return types.
