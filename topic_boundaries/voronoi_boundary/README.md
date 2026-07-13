# Voronoi Boundary
The __Voronoi boundary__ is a method of finding topic boundaries that uses the approach outlined below:

1. Perform K-means clustering to segment data and identify cluster centroids as outlined in the main repository README.
2. For each data point compute two distances:
    2.a. Distance to its own cluster centroid.
    2.b. Distance to the nearest centroid belonging to a different cluster.
3. Compute the Voronoi boundary proximity ratio: `dist_to_own_centroid / dist_to_nearest_other_centroid`. A ratio close to 1.0 indicates the point lies near the Voronoi boundary between its cluster and a neighbouring cluster. A ratio close to 0.0 indicates the point sits deep inside its own cluster.
4. Rank points within each cluster by this ratio in descending order; the top-ranked points are the boundary items.

This is computed entirely in numpy over the embedding matrix and centroid array — no Redis query is required beyond what the pipeline already produces.

Advantages:
- mathematically well-grounded: the Voronoi boundary is the exact decision boundary implied by K-means, so points near it are the truest boundary items by definition
- identifies which neighbouring cluster each boundary point is closest to, enabling directed inter-cluster analysis
- computationally cheap: a single vectorised pairwise distance pass over all points and centroids

Disadvantages:
- operates on the full embedding space rather than using the indexed Redis vectors, so results can diverge slightly if embeddings are normalised differently at index time
- like max distance sort, it is centroid-centric and may miss boundary structure in non-spherical or irregularly shaped clusters
