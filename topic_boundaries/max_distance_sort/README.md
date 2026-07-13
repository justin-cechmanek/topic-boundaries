# Max Distance Sort
The __max distance sort__ is a method of finding topic boundaries that uses the approach outlined below:

1. Perform K-means clustering to segment data and identify cluster centroids as outlined in the main repository README.
2. For each cluster find all points furthest from the cluster centroid.
    2.a. Do this via vector similarity search with the cluster centroid as the query vector, the number of returned results equal to the number of data points in the cluster, and results sorted by furthest distance to the centroid.
    2.b. This uses a RedisVL VectorQuery with a FilterExpression that limits results to data points within the same cluster as the centroid search vector.

Advantages:
- it is simple and straightforward to implement

Disadvantages:
- depending on the shape of the cluster it may not identify all topics on the boundary, and may instead bias its results toward denser or more crowded areas of the boundary.
