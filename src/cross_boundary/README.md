# Cross Boundary
The __cross boundary__ is a method of finding topic boundaries that uses the approach outlined below:

1. Perform K-means clustering to segment data and identify cluster centroids as outlined in the main repository README.
2. For each cluster centroid perform a VectorQuery search using the centroid vector as the query vector to find the nearest data points __not__ within the same cluster as the current centroid. Use a FilterExpression with a negation on the centroid tag to accomplish this.
3. Repeat this for each centroid to gradually build a list of data points on the boundary of there respective cluster. 


Advantages:
- it is relatively simple and straightforward to implement

Disadvantages:
- it is untested and does not currently have strong mathematical rigor to support this approach
