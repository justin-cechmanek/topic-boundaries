# Convex Hull
The __convex hull__ is a method of finding topic boundaries that uses the approach outlined below:

1. Perform K-means clustering to segment data and identify cluster centroids as outlined in the main repository README.
2. For each cluster identify the points on the convex hull of the data point vectors in higher dimensional vector space. For performance we use the Quickhull algorithm.


Advantages:
- it handles non-spherical clusters and will find a full boundary surrounding all sides of a topic

Disadvantages:
- it is computationally expensive and time consuming
