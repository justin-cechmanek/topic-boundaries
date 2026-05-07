# Topic Boundaries
This project allows you to identify the specific content at the edges of topics within a large corpus of text.

## What is a __topic boundary__?
a __topic boundary__  is the set of datapoints (ideas, statements, concepts) that exist at the periphery of our understanding of a specific field of knowledge. It is the edge - or boundary - of our knowldge of a subject. This is generally where new research and innovation is focused. If we consider a specific field of knowledge we can assume the centroid of that field is well explored, so it follows that the remaining interesting areas of future discovery and innovation exist further away. This repository is a collection of tools and methods that attempts to formally quantify and identify these boundaries using multiple different techniques.


## The process
Each different technique is outlined in its own sub directory within the __src__ directory. They all share common initial steps listed below.

1. choose a dataset to analyse
2. load the raw text data and parse the text into data points
3. embed the data points via vectorizer of your choice and load them into a RedisVL vector searc index
4. perform K-means clustering to identify topic groups, aka clusters
5. after clustering is complete, add a tag field to each data point in a topic cluster to formally group them
6. find the cluster centroids, which is the average of all the data points' vectors in the cluster (assuming clusters are approximately spherical)

## Specific techniques

### 1. Maximum distance sorting
For each topic set find the documents furthest from the centroid. Do this by collecting all docs and sorting by greatest distance to centroid.

### 2. Convex hull
Use (convex hull https://en.wikipedia.org/wiki/Convex_hull_algorithms)[https://en.wikipedia.org/wiki/Convex_hull_algorithms], specifically (Quickhull)[https://en.wikipedia.org/wiki/Quickhull] to identify data points within a cluster that exist on the periphery, or hull, of the collection of data points.

### 3. Cross boundary
Using the centroid of one specific cluster, find the nearest data points _not_ in the same cluster. These points exist on the boundary of their respective clusters. This query process must be repeated for each different centroid to to identify further data points on other clusters on other sections of their boundaries.

## Centroid analysis
Topic centroids are interesting on their own also. Finding the data points closest to the centroid to see what the topics are provides a natural grouping of research areas.  You can pass the data points closest to a centroid to an LLM and ask it to name and summarize what each topic is.
