import numpy as np
import pytest

from topic_boundaries.cross_boundary.boundaries import (
    cross_boundary_hits_for_all_clusters,
    nearest_outside_cluster,
)


@pytest.mark.integration
def test_excludes_source_cluster_and_orders_nearest_first(loaded_corpus):
    index, x = loaded_corpus
    hits = nearest_outside_cluster(index, source_cluster_id=0, centroid=x, k=5)
    ids = [h.neighbor_doc_id for h in hits]
    # Only cluster 1 docs may appear (Tag cluster_id != "0").
    assert set(ids) == {"c1_nearx", "c1_z"}
    assert all(h.neighbor_cluster_id != "0" for h in hits)
    # c1_nearx sits near the x-axis centroid -> nearest cross-cluster neighbor.
    assert ids[0] == "c1_nearx"


@pytest.mark.integration
def test_aggregate_across_clusters(loaded_corpus):
    index, x = loaded_corpus
    centroids = np.array([x, x], dtype=np.float32)
    hits = cross_boundary_hits_for_all_clusters(index, centroids, 2, k_per_centroid=5)
    # cluster 0 sees 2 outside docs, cluster 1 sees 3 outside docs.
    assert len(hits) == 5
    assert {h.source_cluster_id for h in hits} == {0, 1}
