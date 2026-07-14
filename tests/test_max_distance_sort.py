import numpy as np
import pytest

from topic_boundaries.max_distance_sort.boundaries import (
    boundary_rankings_for_all_clusters,
    furthest_from_centroid,
)


@pytest.mark.integration
def test_farthest_first_ranking(loaded_corpus):
    index, x = loaded_corpus
    hits = furthest_from_centroid(index, 0, np.array(x, dtype=np.float32))
    ids = [h.doc_id for h in hits]
    assert len(ids) == 3
    # c0_far is 90deg from the x-axis centroid -> farthest -> rank 0.
    assert ids[0] == "c0_far"
    assert ids[-1] == "c0_near"
    assert [h.rank_in_cluster for h in hits] == [0, 1, 2]


@pytest.mark.integration
def test_top_n_slices(loaded_corpus):
    index, x = loaded_corpus
    hits = furthest_from_centroid(index, 0, np.array(x, dtype=np.float32), top_n=1)
    assert [h.doc_id for h in hits] == ["c0_far"]


@pytest.mark.integration
def test_empty_cluster_returns_empty(loaded_corpus):
    index, x = loaded_corpus
    assert furthest_from_centroid(index, 42, np.array(x, dtype=np.float32)) == []


@pytest.mark.integration
def test_rankings_across_all_clusters(loaded_corpus):
    index, x = loaded_corpus
    centroids = np.array([x, x], dtype=np.float32)  # both clusters ranked vs x-axis
    hits = boundary_rankings_for_all_clusters(index, centroids, 2)
    assert {h.cluster_id for h in hits} == {0, 1}
    assert len(hits) == 5  # 3 in cluster 0 + 2 in cluster 1
