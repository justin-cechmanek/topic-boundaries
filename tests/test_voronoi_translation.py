import numpy as np

from topic_boundaries.voronoi_boundary.boundaries import voronoi_boundary_ratio


def test_voronoi_scores_and_neighbors_are_preserved_by_translation():
    vectors = np.array([[1.0, 0.0], [0.25, 0.0], [2.0, 0.0]], dtype=np.float32)
    centroids = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 0, 1])
    expected_ratios, expected_neighbors = voronoi_boundary_ratio(vectors, labels, centroids)
    offset = np.array([10000.0, 10000.0], dtype=np.float32)
    ratios, neighbors = voronoi_boundary_ratio(vectors + offset, labels, centroids + offset)
    np.testing.assert_allclose(ratios, expected_ratios)
    np.testing.assert_array_equal(neighbors, expected_neighbors)
    assert ratios[0] == 1.0
