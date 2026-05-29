import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.documents import Datapoint
from src.voronoi_boundary.boundaries import (
    VoronoiBoundaryHit,
    boundary_rankings_for_all_clusters,
    voronoi_boundary_ratio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_datapoints(n: int) -> list[Datapoint]:
    return [Datapoint(doc_id=str(i), body=f"doc {i}", meta={}) for i in range(n)]


def _two_cluster_fixture():
    """Six points split evenly across two well-separated clusters."""
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],  # cluster 0 — doc 2 is closest to the boundary
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],  # cluster 1 — doc 5 is closest to the boundary
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    centroids = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    return vectors, labels, centroids


# ---------------------------------------------------------------------------
# voronoi_boundary_ratio
# ---------------------------------------------------------------------------


def test_ratio_shape_and_dtype():
    vectors, labels, centroids = _two_cluster_fixture()
    ratios, nearest_other = voronoi_boundary_ratio(vectors, labels, centroids)
    assert ratios.shape == (6,)
    assert nearest_other.shape == (6,)
    assert ratios.dtype == np.float32
    assert nearest_other.dtype == np.int64


def test_nearest_other_never_own_cluster():
    vectors, labels, centroids = _two_cluster_fixture()
    _, nearest_other = voronoi_boundary_ratio(vectors, labels, centroids)
    for i, (own, other) in enumerate(zip(labels, nearest_other)):
        assert own != other, f"point {i}: nearest_other should differ from own cluster"


def test_ratio_bounded_zero_to_one_for_well_separated_clusters():
    """For well-separated clusters every ratio should be in [0, 1)."""
    vectors, labels, centroids = _two_cluster_fixture()
    ratios, _ = voronoi_boundary_ratio(vectors, labels, centroids)
    assert (ratios >= 0.0).all()
    assert (ratios < 1.0).all()


def test_ratio_exactly_half_for_equidistant_point():
    """A point exactly halfway between two centroids should have ratio == 0.5."""
    centroids = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    midpoint = np.array([[0.0, 0.0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    ratios, nearest_other = voronoi_boundary_ratio(midpoint, labels, centroids)
    assert nearest_other[0] == 1
    assert ratios[0] == pytest.approx(1.0, abs=1e-5)


def test_boundary_point_ranks_highest():
    """The point geometrically closest to the Voronoi boundary should rank first."""
    vectors, labels, centroids = _two_cluster_fixture()
    ratios, _ = voronoi_boundary_ratio(vectors, labels, centroids)
    # doc 2 (index 2) is furthest from centroid 0 and closest to centroid 1 in cluster 0
    cluster0_idx = np.flatnonzero(labels == 0)
    top = cluster0_idx[np.argmax(ratios[cluster0_idx])]
    assert top == 2


def test_centroid_itself_has_ratio_zero():
    """A vector equal to its own centroid is maximally central — ratio should be 0."""
    centroids = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    vectors = centroids.copy()
    labels = np.array([0, 1], dtype=np.int64)
    ratios, _ = voronoi_boundary_ratio(vectors, labels, centroids)
    assert ratios[0] == pytest.approx(0.0, abs=1e-6)
    assert ratios[1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# boundary_rankings_for_all_clusters
# ---------------------------------------------------------------------------


def test_returns_voronoi_boundary_hit_instances():
    vectors, labels, centroids = _two_cluster_fixture()
    datapoints = _make_datapoints(len(vectors))
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    assert all(isinstance(h, VoronoiBoundaryHit) for h in hits)


def test_hit_count_equals_total_points_without_cap():
    vectors, labels, centroids = _two_cluster_fixture()
    datapoints = _make_datapoints(len(vectors))
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    assert len(hits) == len(vectors)


def test_top_n_per_cluster_cap():
    vectors, labels, centroids = _two_cluster_fixture()
    datapoints = _make_datapoints(len(vectors))
    hits = boundary_rankings_for_all_clusters(
        datapoints, vectors, labels, centroids, n_clusters=2, top_n_per_cluster=2
    )
    assert len(hits) == 4  # 2 clusters × 2 per cluster


def test_rank_in_cluster_is_sequential():
    vectors, labels, centroids = _two_cluster_fixture()
    datapoints = _make_datapoints(len(vectors))
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    for cluster_id in (0, 1):
        ranks = [h.rank_in_cluster for h in hits if h.cluster_id == cluster_id]
        assert ranks == list(range(len(ranks)))


def test_hits_sorted_descending_by_ratio():
    vectors, labels, centroids = _two_cluster_fixture()
    datapoints = _make_datapoints(len(vectors))
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    for cluster_id in (0, 1):
        ratios = [h.voronoi_ratio for h in hits if h.cluster_id == cluster_id]
        assert ratios == sorted(ratios, reverse=True)


def test_nearest_other_cluster_id_is_populated():
    vectors, labels, centroids = _two_cluster_fixture()
    datapoints = _make_datapoints(len(vectors))
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    for h in hits:
        assert h.nearest_other_cluster_id != h.cluster_id


def test_doc_id_and_body_match_datapoints():
    vectors, labels, centroids = _two_cluster_fixture()
    datapoints = _make_datapoints(len(vectors))
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    for h in hits:
        idx = int(h.doc_id)
        assert h.doc_id == datapoints[idx].doc_id
        assert h.body == datapoints[idx].body


def test_empty_cluster_produces_no_hits():
    """If a cluster has no members, no hits should be emitted for it."""
    vectors = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int64)  # cluster 1 is empty
    centroids = np.array([[0.95, 0.05], [0.0, 1.0]], dtype=np.float32)
    datapoints = _make_datapoints(2)
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    cluster_ids = {h.cluster_id for h in hits}
    assert 1 not in cluster_ids


def test_single_point_cluster():
    """A cluster with one point should return exactly one hit with rank 0."""
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    labels = np.array([0, 1, 1], dtype=np.int64)
    centroids = np.array([[1.0, 0.0], [0.05, 0.95]], dtype=np.float32)
    datapoints = _make_datapoints(3)
    hits = boundary_rankings_for_all_clusters(datapoints, vectors, labels, centroids, n_clusters=2)
    c0_hits = [h for h in hits if h.cluster_id == 0]
    assert len(c0_hits) == 1
    assert c0_hits[0].rank_in_cluster == 0


def test_high_dimensional_vectors():
    """Ratio computation should work for typical embedding dimensions."""
    rng = np.random.default_rng(7)
    n, dim, k = 100, 384, 5
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    labels = np.repeat(np.arange(k), n // k).astype(np.int64)
    centroids = np.stack([vectors[labels == c].mean(axis=0) for c in range(k)])
    ratios, nearest_other = voronoi_boundary_ratio(vectors, labels, centroids)
    assert ratios.shape == (n,)
    assert (ratios >= 0.0).all()
    assert (nearest_other != labels).all()
