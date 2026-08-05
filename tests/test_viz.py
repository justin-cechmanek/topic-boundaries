import json
from pathlib import Path

import numpy as np
import pytest

from topic_boundaries.documents import datapoint_from_record
from topic_boundaries.visualization import boundary_highlight_indices, project_to_2d


def test_boundary_highlight_matches_title_and_abstract():
    dps = [
        datapoint_from_record(
            {
                "arxiv_link": "https://arxiv.org/abs/1111.00001",
                "title": "Paper A",
                "abstract": "Alpha abstract.",
            }
        ),
        datapoint_from_record(
            {
                "arxiv_link": "https://arxiv.org/abs/1111.00002",
                "title": "Paper B",
                "abstract": "Beta abstract.",
            }
        ),
    ]
    results = {
        "method": "max_distance_sort",
        "n_clusters": 2,
        "boundary_by_cluster": [
            [{"title": "Paper A", "abstract": "Alpha abstract."}],
            [],
        ],
    }
    hi = boundary_highlight_indices(results, dps)
    assert hi == {0}


def test_centroid_neighbors_uses_doc_id():
    dps = [
        datapoint_from_record(
            {
                "doc_id": "id-1",
                "body": "One",
            }
        ),
    ]
    results = {
        "method": "centroid_neighbors",
        "n_clusters": 1,
        "clusters": [
            {
                "cluster_id": 0,
                "nearest_to_centroid": [{"doc_id": "id-1", "title": "T1"}],
            }
        ],
    }
    assert boundary_highlight_indices(results, dps) == {0}


def test_project_pca_shapes():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    centroids = rng.standard_normal((3, 8)).astype(np.float32)
    labels = np.array([0] * 7 + [1] * 7 + [2] * 6)
    xy_d, xy_c = project_to_2d(
        vectors,
        labels,
        centroids,
        method="pca",
        random_state=0,
        tsne_perplexity=10.0,
    )
    assert xy_d.shape == (20, 2)
    assert xy_c.shape == (3, 2)


def test_project_3d_shapes():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    centroids = rng.standard_normal((3, 8)).astype(np.float32)
    labels = np.array([0] * 7 + [1] * 7 + [2] * 6)
    for method in ("pca", "tsne"):
        xy_d, xy_c = project_to_2d(
            vectors,
            labels,
            centroids,
            method=method,
            random_state=0,
            tsne_perplexity=10.0,
            n_components=3,
        )
        assert xy_d.shape == (20, 3)
        assert xy_c.shape == (3, 3)


def test_project_rejects_bad_n_components():
    v = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        project_to_2d(
            v,
            np.zeros(4, dtype=int),
            np.zeros((1, 3), dtype=np.float32),
            method="pca",
            random_state=0,
            tsne_perplexity=5.0,
            n_components=4,
        )


def test_tsne_projects_centroids_no_nan_even_empty_cluster():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((20, 8)).astype(np.float32)
    centroids = rng.standard_normal((3, 8)).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10)  # cluster 2 has no members
    xy_d, xy_c = project_to_2d(
        vectors,
        labels,
        centroids,
        method="tsne",
        random_state=0,
        tsne_perplexity=10.0,
    )
    assert xy_d.shape == (20, 2)
    assert xy_c.shape == (3, 2)
    # jointly embedded -> the empty cluster's centroid is still a real point
    assert np.isfinite(xy_c).all()


def test_tsne_high_dim_pre_pca_runs():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((30, 120)).astype(np.float32)  # 120 > pre-PCA 50
    centroids = rng.standard_normal((3, 120)).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    xy_d, xy_c = project_to_2d(
        vectors,
        labels,
        centroids,
        method="tsne",
        random_state=0,
        tsne_perplexity=10.0,
    )
    assert xy_d.shape == (30, 2) and xy_c.shape == (3, 2)
    assert np.isfinite(xy_d).all() and np.isfinite(xy_c).all()


def test_tsne_warns_on_large_n(monkeypatch):
    from topic_boundaries.visualization import core

    # Tune the threshold down so the O(n^2) warning fires on a tiny (fast) run.
    monkeypatch.setattr(core, "TSNE_MAX_RECOMMENDED", 5)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((12, 6)).astype(np.float32)
    centroids = rng.standard_normal((2, 6)).astype(np.float32)
    labels = np.array([0] * 6 + [1] * 6)
    with pytest.warns(UserWarning, match="O\\(n\\^2\\)"):
        core.project_to_2d(
            vectors,
            labels,
            centroids,
            method="tsne",
            random_state=0,
            tsne_perplexity=5.0,
        )


def test_project_precomputed_passthrough():
    coords = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    cents = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    xy_d, xy_c = project_to_2d(
        coords,
        np.array([0, 0, 1]),
        cents,
        method="precomputed",
        random_state=0,
        tsne_perplexity=5.0,
    )
    np.testing.assert_array_equal(xy_d, coords.astype(np.float64))
    np.testing.assert_array_equal(xy_c, cents.astype(np.float64))


def test_project_precomputed_width_mismatch_raises():
    with pytest.raises(ValueError):
        project_to_2d(
            np.zeros((3, 5), dtype=np.float32),  # width 5 != n_components 2
            np.array([0, 0, 1]),
            np.zeros((2, 5), dtype=np.float32),
            method="precomputed",
            random_state=0,
            tsne_perplexity=5.0,
        )


def test_project_unknown_method_raises():
    with pytest.raises(ValueError):
        project_to_2d(
            np.zeros((3, 4), dtype=np.float32),
            np.array([0, 0, 1]),
            np.zeros((1, 4), dtype=np.float32),
            method="isomap",
            random_state=0,
            tsne_perplexity=5.0,
        )


def test_project_umap_shapes():
    pytest.importorskip("umap")
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((30, 8)).astype(np.float32)
    centroids = rng.standard_normal((3, 8)).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    xy_d, xy_c = project_to_2d(
        vectors,
        labels,
        centroids,
        method="umap",
        random_state=0,
        tsne_perplexity=10.0,
        n_components=2,
    )
    assert xy_d.shape == (30, 2)
    assert xy_c.shape == (3, 2)  # UMAP.transform projects centroids too


def test_build_plotly_figure_2d_vs_3d():
    go = pytest.importorskip("plotly.graph_objects")
    from topic_boundaries.documents import Datapoint
    from topic_boundaries.visualization import build_plotly_figure

    dps = [
        Datapoint(doc_id=f"d{i}", body=f"body {i}", meta={"title": f"T{i}"})
        for i in range(4)
    ]
    labels = np.array([0, 0, 1, 1])
    highlight = {1}

    xy2 = np.arange(8, dtype=float).reshape(4, 2)
    c2 = np.array([[0.0, 0.0], [1.0, 1.0]])
    fig2 = build_plotly_figure(dps, labels, highlight, xy2, c2, title="t")
    assert fig2.data and all(isinstance(tr, go.Scatter) for tr in fig2.data)
    assert not any(isinstance(tr, go.Scatter3d) for tr in fig2.data)

    xy3 = np.arange(12, dtype=float).reshape(4, 3)
    c3 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    fig3 = build_plotly_figure(dps, labels, highlight, xy3, c3, title="t")
    assert fig3.data and all(isinstance(tr, go.Scatter3d) for tr in fig3.data)
    assert all(tr.z is not None for tr in fig3.data)  # z coordinate present


def test_results_json_loads_with_viz_helpers():
    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "ml_ai_100_max_distance_sort_n10_top5.json"
    if not path.is_file():
        pytest.skip(f"missing results fixture: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    sample = root / "datasets" / "arxiv" / "ml_ai_100.jsonl"
    if not sample.is_file():
        pytest.skip(f"missing dataset fixture: {sample}")
    from topic_boundaries.documents import load_jsonl

    dps = load_jsonl(sample)
    hi = boundary_highlight_indices(data, dps)
    assert len(hi) > 0
