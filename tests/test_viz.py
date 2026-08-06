from pathlib import Path

import numpy as np
import pytest

from topic_boundaries.documents import datapoint_from_record
from topic_boundaries.visualization import boundary_highlight_indices, project_to_2d


def test_boundary_highlight_matches_doc_id():
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
            [{"doc_id": "https://arxiv.org/abs/1111.00001", "title": "Paper A"}],
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
    # 2D uses WebGL (Scattergl), not the SVG Scatter.
    assert fig2.data and all(isinstance(tr, go.Scattergl) for tr in fig2.data)
    assert not any(isinstance(tr, go.Scatter3d) for tr in fig2.data)

    xy3 = np.arange(12, dtype=float).reshape(4, 3)
    c3 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    fig3 = build_plotly_figure(dps, labels, highlight, xy3, c3, title="t")
    assert fig3.data and all(isinstance(tr, go.Scatter3d) for tr in fig3.data)
    assert all(tr.z is not None for tr in fig3.data)  # z coordinate present


def test_cluster_colors_palette_for_small_k():
    # <= palette size uses the qualitative palette verbatim (no plotly needed).
    from topic_boundaries.visualization.figure import _PALETTE, _cluster_colors

    assert _cluster_colors(3) == _PALETTE[:3]
    assert _cluster_colors(len(_PALETTE)) == _PALETTE


def test_cluster_colors_scale_past_palette():
    pytest.importorskip("plotly")  # >10 branch samples a plotly colorscale
    from topic_boundaries.visualization.figure import _PALETTE, _cluster_colors

    # > palette size: distinct color per cluster, no recycling
    n = len(_PALETTE) + 8
    colors = _cluster_colors(n)
    assert len(colors) == n
    assert len(set(colors)) == n


def test_build_figure_many_clusters_distinct_colors():
    pytest.importorskip("plotly.graph_objects")
    from topic_boundaries.documents import Datapoint
    from topic_boundaries.visualization import build_plotly_figure

    n = 15  # more than the 10-color palette
    dps = [Datapoint(doc_id=f"d{i}", body=f"b{i}", meta={}) for i in range(n)]
    labels = np.arange(n)
    xy = np.random.default_rng(0).standard_normal((n, 2))
    cents = np.random.default_rng(1).standard_normal((n, 2))
    fig = build_plotly_figure(dps, labels, set(), xy, cents, title="t")
    cluster_colors = [
        tr.marker.color for tr in fig.data if str(tr.name).startswith("cluster")
    ]
    assert len(cluster_colors) == n
    assert len(set(cluster_colors)) == n  # no recycling past 10


def test_build_figure_decimates_but_keeps_highlights():
    pytest.importorskip("plotly.graph_objects")
    from topic_boundaries.documents import Datapoint
    from topic_boundaries.visualization import build_plotly_figure

    n = 200
    dps = [Datapoint(doc_id=f"d{i}", body=f"b{i}", meta={}) for i in range(n)]
    labels = np.zeros(n, dtype=int)
    xy = np.random.default_rng(0).standard_normal((n, 2))
    cents = np.array([[0.0, 0.0]])
    highlight = {5, 150, 199}

    with pytest.warns(UserWarning, match="Decimated"):
        fig = build_plotly_figure(
            dps, labels, highlight, xy, cents, title="t", max_points=50
        )
    # Count rendered doc points (exclude the centroid trace).
    rendered = {
        int(t.split("doc_id: d")[1].split("<br>")[0])
        for tr in fig.data
        if str(tr.name).startswith("cluster")
        for t in tr.text
    }
    assert len(rendered) <= 50
    assert highlight <= rendered  # every highlighted point survives decimation


def test_export_png_writes_real_image(tmp_path):
    pytest.importorskip("matplotlib")
    from topic_boundaries.visualization import export_png

    n = 12
    labels = np.array([i % 3 for i in range(n)])
    xy = np.random.default_rng(0).standard_normal((n, 2))
    cents = np.random.default_rng(1).standard_normal((3, 2))
    out = export_png(labels, {1, 4}, xy, cents, tmp_path / "plot.png", title="t")
    assert out.is_file()
    # real PNG: 8-byte magic header, non-trivial size
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.stat().st_size > 1000


def test_export_png_3d(tmp_path):
    pytest.importorskip("matplotlib")
    from topic_boundaries.visualization import export_png

    n = 9
    labels = np.array([i % 3 for i in range(n)])
    xy = np.random.default_rng(0).standard_normal((n, 3))
    cents = np.random.default_rng(1).standard_normal((3, 3))
    out = export_png(labels, set(), xy, cents, tmp_path / "plot3d.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_boundary_results_round_trips_by_doc_id(tmp_path):
    # results/ is gitignored (generated artifacts). Round-trip the real writer
    # over the committed dataset's records and confirm the viz helper matches
    # them back by doc_id.
    import json

    from topic_boundaries.documents import load_jsonl
    from topic_boundaries.visualization import write_boundary_results

    root = Path(__file__).resolve().parents[1]
    dps = load_jsonl(root / "datasets" / "arxiv" / "ml_ai_100.jsonl")
    picked = dps[:2]

    out = tmp_path / "boundary.json"
    write_boundary_results(
        out,
        method="cross_boundary",
        n_clusters=1,
        doc_ids_by_cluster=[[dp.doc_id for dp in picked]],
        datapoints=dps,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    # Writer enriches with title/abstract but the match keys on doc_id.
    assert data["boundary_by_cluster"][0][0]["doc_id"] == picked[0].doc_id
    assert data["boundary_by_cluster"][0][0]["title"]
    assert boundary_highlight_indices(data, dps) == {0, 1}
