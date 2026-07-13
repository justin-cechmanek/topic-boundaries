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
