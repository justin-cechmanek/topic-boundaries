from pathlib import Path

import numpy as np
import pytest

from topic_boundaries.cli import _doc_id_to_title, _json_serial, _parse_kmeans_n_init
from topic_boundaries.documents import Datapoint, load_jsonl


def test_doc_id_to_title_from_arxiv_jsonl():
    root = Path(__file__).resolve().parents[1]
    dps = load_jsonl(root / "datasets" / "sample.jsonl")
    m = _doc_id_to_title(dps)
    link = "https://arxiv.org/abs/2403.15001"
    assert m[link].startswith("Gradient descent")


def test_doc_id_to_title_empty_when_absent():
    dps = [Datapoint(doc_id="x", body="b", meta={})]
    assert _doc_id_to_title(dps) == {"x": ""}


@pytest.mark.parametrize(
    "arg,fallback,expected",
    [
        (None, "auto", "auto"),
        (None, 5, 5),
        ("10", "auto", 10),
        ("  7 ", "auto", 7),
        ("auto", 5, "auto"),
    ],
)
def test_parse_kmeans_n_init(arg, fallback, expected):
    assert _parse_kmeans_n_init(arg, fallback) == expected


def test_json_serial_numpy_and_error():
    assert _json_serial(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError):
        _json_serial(object())
