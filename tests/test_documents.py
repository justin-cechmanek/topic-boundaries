import json
from pathlib import Path

import pytest

from topic_boundaries.documents import datapoint_from_record, load_jsonl


def test_load_sample_jsonl(tmp_path: Path):
    p = tmp_path / "rows.jsonl"
    rows = [
        {"doc_id": "a", "body": "hello"},
        {"abstract": "world", "arxive_link": "https://arxiv.org/abs/1234.5678"},
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    dps = load_jsonl(p)
    assert len(dps) == 2
    assert dps[0].doc_id == "a"
    assert "arxiv.org" in dps[1].doc_id


def test_abstract_preferred_when_body_also_present():
    dp = datapoint_from_record(
        {
            "body": "generic body field",
            "abstract": "arxiv abstract text",
            "arxive_link": "https://arxiv.org/abs/9999.99999",
        }
    )
    assert dp.body == "arxiv abstract text"


def test_datapoint_requires_text():
    with pytest.raises(ValueError):
        datapoint_from_record({"doc_id": "x"})


def test_arxiv_link_alias_for_abstract_records():
    dp = datapoint_from_record(
        {
            "abstract": "paper summary",
            "arxiv_link": "https://arxiv.org/abs/1111.2222",
        }
    )
    assert "1111.2222" in dp.doc_id
    assert dp.body == "paper summary"


@pytest.mark.parametrize(
    "rec,expected_id",
    [
        ({"body": "b", "doc_id": "d1"}, "d1"),
        ({"body": "b", "id": "i1"}, "i1"),
        ({"body": "b", "arxiv_link": "l1"}, "l1"),
        ({"body": "b", "arxive_link": "l2"}, "l2"),
    ],
)
def test_body_record_doc_id_fallback_chain(rec, expected_id):
    assert datapoint_from_record(rec).doc_id == expected_id


def test_empty_inferred_doc_id_raises():
    with pytest.raises(ValueError):
        datapoint_from_record({"abstract": "text but no id"})


def test_meta_excludes_body_and_abstract():
    dp = datapoint_from_record(
        {"abstract": "a", "arxiv_link": "l", "title": "T", "subjects": ["x"]}
    )
    assert "abstract" not in dp.meta and "body" not in dp.meta
    assert dp.meta["title"] == "T" and dp.meta["subjects"] == ["x"]


def test_record_reads_precomputed_vector():
    import numpy as np

    dp = datapoint_from_record({"doc_id": "v", "vector": [0.1, 0.2, 0.3]})
    assert dp.body == ""  # vector-only, no text required
    np.testing.assert_array_equal(dp.vector, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert "vector" not in dp.meta  # not leaked into meta


def test_vector_only_record_still_needs_doc_id():
    with pytest.raises(ValueError):
        datapoint_from_record({"vector": [0.1, 0.2]})


def test_record_without_text_or_vector_raises():
    with pytest.raises(ValueError, match="abstract.*body.*vector"):
        datapoint_from_record({"doc_id": "x", "title": "no content"})
