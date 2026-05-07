import json
from pathlib import Path

from src.documents import datapoint_from_record, load_jsonl


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
    try:
        datapoint_from_record({"doc_id": "x"})
    except ValueError:
        return
    raise AssertionError("expected ValueError")
