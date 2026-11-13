import pytest

from topic_boundaries import RecordsSource, datapoint_from_record


@pytest.mark.parametrize(
    "record",
    [
        {"abstract": "text", "arxiv_link": 0, "title": "fallback"},
        {"abstract": "text", "doc_id": 0},
        {"body": "text", "doc_id": 0, "id": "fallback"},
        {"body": "text", "id": 0},
        {"vector": [1.0, 0.0], "doc_id": 0, "title": "fallback"},
        {"vector": [1.0, 0.0], "id": 0},
    ],
)
def test_numeric_zero_is_a_document_identifier(record):
    assert datapoint_from_record(record).doc_id == "0"


def test_records_source_accepts_zero_based_identifiers():
    source = RecordsSource(
        [{"id": 0, "text": "first"}, {"id": 1, "text": "second"}],
        id_col="id", body_col="text",
    )
    assert [point.doc_id for point in source.load()] == ["0", "1"]
