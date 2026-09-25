from io import BytesIO
import xml.etree.ElementTree as ET

import pytest

from topic_boundaries.collect_datasets import (
    _paper_record_from_feed,
    download_pdf,
    fetch_page,
    fetch_paper,
    normalize_arxiv_id,
    parse_entry,
)

# One arXiv Atom <entry>, verbatim shape from the export API (no network).
ENTRY_XML = """
<entry xmlns="http://www.w3.org/2005/Atom">
  <id>http://arxiv.org/abs/2403.15001v1</id>
  <published>2024-03-15T10:00:00Z</published>
  <title>A Study of Topic Boundaries</title>
  <summary>  We investigate topic boundaries in embeddings.  </summary>
  <author><name>Alice Chen</name></author>
  <author><name>Bob Kumar</name></author>
  <category term="cs.CL"/>
  <category term="cs.LG"/>
</entry>
"""


def test_parse_entry_extracts_fields():
    entry = ET.fromstring(ENTRY_XML)
    rec = parse_entry(entry)
    assert rec["title"] == "A Study of Topic Boundaries"
    assert rec["abstract"] == "We investigate topic boundaries in embeddings."
    assert rec["publication_date"] == "2024-03-15"  # truncated to YYYY-MM-DD
    assert rec["arxiv_link"] == "http://arxiv.org/abs/2403.15001v1"
    assert rec["authors"] == ["Alice Chen", "Bob Kumar"]
    assert rec["subjects"] == ["cs.CL", "cs.LG"]


def test_parse_entry_handles_missing_optional_fields():
    entry = ET.fromstring(
        '<entry xmlns="http://www.w3.org/2005/Atom">'
        "<id>http://arxiv.org/abs/1.2</id><title>T</title><summary>S</summary>"
        "</entry>"
    )
    rec = parse_entry(entry)
    assert rec["authors"] == []
    assert rec["subjects"] == []
    assert rec["publication_date"] == ""


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2403.15001", "2403.15001"),
        ("2403.15001v2", "2403.15001v2"),
        ("0704.0001", "0704.0001"),
        ("https://arxiv.org/abs/2403.15001", "2403.15001"),
        ("http://arxiv.org/abs/2403.15001v1", "2403.15001v1"),
        ("https://arxiv.org/pdf/2403.15001.pdf", "2403.15001"),
        ("arxiv.org/abs/cs.CL/0303001", "cs.CL/0303001"),
        ("hep-th/9901001v3", "hep-th/9901001v3"),
    ],
)
def test_normalize_arxiv_id(value, expected):
    assert normalize_arxiv_id(value) == expected


def test_normalize_arxiv_id_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_arxiv_id("not-an-arxiv-id")


def test_paper_record_from_feed_parses_first_entry():
    feed = (
        '<feed xmlns="http://www.w3.org/2005/Atom">' + ENTRY_XML + "</feed>"
    ).encode("utf-8")
    rec = _paper_record_from_feed(feed)
    assert rec["title"] == "A Study of Topic Boundaries"
    assert rec["arxiv_link"] == "http://arxiv.org/abs/2403.15001v1"


def test_paper_record_from_feed_raises_on_empty():
    empty = b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    with pytest.raises(ValueError):
        _paper_record_from_feed(empty)


def test_http_helpers_fetch_feed_and_pdf(monkeypatch, tmp_path):
    feed = ('<feed xmlns="http://www.w3.org/2005/Atom">' + ENTRY_XML + "</feed>").encode()
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return BytesIO(b"%PDF" if url.endswith(".pdf") else feed)

    monkeypatch.setattr("topic_boundaries.collect_datasets.urlopen", fake_urlopen)
    root = fetch_page("cat:cs.CL", 0, 1)
    assert root.find("{http://www.w3.org/2005/Atom}entry") is not None
    assert fetch_paper("2403.15001")["title"] == "A Study of Topic Boundaries"
    pdf = download_pdf("2403.15001", tmp_path / "paper.pdf")
    assert pdf.read_bytes() == b"%PDF"
    assert [timeout for _, timeout in calls] == [60, 60, 120]
