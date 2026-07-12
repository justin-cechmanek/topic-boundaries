import xml.etree.ElementTree as ET

from src.collect_datasets import parse_entry

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
    assert rec["arxive_link"] == "http://arxiv.org/abs/2403.15001v1"
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
