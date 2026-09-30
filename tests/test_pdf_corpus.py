from pathlib import Path

import pytest

from topic_boundaries.pdf_corpus import chunk_text, normalize_whitespace


def test_normalize_whitespace_collapses_line_noise():
    assert normalize_whitespace("  a  \n\n  b\tc ") == "a\n\nb c"


def test_chunk_respects_paragraph_boundaries():
    text = "para one.\n\npara two is here.\n\npara three."
    chunks = chunk_text(text, max_chars=30, overlap=0)
    assert chunks == ["para one.\n\npara two is here.", "para three."]


def test_oversized_paragraph_hard_split_with_overlap():
    para = "word " * 100  # ~500 chars, one paragraph, no blank lines
    chunks = chunk_text(para, max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # overlap: the tail of chunk[0] actually reappears at the start of chunk[1].
    assert chunks[0][-15:] in chunks[1]
    # no duplicate tail crawl: char count roughly tracks input + overlap, not 5x.
    assert sum(len(c) for c in chunks) < 2 * len(para)


def test_overlap_applies_between_paragraph_chunks():
    text = "alpha beta gamma.\n\ndelta epsilon zeta.\n\neta theta iota."
    chunks = chunk_text(text, max_chars=35, overlap=6)
    assert len(chunks) > 1
    assert all(len(c) <= 35 for c in chunks)
    assert chunks[0][-6:] in chunks[1]


@pytest.mark.parametrize("max_chars, overlap", [(0, 0), (10, -1), (10, 10)])
def test_chunk_rejects_invalid_limits(max_chars: int, overlap: int):
    with pytest.raises(ValueError, match="max_chars"):
        chunk_text("some text", max_chars=max_chars, overlap=overlap)


@pytest.mark.parametrize(
    "pdf_name",
    ["knowledge_boundaries_paper_proposal.pdf"],
)
def test_project_proposal_pdf_chunks(pdf_name: str):
    pytest.importorskip("pypdf")
    from topic_boundaries.pdf_corpus import pdf_to_datapoints

    root = Path(__file__).resolve().parents[1]
    pdf = root / pdf_name
    if not pdf.is_file():
        pytest.skip(f"missing {pdf}")
    # ~2.5k chars of extractable text; 500-char chunks give several real chunks.
    dps = pdf_to_datapoints(pdf, max_chars=500)
    assert len(dps) >= 5
    assert all(len(d.body) > 50 for d in dps[:3])
    # doc_id format "{stem}#chunk{i}" and per-chunk meta.
    stem = pdf.stem
    assert dps[0].doc_id == f"{stem}#chunk0"
    assert dps[3].doc_id == f"{stem}#chunk3"
    assert dps[0].meta["chunk_index"] == 0
    assert dps[0].meta["source_pdf"].endswith(pdf_name)
