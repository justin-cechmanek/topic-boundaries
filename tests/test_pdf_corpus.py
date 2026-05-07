from pathlib import Path

import pytest

from src.pdf_corpus import chunk_text, normalize_whitespace


def test_normalize_whitespace_collapses_line_noise():
    assert normalize_whitespace("  a  \n\n  b\tc ") == "a\nb c"


def test_chunk_respects_paragraph_boundaries():
    text = "para one.\n\npara two is here.\n\npara three."
    chunks = chunk_text(text, max_chars=30, overlap=0)
    assert len(chunks) >= 2
    joined = "\n".join(chunks)
    assert "para one" in joined


@pytest.mark.parametrize(
    "pdf_name",
    ["knowledge_boundaries_paper_proposal.pdf"],
)
def test_project_proposal_pdf_chunks(pdf_name: str):
    pytest.importorskip("pypdf")
    from src.pdf_corpus import pdf_to_datapoints

    root = Path(__file__).resolve().parents[1]
    pdf = root / pdf_name
    if not pdf.is_file():
        pytest.skip(f"missing {pdf}")
    dps = pdf_to_datapoints(pdf, max_chars=2000)
    assert len(dps) >= 5
    assert all(len(d.body) > 50 for d in dps[:3])
