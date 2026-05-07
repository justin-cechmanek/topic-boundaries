from pathlib import Path

from src.cli import _doc_id_to_title
from src.documents import load_jsonl


def test_doc_id_to_title_from_arxiv_jsonl():
    root = Path(__file__).resolve().parents[1]
    dps = load_jsonl(root / "datasets" / "sample.jsonl")
    m = _doc_id_to_title(dps)
    link = "https://arxiv.org/abs/2403.15001"
    assert m[link].startswith("Gradient descent")
