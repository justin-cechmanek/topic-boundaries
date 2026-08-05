"""
Harvest arXiv Atom API results into JSONL under datasets/.

Example:
  PYTHONPATH=. python -m topic_boundaries.collect_datasets --query "cat:cs.CL" --max-results 200 \\
      --out datasets/arxiv/cs_cl.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf"
NS = {"atom": "http://www.w3.org/2005/Atom"}

# arXiv identifiers: new-style "YYMM.NNNNN" (optional vN) or old-style
# "archive[.subclass]/YYMMNNN" (optional vN). Works on a bare id or any URL.
_ARXIV_ID = re.compile(
    r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z\-.]+/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)


def normalize_arxiv_id(id_or_url: str) -> str:
    """Extract a bare arXiv id from an id or an abs/pdf URL."""
    m = _ARXIV_ID.search(id_or_url.strip())
    if not m:
        raise ValueError(f"Could not parse an arXiv id from {id_or_url!r}")
    return m.group(1)


def fetch_page(search_query: str, start: int, max_results: int) -> ET.Element:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return ET.fromstring(r.content)


def parse_entry(entry: ET.Element) -> dict:
    title = (entry.findtext("atom:title", default="", namespaces=NS) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=NS) or "").strip()
    published = (entry.findtext("atom:published", default="", namespaces=NS) or "").strip()
    link_el = entry.find("atom:id", NS)
    arxiv_link = link_el.text.strip() if link_el is not None and link_el.text else ""
    authors = [
        (a.findtext("atom:name", default="", namespaces=NS) or "").strip()
        for a in entry.findall("atom:author", NS)
    ]
    subjects = [
        (c.get("term") or "").strip()
        for c in entry.findall("atom:category", NS)
        if c.get("term")
    ]
    return {
        "authors": authors,
        "publication_date": published[:10] if published else "",
        "title": title,
        "abstract": summary,
        "subjects": subjects,
        "arxiv_link": arxiv_link,
    }


def _paper_record_from_feed(content: bytes) -> dict:
    """Parse the first <entry> of an Atom feed into a record (single-paper lookup)."""
    root = ET.fromstring(content)
    entry = root.find("atom:entry", NS)
    if entry is None:
        raise ValueError("arXiv feed contained no <entry> (unknown id?)")
    return parse_entry(entry)


def fetch_paper(id_or_url: str) -> dict:
    """Metadata/abstract record for a single arXiv paper by id or URL."""
    arxiv_id = normalize_arxiv_id(id_or_url)
    url = f"{ARXIV_API}?{urllib.parse.urlencode({'id_list': arxiv_id, 'max_results': 1})}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return _paper_record_from_feed(r.content)


def download_pdf(id_or_url: str, dest: Path) -> Path:
    """Download a paper's PDF to ``dest`` and return the path."""
    arxiv_id = normalize_arxiv_id(id_or_url)
    r = requests.get(f"{ARXIV_PDF}/{arxiv_id}.pdf", timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def harvest(search_query: str, max_results: int, page_size: int = 100) -> list[dict]:
    rows: list[dict] = []
    start = 0
    while len(rows) < max_results:
        batch = min(page_size, max_results - len(rows))
        root = fetch_page(search_query, start=start, max_results=batch)
        entries = root.findall("atom:entry", NS)
        if not entries:
            break
        for e in entries:
            rows.append(parse_entry(e))
            if len(rows) >= max_results:
                break
        start += len(entries)
        time.sleep(3.1)  # arXiv polite polling guideline (~3s between requests)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Collect arXiv abstracts to JSONL.")
    p.add_argument("--query", required=True, help='e.g. cat:cs.LG OR ti:"large language model"')
    p.add_argument("--max-results", type=int, default=100)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = harvest(args.query, args.max_results)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} records to {args.out}")


if __name__ == "__main__":
    main()
