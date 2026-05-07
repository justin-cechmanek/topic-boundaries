"""
Harvest arXiv Atom API results into JSONL under datasets/.

Example:
  PYTHONPATH=. python -m src.collect_datasets --query "cat:cs.CL" --max-results 200 \\
      --out datasets/arxiv/cs_cl.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


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
        "arxive_link": arxiv_link,
    }


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
