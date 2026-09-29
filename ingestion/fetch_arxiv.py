"""Pull paper metadata straight from arXiv's public API — no account/API key
needed (unlike the Kaggle metadata dump). Respects arXiv's rate-limit
guidance (one request every ~3s, batches of 200) and writes JSONL
incrementally so the run is resumable if interrupted.

Target: ~30k recent cs.AI / cs.LG / cs.CL papers. Big enough to force real
ingestion/indexing engineering, small enough to embed on a laptop GPU+CPU in
well under an hour.
"""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

API_URL = "https://export.arxiv.org/api/query"
CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]
BATCH_SIZE = 200
TARGET_COUNT = 30_000
SLEEP_SECONDS = 3.0
OUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "arxiv.jsonl"

NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def _query(search_query: str, start: int, max_results: int) -> str:
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "athena-agent/0.1 (research project)", "Accept": "*/*"}
    # A fresh connection per request, not a shared keep-alive client: a
    # long-lived connection across ~10 minutes of polling once left the
    # socket in CLOSE-WAIT with the process hung forever on read() (observed
    # during testing) — httpx's per-call timeout doesn't save you if the
    # underlying connection is already stuck half-closed.
    with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)) as client:
        resp = client.get(API_URL, params=params, headers=headers)
        resp.raise_for_status()
        return resp.text


def _parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    entries = []
    for entry in root.findall("atom:entry", NS):
        arxiv_id = entry.findtext("atom:id", default="", namespaces=NS).rsplit("/", 1)[-1]
        title = " ".join(entry.findtext("atom:title", default="", namespaces=NS).split())
        abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=NS).split())
        published = entry.findtext("atom:published", default="", namespaces=NS)
        authors = [
            a.findtext("atom:name", default="", namespaces=NS)
            for a in entry.findall("atom:author", NS)
        ]
        categories = [c.get("term") for c in entry.findall("atom:category", NS)]
        entries.append(
            {
                "id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "published": published,
                "authors": authors,
                "categories": categories,
            }
        )
    return entries


def already_fetched(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def fetch(target_count: int = TARGET_COUNT) -> None:
    # NOTE: arXiv's edge WAF returns 406 for any boolean "OR" search_query
    # (confirmed: single "cat:X" always works, "cat:X OR cat:Y" always 406,
    # even parenthesized/differently-encoded) — so each category is queried
    # separately and results are merged with id-based dedup instead.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = already_fetched(OUT_PATH)
    print(f"resuming with {len(seen)} papers already fetched", flush=True)
    per_category_target = target_count // len(CATEGORIES)

    with OUT_PATH.open("a") as out:
        for category in CATEGORIES:
            search_query = f"cat:{category}"
            start = 0
            fetched_this_category = 0

            while fetched_this_category < per_category_target and len(seen) < target_count:
                try:
                    xml_text = _query(search_query, start=start, max_results=BATCH_SIZE)
                    entries = _parse_entries(xml_text)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{category}] request failed at start={start}: {exc}; backing off", flush=True)
                    time.sleep(SLEEP_SECONDS * 5)
                    continue

                if not entries:
                    print(f"[{category}] no more entries; moving to next category", flush=True)
                    break

                new = 0
                for e in entries:
                    if e["id"] not in seen:
                        out.write(json.dumps(e) + "\n")
                        seen.add(e["id"])
                        new += 1
                out.flush()

                start += BATCH_SIZE
                fetched_this_category += len(entries)
                print(f"[{category}] {fetched_this_category}/{per_category_target} scanned, {len(seen)} total unique (+{new} new)", flush=True)
                time.sleep(SLEEP_SECONDS)

    print(f"done: {len(seen)} unique papers written to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    fetch()
