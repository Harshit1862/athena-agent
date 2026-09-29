"""Turn data/raw/arxiv.jsonl into the two indexes hybrid RAG needs:

- Dense: LanceDB table of (id, title, abstract, ..., vector) — embedded in
  batches via nomic-embed-text.
- Sparse: a BM25 index over the same text, pickled alongside the id list so
  rag/retriever.py can look up full records by id after either search.

One chunk per paper (title + abstract): arXiv abstracts are short enough
(~150-300 words) that splitting them would only hurt retrieval, so chunking
here is a no-op by design, not an oversight.

Bulk embedding is CPU-bound and slow (~17s per 64-doc batch, ~2hrs for the
full corpus) on the default CPU-only Ollama instance. Nothing needs the GPU
during ingestion (qwen3:8b is only loaded for chat), so for this one-time
job run with the GPU instance instead — ~50x faster once warm:

    ATHENA_EMBED_BASE_URL=http://127.0.0.1:11435 python -m ingestion.build_index

Per-query embedding at chat time stays on the CPU instance (harness/config.py
default) since one query is cheap either way and it keeps VRAM free for the
reasoning model.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import lancedb
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ingestion.embed_client import embed_batch

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "raw" / "arxiv.jsonl"
LANCE_DIR = DATA_DIR / "lancedb"
BM25_PATH = DATA_DIR / "bm25.pkl"
EMBED_BATCH_SIZE = 64


def load_records() -> list[dict]:
    records = []
    with RAW_PATH.open() as f:
        for line in f:
            records.append(json.loads(line))
    return records


def chunk_text(r: dict) -> str:
    return f"{r['title']}\n\n{r['abstract']}"


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build() -> None:
    records = load_records()
    print(f"loaded {len(records)} papers")

    texts = [chunk_text(r) for r in records]

    # --- dense index -------------------------------------------------------
    db = lancedb.connect(str(LANCE_DIR))
    rows = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="embedding"):
        batch_texts = texts[i : i + EMBED_BATCH_SIZE]
        batch_records = records[i : i + EMBED_BATCH_SIZE]
        vectors = embed_batch(batch_texts)
        for rec, text, vec in zip(batch_records, batch_texts, vectors):
            rows.append(
                {
                    "id": rec["id"],
                    "title": rec["title"],
                    "abstract": rec["abstract"],
                    "authors": ", ".join(rec["authors"]),
                    "categories": ", ".join(rec["categories"]),
                    "published": rec["published"],
                    "text": text,
                    "vector": vec,
                }
            )

    if "papers" in db.list_tables():
        db.drop_table("papers")
    db.create_table("papers", data=rows)
    print(f"dense index built: {len(rows)} rows in {LANCE_DIR}")

    # --- sparse index --------------------------------------------------------
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    ids = [r["id"] for r in records]
    with BM25_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    print(f"sparse index built: {BM25_PATH}")


if __name__ == "__main__":
    build()
