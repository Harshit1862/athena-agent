"""Hybrid retrieval: dense (LanceDB) + sparse (BM25) fused with reciprocal
rank fusion, then a cross-encoder reranks the fused candidates for the final
top-k. This combination catches both semantic matches (dense) and exact
keyword/acronym matches (sparse) that embeddings often miss — the standard
recipe for "best in class" RAG over a single corpus.

Reranker runs on CPU on purpose (harness/config.py keeps the GPU for qwen3:8b).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import lancedb
from sentence_transformers import CrossEncoder

from harness import config
from ingestion.embed_client import embed_batch
from ingestion.build_index import tokenize

DATA_DIR = Path(__file__).parent.parent / "data"
LANCE_DIR = DATA_DIR / "lancedb"
BM25_PATH = DATA_DIR / "bm25.pkl"

RRF_K = 60  # standard reciprocal-rank-fusion constant
DENSE_K = 20
SPARSE_K = 20
RERANK_TOP_K = 5


@dataclass
class Hit:
    id: str
    title: str
    abstract: str
    authors: str
    categories: str
    score: float


class HybridRetriever:
    def __init__(self):
        db = lancedb.connect(str(LANCE_DIR))
        self.table = db.open_table("papers")
        with BM25_PATH.open("rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_ids = data["ids"]
        self._reranker: CrossEncoder | None = None  # lazy-loaded, CPU

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        return self._reranker

    def _dense_search(self, query: str, k: int) -> list[str]:
        vec = embed_batch([query])[0]
        results = self.table.search(vec).limit(k).to_list()
        return [r["id"] for r in results]

    def _sparse_search(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.bm25_ids[i] for i in ranked]

    def _fuse(self, dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
        scores: dict[str, float] = {}
        for rank, doc_id in enumerate(dense_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc_id in enumerate(sparse_ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def search(self, query: str, top_k: int = RERANK_TOP_K) -> list[Hit]:
        dense_ids = self._dense_search(query, DENSE_K)
        sparse_ids = self._sparse_search(query, SPARSE_K)
        fused_ids = self._fuse(dense_ids, sparse_ids)[: DENSE_K + SPARSE_K]

        candidates = self.table.search().where(
            f"id IN ({', '.join(repr(i) for i in fused_ids)})"
        ).to_list()
        by_id = {c["id"]: c for c in candidates}
        ordered = [by_id[i] for i in fused_ids if i in by_id]

        pairs = [(query, c["text"]) for c in ordered]
        rerank_scores = self.reranker.predict(pairs) if pairs else []
        reranked = sorted(zip(ordered, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            Hit(
                id=c["id"],
                title=c["title"],
                abstract=c["abstract"],
                authors=c["authors"],
                categories=c["categories"],
                score=float(score),
            )
            for c, score in reranked
        ]
