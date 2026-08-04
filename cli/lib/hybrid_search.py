import os

from constants import CACHE_DIR
from lib.keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from utils import load_movies


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]

    
def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score  


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists(os.path.join(CACHE_DIR, "index.pkl")):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return [
            {
                "id": doc_id,
                "title": self.idx.docmap[doc_id]["title"],
                "document": self.idx.docmap[doc_id]["description"][:100],
                "score": score,
            }
            for doc_id, score in self.idx.bm25_search(query, limit)
        ]

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        pool = 500 * limit

        bm_query = self._bm25_search(query, pool)
        bm_norm = normalize([r["score"] for r in bm_query])

        sem_query = self.semantic_search.search_chunks(query, pool)
        sem_norm = normalize([r["score"] for r in sem_query])

        results: dict[int, dict] = {}
        for r, s in zip(sem_query, sem_norm):
            doc_id = r["id"]
            results[doc_id] = {
                "id": doc_id,
                "title": r["title"],
                "document": r["document"],
                "keyword_score": 0.0,
                "semantic_score": s,
                "hybrid_score": 0.0,
            }

        for r, s in zip(bm_query, bm_norm):
            doc_id = r["id"]
            if doc_id not in results:
                results[doc_id] = {
                    "id": doc_id,
                    "title": r["title"],
                    "document": r["document"],
                    "keyword_score": 0.0,
                    "semantic_score": 0.0,
                    "hybrid_score": 0.0,
                }
            results[doc_id]["keyword_score"] = s

        for value in results.values():
            value["hybrid_score"] = hybrid_score(value["keyword_score"], value["semantic_score"], alpha)

        ranked = sorted(results.values(), key=lambda d: d["hybrid_score"], reverse=True)
        return ranked[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> list[dict]:
    documents = load_movies()
    hybrid = HybridSearch(documents)
    return hybrid.weighted_search(query, alpha, limit)