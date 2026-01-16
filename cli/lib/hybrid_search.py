import os
from typing import Optional
from .query_enhancement import enhance_query
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import load_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25_results = self._bm25_search(query, limit * 500)
        bm25_scores = [result["score"] for result in bm25_results]
        normalized_bm25_scores = normalize_scores(bm25_scores)
        normalized_results = {
            result["id"]: {
                "bm25_score": score,
                "document": result,
            }
            for result, score in zip(bm25_results, normalized_bm25_scores)
        }

        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        semantic_scores = [result["score"] for result in semantic_results]
        normalized_semantic_scores = normalize_scores(semantic_scores)

        for result, score in zip(semantic_results, normalized_semantic_scores):
            if result["id"] not in normalized_results:
                normalized_results[result["id"]] = result
            normalized_results[result["id"]]["semantic_score"] = score

        for _, result in normalized_results.items():
            result["hybrid_score"] = hybrid_score(
                result.get("bm25_score", 0.0), result.get("semantic_score", 0.0), alpha
            )

        return sorted(
            normalized_results.values(), key=lambda x: x["hybrid_score"], reverse=True
        )[:limit]

    def rrf_search(self, query, k, limit=10, enhance=None):
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        # Create a dictionary mapping document IDs to the documents themselves and their BM25 and semantic ranks (not scores).
        ranks = {}
        for i, result in enumerate(bm25_results):
            doc_id = result["id"]
            ranks[doc_id] = {
                "document": result,
                "bm25_rank": i + 1,
                "semantic_rank": limit * 500,
            }

        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id in ranks:
                ranks[doc_id]["semantic_rank"] = i + 1
            else:
                ranks[doc_id] = {
                    "document": result,
                    "bm25_rank": limit * 500,
                    "semantic_rank": i + 1,
                }

        for doc_id, rank in ranks.items():
            rank["rrf_score"] = rrf_score(rank["bm25_rank"], k) + rrf_score(
                rank["semantic_rank"], k
            )

        return sorted(ranks.values(), key=lambda x: x["rrf_score"], reverse=True)[
            :limit
        ]


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    max_score = max(scores)
    min_score = min(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(score - min_score) / (max_score - min_score) for score in scores]


def weighted_search_command(query, alpha, limit):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search.weighted_search(query, alpha, limit)


def rrf_search_command(query: str, k: int, limit: int, enhance: Optional[str] = None):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    enhanced_query = query
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)

    results = hybrid_search.rrf_search(enhanced_query, k, limit)
    return {
        "results": results,
        "method": enhance,
        "query": query,
        "enhanced_query": enhanced_query,
    }
