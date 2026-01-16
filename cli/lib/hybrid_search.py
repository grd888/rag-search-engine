import os
from typing import Optional
from .query_enhancement import enhance_query
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    load_movies,
    format_search_result,
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
)
from .reranking import rerank


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

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        fused = reciprocal_rank_fusion(bm25_results, semantic_results, k)
        return fused[:limit]


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


def reciprocal_rank_fusion(
    bm25_results: list[dict], semantic_results: list[dict], k: int = RRF_K
) -> list[dict]:
    rrf_scores = {}

    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["bm25_rank"] is None:
            rrf_scores[doc_id]["bm25_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    for rank, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "rrf_score": 0.0,
                "bm25_rank": None,
                "semantic_rank": None,
            }
        if rrf_scores[doc_id]["semantic_rank"] is None:
            rrf_scores[doc_id]["semantic_rank"] = rank
            rrf_scores[doc_id]["rrf_score"] += rrf_score(rank, k)

    sorted_items = sorted(
        rrf_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True
    )

    rrf_results = []
    for doc_id, data in sorted_items:
        result = format_search_result(
            doc_id=doc_id,
            title=data["title"],
            document=data["document"],
            score=data["rrf_score"],
            rrf_score=data["rrf_score"],
            bm25_rank=data["bm25_rank"],
            semantic_rank=data["semantic_rank"],
        )
        rrf_results.append(result)

    return rrf_results


def rrf_search_command(
    query: str,
    k: int = RRF_K,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = searcher.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        results = rerank(query, results, method=rerank_method, limit=limit)
        reranked = True

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "reranked": reranked,
        "results": results,
    }
