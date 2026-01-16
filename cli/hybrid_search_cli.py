import argparse
from lib.hybrid_search import (
    normalize_scores,
    weighted_search_command,
    rrf_search_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize command")
    normalize_parser.add_argument(
        "scores", nargs="+", type=float, help="Scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Weighted search command"
    )
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument(
        "--alpha", type=float, default=0.5, help="Alpha value"
    )
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Limit")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF search command")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="K value")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Limit")
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite"],
        help="Query enhancement method",
    )
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_search_command(args.query, args.alpha, args.limit)
            # print(results[0])
            for i, result in enumerate(results):
                print(f"{i + 1}. \t{result['document']['title']}")
                print(f"\tHybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"\tBM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}"
                )
                print(f"{result['document']['document'][:100]}...")
                print()
        case "rrf-search":
            rrf_result = rrf_search_command(args.query, args.k, args.limit, args.enhance)
            results = rrf_result["results"]
            query = rrf_result["query"]
            enhanced_query = rrf_result["enhanced_query"]
            method = rrf_result["method"]
            print( f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
            print() 
            for i, result in enumerate(results):
                print(f"{i + 1}. \t{result['document']['title']}")
                print(f"\tRRF Score: {result['rrf_score']:.3f}")
                print(
                    f"\tBM25 Rank: {result['bm25_rank']:.3f}, Semantic Rank: {result['semantic_rank']:.3f}"
                )
                print(f"\t{result['document']['document'][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
