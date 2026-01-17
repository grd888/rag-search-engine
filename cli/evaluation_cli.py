import argparse
import json
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        dataset = json.load(f)

    for test_case in dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        movies = load_movies()
        searcher = HybridSearch(movies)
        results = searcher.rrf_search(query, k=60, limit=limit)
        titles = [result["title"] for result in results]
        relevant_titles = [title for title in relevant_docs]
        precision_at_k = len(set(titles) & set(relevant_titles)) / limit
        
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_at_k:.4f}")
        print(f"  - Retrieved: {titles}")
        print(f"  - Relevant: {relevant_docs}")
        
if __name__ == "__main__":
    main()