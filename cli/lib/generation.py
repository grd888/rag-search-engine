import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import load_movies

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"


def generate_answer(query: str, docs: list[dict]) -> str:
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{chr(10).join(docs)}

Provide a comprehensive answer that addresses the query:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text
 
  
def summarize_results(query: str, results: list[dict]):
    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{chr(10).join(results)}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def rag_command(query: str) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query, k=60, limit=5)
    docs = [f"{doc['title']} - {doc['document']}" for doc in search_results]
    answer = generate_answer(query, docs)

    print("Search Results:")
    for res in search_results:
        print(f"- {res['title']}")
    print()
    print("RAG Response:")
    print(answer)


def summarize_command(query: str, limit: int = 5) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
    docs = [f"{doc['title']} - {doc['document']}" for doc in search_results]
    summary = summarize_results(query, docs)
    print("Search Results:")
    for res in search_results:
        print(f"- {res['title']}")
    print()
    print("LLM Summary:")
    print(summary)
    
