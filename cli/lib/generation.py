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

def summarize_with_citations(query: str, documents: list[dict]):
    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def answer_question(question: str, context: list[dict]):
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


# Commands
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


def citations_command(query: str, limit: int = 5) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(query, k=60, limit=limit)
    docs = [f"{doc['title']} - {doc['document']}" for doc in search_results]
    summary = summarize_with_citations(query, docs)
    print("Search Results:")
    for res in search_results:
        print(f"- {res['title']}")
    print()
    print("LLM Answer:")
    print(summary)
    
def question_answering_command(question: str, limit: int = 5) -> dict:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    search_results = hybrid_search.rrf_search(question, k=60, limit=limit)
    docs = [f"{doc['title']} - {doc['document']}" for doc in search_results]
    answer = answer_question(question=question, context=docs)
    print("Search Results:")
    for res in search_results:
        print(f"- {res['title']}")
    print()
    print("Answer:")
    print(answer)
    