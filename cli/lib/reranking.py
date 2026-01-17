import os
import json
from time import sleep

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"

def llm_rerank_cross_encoder(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []
    pairs = []
    for doc in documents:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)

    reranked = []
    for i, score in enumerate(scores):
        reranked.append({**documents[i], "cross_encoder_score": score})
    reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return reranked[:limit]

def llm_rerank_batch(query: str, documents: list[dict], limit: int = 5) -> list[dict]:
    if not documents:
        return []

    doc_map = {}
    doc_list = []
    for doc in documents:
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        doc_list.append(
            f"{doc_id}: {doc.get('title', '')} - {doc.get('document', '')[:200]}"
        )

    doc_list_str = "\n".join(doc_list)
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""
    response = client.models.generate_content(model=model, contents=prompt)
    ranking_text = (response.text or "").strip()
    
    parsed_ids = json.loads(ranking_text)
  
    reranked = []
    for i, doc_id in enumerate(parsed_ids):
        if doc_id in doc_map:
            reranked.append({**doc_map[doc_id], "batch_rank": i + 1})

    return reranked[:limit]


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "rerank_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_docs[:limit]


def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    match method:
       case "individual":     
            return llm_rerank_individual(query, documents, limit)
       case "batch":
            return llm_rerank_batch(query, documents, limit)
       case "cross_encoder":
            return llm_rerank_cross_encoder(query, documents, limit)
       case _:
            return documents[:limit]
