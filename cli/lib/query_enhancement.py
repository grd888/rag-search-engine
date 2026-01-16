import os
from typing import Optional 
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

MODEL_ID = "gemini-3-flash-preview"
client = genai.Client(api_key=api_key)

def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_check(query)
        case "rewrite":
            return rewrite_query(query)
        case _:
            return query
        
def spell_check(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text

