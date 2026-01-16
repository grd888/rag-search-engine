import os
from typing import Optional 
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

MODEL_ID = "gemini-3-flash-preview"

def spell_check(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

    Only correct obvious typos. Don't change correctly spelled words.

    Query: "{query}"

    If no errors, return the original query.
    Corrected:"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text

def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_check(query)
        case _:
            return query