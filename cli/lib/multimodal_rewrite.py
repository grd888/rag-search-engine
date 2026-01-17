import mimetypes
from google import genai
from dotenv import load_dotenv
import os
from google.genai.types import Part

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


def describe_command(image: str, query: str) -> None:
    print(f"Describing image: {image} with query: {query}")
    mime, _ = mimetypes.guess_type(image)
    mime = mime or "image/jpeg"

    with open(image, "rb") as f:
        data = f.read()

    system_prompt = f"""
    Analyze this image and rewrite the text query for better search results.
    Image: {data}
    Text Query: {query}
    
    - Combine visual and textual information
    - Focus on movie-specific details (actors, plot, genre, etc.)
    - Return only the rewritten query, without any additional commentary
    """

    parts = [system_prompt, Part.from_bytes(data=data, mime_type=mime), query.strip()]

    response = client.models.generate_content(model=model, contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
