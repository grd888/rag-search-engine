import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")
client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-3-flash-preview"
prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

response = client.models.generate_content(
  model=MODEL_ID, 
  contents=prompt,
)
candidates_token_count = response.usage_metadata.candidates_token_count
prompt_token_count = response.usage_metadata.prompt_token_count

print(f"Prompt Tokens: {prompt_token_count}")
print(f"Response Tokens: {candidates_token_count}")
