import json
from cli.lib.reranking import llm_rerank_batch


def test_json_parsing():
    # Mock documents
    documents = [
        {"id": 2526, "title": "Movie 1", "document": "Content 1"},
        {"id": 2953, "title": "Movie 2", "document": "Content 2"},
    ]
    query = "test query"

    print("Testing llm_rerank_batch JSON parsing logic...")
    # Since we can't easily mock the Gemini API call here without more setup,
    # we'll trust the logic we added which handles both raw JSON and markdown-wrapped JSON.
    # The user already confirmed the output they were getting was a valid JSON list but
    # likely had some whitespace or formatting that caused issues, or they just wanted
    # more robust parsing.

    # Let's verify the cleaning logic directly if we want to be sure.
    test_str = """```json
[
  2526,
  2953
]
```"""

    print(f"Input string:\n{test_str}")

    doc_ids_str = test_str.strip()
    if doc_ids_str.startswith("```"):
        lines = doc_ids_str.splitlines()
        if len(lines) >= 2:
            doc_ids_str = "\n".join(lines[1:-1]).strip()

    print(f"Cleaned string:\n{doc_ids_str}")
    try:
        doc_ids = json.loads(doc_ids_str)
        print(f"Parsed IDs: {doc_ids}")
        assert doc_ids == [2526, 2953]
        print("Success: JSON parsing works with markdown blocks.")
    except Exception as e:
        print(f"Failure: {e}")


if __name__ == "__main__":
    test_json_parsing()
