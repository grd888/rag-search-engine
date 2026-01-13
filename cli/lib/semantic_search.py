
from sentence_transformers import SentenceTransformer

def verify_model() -> None:
  semantic_search = SemanticSearch()
  print(f"Model Loaded: {semantic_search.model}")
  print(f"Max sequence length: {semantic_search.model.max_seq_length}")

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        