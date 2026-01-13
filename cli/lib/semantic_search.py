import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_DIR, load_movies
import os

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

def verify_model() -> None:
  semantic_search = SemanticSearch()
  print(f"Model Loaded: {semantic_search.model}")
  print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str) -> None:
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
  semantic_search = SemanticSearch()
  documents = load_movies()
  embeddings = semantic_search.load_or_create_embeddings(documents)
  print(f"Number of docs:   {len(documents)}")
  print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        
    def generate_embedding(self, text: str):
      # if input text is empty or contains only whitespace, raise a ValueError
      
      if not text or not text.strip():
        raise ValueError("Input text cannot be empty or contain only whitespace")
      
      # generate and return the embedding
      embeddings = self.model.encode([text])
      return embeddings[0]
    
    def build_embeddings(self, documents: list[dict]):
      # store the documents
      self.documents = documents
      # create a mapping from document id to document text
      self.document_map = {doc['id']: doc for doc in documents}
      doc_list = [f"{doc['title']}: {doc['description']}" for doc in documents]
      # generate embeddings for all documents
      self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
      os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
      np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
      return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]):
      # store the documents
      self.documents = documents
      # create a mapping from document index to document text
      self.document_map = {i: doc for i, doc in enumerate(documents)}
      # check if embeddings exist
      
      if os.path.exists(os.path.join(CACHE_DIR, "movie_embeddings.npy")):
        self.embeddings = np.load(os.path.join(CACHE_DIR, "movie_embeddings.npy"))
        if len(self.embeddings) == len(documents):
          return self.embeddings
      # if not, build embeddings
      return self.build_embeddings(documents)