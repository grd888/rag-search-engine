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
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int) -> None:
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)
    """
    Print results in this format:
    1. Spaceflight IC-1: An Adventure in Space (score: 0.4406)
       The opening narrative is given by a man in a high ranking military uniform. He tells us the film is ...

    2. Adventureland (score: 0.4150)
       In 1987, James Brennan (Jesse Eisenberg) has two plans. The first plan is to have a summer vacation ...
    """
    for i, result in enumerate(results, 1):
        print(f"{i}. \t{result['title']} (score: {result['score']})")
        # truncate description to 100 characters and add trailing ellipsis
        print("\t" + (result['description'][:100] + "..." if len(result['description']) > 100 else result['description']))
        print()

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
        embeddings = self.model.encode([text.strip()])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        # store the documents
        self.documents = documents
        # create a mapping from document id to document text
        self.document_map = {doc["id"]: doc for doc in documents}
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

    def search(self, query: str, k: int = 5):
        if self.embeddings is None:
            raise ValueError(
                "Embeddings not loaded. Call load_or_create_embeddings first."
            )
        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )
            
        query_embedding = self.generate_embedding(query)
        # Create a list of (similarity_score, document) tuples.

        similarity_document_pairs = [
            (cosine_similarity(query_embedding, emb), doc)
            for emb, doc in zip(self.embeddings, self.documents)
        ]
        # Sort by similarity score in descending order
        similarity_document_pairs.sort(key=lambda x: x[0], reverse=True)

        """
        Return the top results (up to limit) as a list of dictionaries, each containing:
          score: The cosine similarity score
          title: The movie title
          description: The movie description
        """
        top_k_pairs = similarity_document_pairs[:k]
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in top_k_pairs
        ]
