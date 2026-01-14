import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import (
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    load_movies,
)
import os
import re

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


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
        print(
            "\t"
            + (
                result["description"][:100] + "..."
                if len(result["description"]) > 100
                else result["description"]
            )
        )
        print()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []
    # When overlap is greater than zero, each previous chunk should share the last overlap words with the following chunk.
    # For example, if the text is "The quick brown fox jumps over the lazy dog" and the chunk size is 5 and the overlap is 2,
    # the chunks should be ["The quick brown fox jumps", "fox jumps over the lazy", "over the lazy dog"]

    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    # Each chunk should contain up to max_chunk_size sentences
    # Support overlap by number of sentences
    for i in range(0, len(sentences), max_chunk_size - overlap):
        current_chunk = sentences[i : i + max_chunk_size]
        if chunks and len(current_chunk) <= overlap:
            break
        chunks.append(" ".join(current_chunk))

    return chunks

def embed_chunks_command():
    movies = load_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]):
        # store the documents
        self.documents = documents
        # create a mapping from document id to document text
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []
        
        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue
            
            chunks = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )
            # track which document each chunk came from
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )
        # generate embeddings for all chunks
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(CHUNK_METADATA_PATH), exist_ok=True)

        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        # store the documents
        self.documents = documents
        # create a mapping from document id to document text
        self.document_map = {doc["id"]: doc for doc in documents}
        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
