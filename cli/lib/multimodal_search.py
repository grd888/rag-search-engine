from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies
import numpy as np


class MultiModalSearch:
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.documents = documents
        self.model = SentenceTransformer("clip-ViT-B-32")
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> list[float]:
        image = Image.open(image_path)
        return self.model.encode([image], show_progress_bar=False)[0]

    def search_with_image(self, image_path: str, top_k: int = 5) -> list[dict]:
        image_embedding = self.embed_image(image_path)
        scores = []
        for text_embedding in self.text_embeddings:
            # calculate cosine similarity
            score = cosine_similarity(image_embedding, text_embedding)
            scores.append(score)
        results = []
        for i, score in enumerate(scores):
            doc = self.documents[i]
            results.append(
                {
                    "score": score,
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"][:100] + "...",
                }
            )
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]


def verify_image_embedding(image_path: str) -> None:
    msm = MultiModalSearch([])
    embedding = msm.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Last 5 values: {embedding[-5:]}")
    print(f"All values are finite: {all(map(lambda x: x == x, embedding))}")


def image_search_command(image_path: str, top_k: int = 5) -> None:
    movies = load_movies()
    msm = MultiModalSearch(movies)
    results = msm.search_with_image(image_path, top_k)
    return results


def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
