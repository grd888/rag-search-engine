from PIL import Image
from sentence_transformers import SentenceTransformer

class MultiModalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.image_model = SentenceTransformer("clip-ViT-B-32")
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def embed_image(self, image_path: str) -> list[float]:
        image = Image.open(image_path)
        return self.image_model.encode([image])[0]


def verify_image_embedding(image_path: str) -> None:
    msm = MultiModalSearch()
    embedding = msm.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Last 5 values: {embedding[-5:]}")
    print(f"All values are finite: {all(map(lambda x: x == x, embedding))}")