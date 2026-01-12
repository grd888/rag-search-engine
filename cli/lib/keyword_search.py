import string
import os
import pickle
from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, CACHE_DIR, load_movies, load_stop_words

_STOP_WORDS = None
_STEMMER = PorterStemmer()


def _get_stop_words() -> list[str]:
    global _STOP_WORDS
    if _STOP_WORDS is None:
        _STOP_WORDS = load_stop_words()
    return _STOP_WORDS


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except FileNotFoundError:
        print("Inverted index not found. Please run 'build' command first.")
        exit(1)
    seen, results = set(), []

    query_tokens = tokenize_text(query)

    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in seen:
                results.append(idx.doc_map[doc_id])
                seen.add(doc_id)
            if len(results) >= limit:
                break

    return results


def has_matching_tokens(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        if query_token in title_tokens:
            return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    stop_words = _get_stop_words()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = [
        _STEMMER.stem(token) for token in tokens if token and token not in stop_words
    ]
    return valid_tokens


class InvertedIndex:
    def __init__(self) -> None:
        # a mapping of tokens to document IDs
        self.index: dict[str, set[int]] = {}
        # a mapping of document IDs to their full document objects
        self.doc_map: dict[int, dict] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def _add_document(self, doc_id: int, text: str):
        # tokenize the document text
        tokens = tokenize_text(text)
        # add each token to the index with the document ID
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        """
        Get the set of document IDs for a given term, and return them as a list,
        sorted in ascending order. (Assume input is a single word/token)
        """
        return sorted(list(self.index.get(term, set())))

    def build(self) -> None:
        """
        Build the inverted index from the documents in the dataset.
        """
        movies = load_movies()
        for movie in movies:
            # concatenate the title and description
            self._add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.doc_map[movie["id"]] = movie

    def save(self) -> None:
        """
        Save the inverted index to a file.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.doc_map, f)

    def load(self) -> None:
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        if not os.path.exists(self.docmap_path):
            raise FileNotFoundError(f"Document map file not found: {self.docmap_path}")
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.doc_map = pickle.load(f)
