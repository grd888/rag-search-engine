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
    docs = inverted_index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    query_tokens = tokenize_text(query)

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_tokens(query_tokens, title_tokens):
            results.append(movie)
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

    def build(self):
        """
        Build the inverted index from the documents in the dataset.
        """
        movies = load_movies()
        for movie in movies:
            # concatenate the title and description
            self._add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.doc_map[movie["id"]] = movie

    def save(self):
        """
        Save the inverted index to a file.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.doc_map, f)
