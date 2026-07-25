from typing import Any, TypedDict
from collections import Counter
import os
import string
import json
import pickle
from nltk.stem import PorterStemmer
import math

stemmer = PorterStemmer()
# Create a translation table that maps all punctuation to None
translator = str.maketrans('', '', string.punctuation)



class InvertedIndex:
    def __init__(self)-> None:
        self.index: dict[str, set] = {}
        self.docmap: dict[int, Any] = {}
        self.term_frequencies:dict[int, Counter] = {}

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = stemmed_tokens(filter_stopwords(tokenise_query(text)))
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1 

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def build(self) -> None:
        for movie in load_movies():
            self._add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as f:
            pickle.dump(self.index, f)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as f:
            pickle.dump(self.docmap, f)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self)->None:
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as file:
            self.index = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as file:
            self.docmap = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as file:
            self.term_frequencies = pickle.load(file)
    
    def get_tf(self, doc_id, term)-> int:
        return self.term_frequencies[doc_id].get(term,0)

    

class Movie(TypedDict):
    id: int
    title: str
    description: str



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")




def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as file:
        return file.read().splitlines()


def tokenise_query(query):
    res = preprocess_text(query)
    return res.split() 

def preprocess_text(str):
    return str.lower().translate(translator)

def has_matching_tokens(query_tokens, title_tokens):
    for title in title_tokens:
        for query in query_tokens:
            if query in title:
                return True
    return False

def filter_stopwords(query):
    p_stopwords = []
    stopwords = load_stopwords()
    for word in stopwords:
        p_stopwords.append(preprocess_text(word))
    res = []
    for word in query:
        if word not in stopwords:
            res.append(word)
    return res

def stemmed_tokens(query):
    res = []
    for token in query:
        word = stemmer.stem(token)
        res.append(word)
    return res


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()
    print(f"Built index: {len(index.index)} terms over {len(index.docmap)} movies")
    print(f"Saved to {CACHE_DIR}")

def tokenise(term):
    token = stemmed_tokens(filter_stopwords(tokenise_query(term)))
    if len(token) != 1:
        raise ValueError("The result has more than one token")
    return token[0]

def tf(doc_id, term):
    index = InvertedIndex()
    index.load()
    token = tokenise(term)
    return index.get_tf(doc_id, token)

def idf(term):
    index = InvertedIndex()
    index.load()
    token = tokenise(term)
    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(token))
    idf = math.log((total_doc_count+1)/(term_match_doc_count+1))
    return idf
    
def tfidf(doc_id, term):
    index = InvertedIndex()
    index.load()
    tf_score = tf(doc_id, term)
    idf_score = idf(term)
    tf_idf = tf_score *idf_score
    return tf_idf

