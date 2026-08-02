from typing import TypedDict
import string
import json
from nltk.stem import PorterStemmer

from constants import DATA_PATH, STOPWORDS_PATH

stemmer = PorterStemmer()
# Create a translation table that maps all punctuation to None
translator = str.maketrans('', '', string.punctuation)


class Movie(TypedDict):
    id: int
    title: str
    description: str


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords():
    with open(STOPWORDS_PATH, "r") as file:
        return {preprocess_text(line) for line in file.read().splitlines()}


def tokenise_query(query):
    res = preprocess_text(query)
    return res.split()

def preprocess_text(str):
    return str.lower().translate(translator)

def filter_stopwords(query):
    stopwords = load_stopwords()
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
