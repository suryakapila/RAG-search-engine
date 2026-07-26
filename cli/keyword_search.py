from typing import Any
from collections import Counter
import os
import pickle
import math

from constants import CACHE_DIR, BM25_K1, BM25_B, LIMIT
from utils import stemmed_tokens, filter_stopwords, tokenise_query, load_movies


class InvertedIndex:
    def __init__(self)-> None:
        self.index: dict[str, set] = {}
        self.docmap: dict[int, Any] = {}
        self.term_frequencies:dict[int, Counter] = {}
        self.doc_lengths:dict[int, int] = {}

    def _add_document(self, doc_id: int, text: str) -> None:
        tokens = stemmed_tokens(filter_stopwords(tokenise_query(text)))
        self.doc_lengths[doc_id] = len(tokens)
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
        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), "wb") as f:
            pickle.dump(self.doc_lengths, f)


    def load(self)->None:
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as file:
            self.index = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as file:
            self.docmap = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as file:
            self.term_frequencies = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), "rb") as file:
            self.doc_lengths = pickle.load(file)

    def get_tf(self, doc_id, term)-> int:
        return self.term_frequencies[doc_id].get(term,0)

    def get_bm25_idf(self, term: str) -> float:
        #total_doc_count
        N = len(self.docmap)
        #term_match_doc_count
        df = len(self.get_documents(term))
        score = math.log((N-df+0.5)/(df+0.5) + 1)
        return score

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b = BM25_B):
        tf = self.get_tf(doc_id, term)
        # Length normalization factor
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        #sat_tf = (tf * (k1 + 1)) / (tf + k1)
        return tf_component

    def __get_avg_doc_length(self) -> float:
        doc_total = len(self.doc_lengths)
        if doc_total == 0:
            return 0.0
        doc_lengths_total = 0
        for doc_id in self.doc_lengths:
            doc_lengths_total += self.doc_lengths[doc_id]
        if len(self.docmap) != len(self.doc_lengths):
            print(f"docmap count: {len(self.docmap)}, doc_lengths count: {len(self.doc_lengths)}")
                        
        #print(f"total_docs={len(self.doc_lengths)}, total_tokens={doc_lengths_total}, avg={doc_lengths_total/doc_total}")

        return doc_lengths_total/doc_total

    def bm25(self,doc_id,term)-> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        
        #if doc_id == 2275:
         #   print(f"{doc_id}: {term}")
         #  print(f"bm25_tf = {bm25_tf}")
         #   print(f"bm25_idf = {bm25_idf}")
         #   print(f"{bm25_tf * bm25_idf}")
         #   print(f"{bm25_tf * bm25_idf:.2f}")
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = stemmed_tokens(filter_stopwords(tokenise_query(query)))
        scores:dict[int, float] = {}
        for doc_id in self.docmap:
            total_score = 0
            for token in tokens:
                score = self.bm25(doc_id, token)
                total_score += score
            scores[doc_id] = total_score
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]


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

def bm25_idf_command(term):
    index = InvertedIndex()
    index.load()
    token = tokenise(term)
    score= index.get_bm25_idf(token)
    return score

def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()
    index.load()
    token = tokenise(term)
    bm25_tf = index.get_bm25_tf(doc_id, token, k1, b)
    return bm25_tf

def bm25_search_command(query, limit = LIMIT):
    index = InvertedIndex()
    index.load()
    ranked = index.bm25_search(query, limit)
    return [(doc_id, index.docmap[doc_id]["title"], score) for doc_id, score in ranked]