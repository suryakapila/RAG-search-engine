import json
import os
import re

import numpy as np
from constants import (
    CACHE_DIR,
    CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    LIMIT,
    OVERLAP,
)
from search_utils import format_search_results
from sentence_transformers import SentenceTransformer
from utils import load_movies


class SemanticSearch():
    def __init__(self, model_name: str = "all-MiniLM-L6-v2")-> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text):
        if not text.strip():
            raise ValueError("Invalid text input")
        embeddings = self.model.encode([text])
        return embeddings[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        movies = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie = f"{doc['title']}: {doc['description']}"
            movies.append(movie)
        self.embeddings = self.model.encode(movies, show_progress_bar = True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "movie_embeddings.npy"), "wb") as f:
            np.save(f, np.array(self.embeddings))
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "rb") as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
            
        
        # create a list of tuples (similarity, doc_id) and sort it in descending order
        sorted_docs = sorted(
            [(similarity, self.documents[i]["id"]) for i, similarity in enumerate(similarities)],
            key=lambda x: x[0],
            reverse=True
        )
        # return the top limit results as a list of dict , each containing score, title, decription
        top_docs = sorted_docs[:limit]
        
        return [
            {
                "score": score,
                "title": self.document_map[doc_id]["title"],
                "description": self.document_map[doc_id]["description"]
            }
            for score, doc_id in top_docs
        ]   
        

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        
    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        chunks= []
        metadata: list[dict] = []
        self.documents = documents
        for movie_idx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc
            if doc['description'] == "":
                continue
            semantically_chunked = semantic_chunk(doc['description'], 4, 1)
            for i, chunk in enumerate(semantically_chunked):
                chunks.append(chunk)
                metadata.append({
                    "movie_idx": movie_idx,
                    "movie_id": doc["id"],
                    "chunk_idx": i,
                    "total_chunks": len(semantically_chunked),
                })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar = True)
        self.chunk_metadata = metadata
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "chunk_embeddings.npy"), "wb") as f:
            np.save(f, np.array(self.chunk_embeddings))
        with open(os.path.join(CACHE_DIR, "chunk_metadata.json"), "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(chunks)}, f, indent=2)   
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(os.path.join(CACHE_DIR, "chunk_embeddings.npy")) and os.path.exists(os.path.join(CACHE_DIR, "chunk_metadata.json")):
            with open(os.path.join(CACHE_DIR, "chunk_embeddings.npy"), "rb") as f:
                self.chunk_embeddings = np.load(f)
            with open(os.path.join(CACHE_DIR, "chunk_metadata.json"), "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents) 
    
    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        movie_score = {}
        for chunk_idx, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            movie_idx = self.chunk_metadata[chunk_idx]["movie_idx"]
            if movie_idx not in movie_score or score > movie_score[movie_idx]:
                movie_score[movie_idx] = score
        sorted_movies = sorted(movie_score.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_movies[:limit]

        chunk_metadata_by_doc_id: dict[int, list] = {}
        for meta in self.chunk_metadata:
            chunk_metadata_by_doc_id.setdefault(meta["movie_id"], []).append(meta)

        return format_search_results(top_movies, self.documents, chunk_metadata_by_doc_id)


def verify_model():
    semantic = SemanticSearch()
    print(f"Model loaded: {semantic.model}")
    print(f"Max sequence length: {semantic.model.max_seq_length}")

def embed_text(text):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    
def verify_embeddings():
    semantic = SemanticSearch()
    documents = load_movies()
    embeddings = semantic.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    
def embed_query_text(query):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)
    
def search(query, limit =  LIMIT):
    semantic = SemanticSearch()
    documents = load_movies()
    semantic.load_or_create_embeddings(documents)
    results = semantic.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {limit} results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} (score: {result['score']:.4f})  {result['description']}")
        
def chunk(text, chunk_size = CHUNK_SIZE, overlap = OVERLAP):
    if not text.strip():
        raise ValueError("Invalid text input")
    step = chunk_size - int(overlap)
    if step <= 0:
        raise ValueError("Overlap must be less than chunk size")
    words = text.split()
    chunks = []
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i + chunk_size]))
        
    print(f"Chunking {len(text)} characters")
    for i, c in enumerate(chunks, start=1):
        print(f"{i}. {c}")
    
    
def semantic_chunk(text: str,max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,overlap: int = DEFAULT_CHUNK_OVERLAP,) -> list[str]:
    strip_text = text.strip()
    if strip_text == "":
        return []
    sentences = re.split(r"(?<=[.!?])\s+", strip_text)        
    chunks = []
    i = 0
    n_sentences = len(sentences)
    if len(sentences) == 1 and  not sentences[0].endswith(('!', '?', '.')):
        sentence = sentences[0].strip()
        chunks.append(sentence)
    else:
        while i < n_sentences:
            chunk_sentences = sentences[i : i + max_chunk_size]
            if chunks and len(chunk_sentences) <= overlap:
                break
            stripped_chunk_sentences = []
            for sentence in chunk_sentences:
                strip_sentence = sentence.strip()
                if strip_sentence != "":
                    stripped_chunk_sentences.append(strip_sentence)
            if len(stripped_chunk_sentences):
                chunks.append(" ".join(stripped_chunk_sentences))
            i += max_chunk_size - overlap
    return chunks

def embed_chunks():
    semantic = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = semantic.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")
    
def search_chunked(query: str, limit: int = LIMIT):
    documents = load_movies()
    semantic = ChunkedSemanticSearch()
    semantic.load_or_create_chunk_embeddings(documents)
    results = semantic.search_chunks(query, limit)
    return results