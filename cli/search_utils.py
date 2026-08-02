from constants import SCORE_PRECISION


def format_search_results(top_movies, documents, chunk_metadata):
    results = []
    for movie_idx, score in top_movies:
        doc_id = documents[movie_idx]["id"]
        title = documents[movie_idx]["title"]
        document = documents[movie_idx]["description"][:100]
        metadata = chunk_metadata.get(doc_id, [])
        results.append({
            "id": doc_id,
            "title": title,
            "document": document,
            "score": round(float(score), SCORE_PRECISION),
            "metadata": metadata or {},
        })
    return results
