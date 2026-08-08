import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder


load_dotenv()

MODEL = "openrouter/free"


def _client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def rewrite_query(query:str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep the rewritten query concise (under 10 words)
    - It should be a Google-style search query, specific enough to yield relevant results
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    If you cannot improve the query, output the original unchanged.
    Output only the rewritten query text, nothing else.

    User query: "{query}"
    """
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()

def expand_query(query):
    prompt = f"""Expand the user-provided movie search query below with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    Output only the additional terms; they will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    User query: "{query}"
    """
    response = _client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
    return response.choices[0].message.content.strip()

def rerank_rrf(query, doc):
    prompt = f"""Rate how well this movie matches the search query.
    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("document", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""
    response = _client().chat.completions.create(
        model = MODEL,
        messages = [{"role": "user", "content":prompt}]
    )
    text = response.choices[0].message.content.strip()
    try:
        return float(text)
    except ValueError:
        return 0.0
    
def rerank_batch(query, candidates):
    """Ask the LLM to rank candidate movies by relevance.

    Returns a list of candidate ids ordered best-match-first. Any ids the model
    omits are appended in their original order so no candidate is dropped; unknown
    ids returned by the model are ignored. On a parse failure the original order
    is returned unchanged.
    """
    doc_list_str = "\n".join(
        f'{c["id"]}: {c["title"]} - {c["document"]}' for c in candidates
    )
    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return the movie IDs in order of relevance, best match first.

    Your response must be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    response = _client().chat.completions.create(
        model = MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()

    original_ids = [c["id"] for c in candidates]
    try:
        ranked_ids = [int(x) for x in json.loads(text)]
    except (json.JSONDecodeError, ValueError, TypeError):
        return original_ids

    valid = set(original_ids)
    seen = set()
    ordered = []
    for doc_id in ranked_ids:
        if doc_id in valid and doc_id not in seen:
            ordered.append(doc_id)
            seen.add(doc_id)
    # Append any candidates the model dropped, preserving original order.
    ordered.extend(doc_id for doc_id in original_ids if doc_id not in seen)
    return ordered
    
    
def rerank_cross_encoder(query, candidates):
    """Score candidates with a local cross-encoder model.

    Returns one relevance score per candidate, in the same order as the input
    (higher = more relevant). Runs offline with no API calls, scoring each
    (query, doc) pair directly.
    """
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    pairs = [
        [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
        for doc in candidates
    ]
    # `predict` returns one relevance score per pair (higher = more relevant).
    scores = cross_encoder.predict(pairs)
    return [float(s) for s in scores]
    
    

def enhance_query(query: str, method: str) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query
