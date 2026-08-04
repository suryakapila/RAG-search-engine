# rag-search-engine

Building a retrieval stack from scratch — sparse ranking, dense retrieval, and their fusion — as the foundation for a full RAG system.

## At a glance

- **Sparse retrieval** — inverted index, TF-IDF, and BM25 (with `k1` saturation and `b` length normalisation)
- **Dense retrieval** — `all-MiniLM-L6-v2` sentence-transformer embeddings, cosine similarity, sentence-level chunking with best-chunk aggregation
- **Hybrid ranking** — min-max score reconciliation and weighted-sum fusion with a tunable `α`
- Python 3.14, `uv`-managed environment, argparse CLIs, `pickle`/`numpy`/`json` caches

## Motivation

The point of this project is to understand retrieval by *building* the algorithms, not by wiring together a framework. Each phase implements a technique end-to-end so its limitations directly motivate the next — sparse ranking exposes the paraphrase problem, dense retrieval exposes the exact-match problem, and hybrid fusion exists precisely because those failure modes are complementary.

## Dataset

A public JSON dataset of ~10k movies (~26 MB), living at `data/movies.json`:

```json
{
  "movies": [
    { "id": 1, "title": "...", "description": "...", ... },
    ...
  ]
}
```

`data/` and `cache/` are both gitignored — datasets and derived artifacts (indexes, embeddings) don't belong in version control.

---

## Phase 1a — Keyword search

Classical sparse retrieval. Deterministic, interpretable, no model.

### Pipeline

1. Load documents.
2. Preprocess: lowercase → tokenise → filter stopwords → Porter-stem.
3. Build an **inverted index**: `token → set(doc_ids)`, plus per-doc term frequencies and lengths.
4. Persist to `cache/*.pkl` so queries don't rebuild.
5. Score with TF-IDF or BM25 and rank.

### TF-IDF

```
tfidf(t, d) = tf(t, d) * log((N + 1) / (df(t) + 1))
```

- `tf(t, d)` — raw count of term `t` in document `d`
- `df(t)` — number of documents containing `t`
- `N` — total documents
- The additive `+1` avoids `log(0)` and division-by-zero

Intuition: reward terms frequent in *this* document but rare *across the corpus*.

### BM25

Two refinements over TF-IDF.

**IDF with negative-avoidance smoothing:**
```
idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

**TF with saturation and length normalisation:**
```
tf_component(t, d) = tf(t, d) * (k1 + 1)
                     ---------------------------------------------
                     tf(t, d) + k1 * (1 - b + b * (|d| / avgdl))
```

- `k1` (default `1.5`) — **term-frequency saturation**. The 10th mention of a term shouldn't be 10× more valuable than the 1st.
- `b` (default `0.75`) — **length normalisation**. `b = 0` disables it; `b = 1` fully penalises documents longer than `avgdl`.

Final: `bm25(t, d) = tf_component(t, d) * idf(t)`, summed across query terms.

### Where sparse retrieval stops working

It matches surface tokens only. A query `"british bear"` cannot find a movie described as *"a cub from the UK"* — no vocabulary overlap. Synonyms, paraphrase, and vaguer intent all fall through. That gap motivates dense retrieval.

---

## Phase 1b — Semantic search

Encode query and documents as fixed-dimensional vectors, rank by cosine similarity.

### Model

`all-MiniLM-L6-v2` — 384-dim, ~90MB, fast on CPU. Not the strongest embedder available but small and quick enough for iterative experimentation. The `SentenceTransformer` interface makes it swappable.

### Similarity

```
cosine(u, v) = (u · v) / (||u|| * ||v||)   ∈ [-1, 1]
```

Cosine is chosen over raw dot-product so vector magnitudes don't skew ranking — only direction matters.

### Two indexing strategies

- **`SemanticSearch`** — one embedding per movie (`title + description` concatenated).
- **`ChunkedSemanticSearch`** — split each description into sentence-based chunks, embed each chunk, and at query time take the *best chunk score per movie* as that movie's score.

Chunking matters when relevant content is a small portion of a long description; a single doc-level embedding averages that signal away.

### Chunking

Sentence-boundary splitting via `re.split(r"(?<=[.!?])\s+", text)`, then packing sentences into windows of `max_chunk_size = 4` with configurable overlap. Not production-grade — a real system would use token-aware or structure-aware splitters (paragraph → sentence → word) with configurable overlap — but the abstraction is in place to swap later.

### Where dense retrieval stops working

Dense retrieval is *too* forgiving. Rare terms, proper nouns, and exact-match intent (product IDs, unique names, precise phrases) get diluted by conceptually related but literally wrong results. It also inherits the embedder's biases wholesale.

---

## Phase 1c — Hybrid search

### The score-scale problem

BM25 scores are unbounded positive floats — corpus- and query-dependent. Cosine similarities live in `[-1, 1]`. Summing them directly would let BM25 dominate on magnitude alone, regardless of relative quality. Both sides need to be brought onto a common scale first.

### Min-max normalisation

```
norm(s) = (s - min(S)) / (max(S) - min(S))
```

Applied independently to each side's candidate scores. Edge cases:
- Empty input → empty output.
- `min == max` → all `1.0` (tied at the top of their side).

### Weighted-sum fusion

```
hybrid(d) = α * bm25_norm(d) + (1 - α) * semantic_norm(d)
```

- `α = 1.0` → pure BM25
- `α = 0.0` → pure semantic
- `α = 0.5` → balanced

### Merge flow

1. Fetch a wide candidate pool (`500 * limit`) from each side.
2. Normalise each side's scores independently.
3. Merge on `doc_id`; documents present in only one side get `0.0` on the missing side.
4. Compute `hybrid` per merged doc.
5. Sort descending, return top `limit`.

### Observed behaviour on `"british bear"`

- `α = 0.2` (semantic-heavy): `Legends of the Fall` (semantic=1.000) climbs to #2 despite weak keyword overlap.
- `α = 0.8` (keyword-heavy): `The Duchess` (bm25=0.707) climbs to #2 despite unrelated content, because "british" hits hard.
- `α = 0.5`: bear-themed films cluster at the top; British-themed films rank mid-list.

### Design tradeoffs and known limitations

- **`500 * limit` candidate pool** is blunt. Fine on ~10k docs; at scale, replace with per-side top-K + a two-stage rerank.
- **Weighted sum vs RRF.** Weighted sum needs comparable scales (hence normalisation); Reciprocal Rank Fusion works on ranks alone and sidesteps calibration entirely — see Phase 1d.
- **`α` is a static knob.** A real system would learn `α` — globally, or per-query via a classifier — rather than treating it as a global constant.
- **Missing-side score = 0.0** biases against docs that only one side surfaces. Alternatives: fill with a low percentile, or restrict to the intersection. Left as a deliberate simplification.

---

## Phase 1d — Reciprocal Rank Fusion

Weighted-sum fusion (Phase 1c) works, but its whole apparatus exists to paper over a single problem: BM25 scores and cosine similarities aren't comparable. RRF sidesteps that problem instead of solving it — it fuses on **rank position alone** and never looks at the raw scores.

### The formula

```
rrf(d) = Σ  1 / (k + rank_i(d))
        i∈sources
```

Each source (BM25, semantic) contributes `1 / (k + rank)` for a document, summed across the sources that surfaced it. `rank` is 1-based; `k` (default `60`) dampens how much the top ranks dominate — larger `k` flattens the curve and lets deeper results matter more.

### Why RRF over weighted score

- **No score calibration.** Weighted sum requires min-max normalisation per side, per query, just to make the two scales addable — and normalisation is itself lossy and query-dependent (a query where every BM25 hit is weak still gets stretched to `[0, 1]`). RRF reads only the ordering, which is exactly what each retriever is actually good at producing. The entire normalisation step disappears.
- **No `α` to tune.** Weighted sum has a free parameter that silently decides the outcome (see the `"british bear"` sweep above); the "right" `α` is query-dependent and unlearned here. RRF has `k`, but `k` is a mild, well-behaved dampener — the results are far less sensitive to it than to `α`.
- **Robust to distribution shape.** Score fusion is skewed by outliers and by how peaked each score distribution is. A single dominant BM25 hit can drag the whole fused ranking. Rank fusion is immune — rank 1 is rank 1 regardless of whether it won by a landslide or a hair.
- **Naturally handles missing sides.** A document found by only one retriever simply contributes one term. There's no "fill the missing side with `0.0`" decision (which, in weighted sum, actively biased against single-source docs).

### What you give up

RRF throws away *magnitude*. "Barely relevant at rank 3" and "overwhelmingly relevant at rank 3" are identical to it. When a retriever's scores are genuinely well-calibrated, weighted sum can exploit that signal and RRF can't. In practice, across heterogeneous retrievers, robustness usually wins — which is why RRF is the common default.

### Merge flow

1. Fetch a wide candidate pool (`500 * limit`) from each side.
2. Record each document's 1-based rank on each side (best chunk rank on the semantic side).
3. Merge on `doc_id`; sum `1 / (k + rank)` across whichever sides surfaced it.
4. Sort descending, return top `limit`.

### Observed behaviour on `"a bear from peru"`

`k = 60`: *Paddington* tops the list at `0.033` — it's rank 1 on **both** sides (`1/61 + 1/61 ≈ 0.0328`), so the two retrievers agreeing is what carries it, not either one's raw score. Documents that only one side ranks highly still appear, but a single strong signal can't outweigh two independent agreements near the top.

---

## Running it

Python 3.14, [`uv`](https://docs.astral.sh/uv/) for env + deps.

```bash
uv sync
```

**Keyword:**
```bash
uv run python cli/keyword_search_cli.py search "british bear"
```

**Semantic:**
```bash
uv run python cli/semantic_search_cli.py search "british bear" --limit 10
uv run python cli/semantic_search_cli.py search_chunked "british bear" --limit 10
```

**Hybrid:**
```bash
uv run python cli/hybrid_search_cli.py weighted-search "british bear" --alpha 0.5 --limit 10
uv run python cli/hybrid_search_cli.py rrf-search "british bear" -k 60 --limit 10
uv run python cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
```

The first semantic/hybrid run downloads the embedding model and builds embeddings into `cache/` — one-time delay.

---

## Roadmap

- **Phase 2 — Productionisation.** Loader abstraction (PDF/JSON/MD/HTML), structure-aware chunkers with parent-document retrieval, migration off `.npy`/`pickle` to a real vector store (Chroma → Qdrant), and an eval harness (recall@k, faithfulness, citation correctness) before swapping models.
- **Phase 3 — Agentic layer.** Reranking (cross-encoder or LLM-as-judge) and LLM generation routed through OpenRouter for model flexibility.

---

## Interview / study recap

A one-page refresher for me. Skim before an IR/RAG discussion.

**Inverted index.** `token → set(doc_ids)`. Preprocess with tokenise/stopword/stem for recall; store per-doc TF and lengths for scoring. Query cost = O(|query terms| × |postings|); good scoring is what you build on top.

**TF-IDF intuition.** Frequent-here + rare-across-corpus = important. Log flattens IDF so a term in half the docs isn't 100× more important than one in every doc.

**BM25 additions over TF-IDF.**
- **Saturation (`k1`)** — the 10th mention of a term shouldn't be 10× more valuable than the 1st.
- **Length normalisation (`b`)** — long documents shouldn't win purely by size.

**Cosine vs dot product.** Cosine cancels magnitude; direction alone determines similarity. Matters when embeddings aren't unit-normalised (many aren't out of the box).

**Why chunk.** Doc-level embeddings average away the specific passage that matches. Chunking preserves local signal; the tradeoff is more storage and a choice of aggregation (max, mean, sum, RRF-over-chunks). Chunk size, overlap, and boundary rules are all tunable and all matter.

**Why hybrid.** Sparse and dense fail differently — sparse misses paraphrase; dense misses exact terms and rare tokens. Fusion reduces both failure modes rather than trading one for the other.

**Score reconciliation.** BM25 magnitudes and cosine similarities aren't comparable. Min-max normalise each side per-query, then combine.

**Weighted sum vs RRF.**
- **Weighted sum:** `α * A + (1 - α) * B`. Needs calibrated / normalised scores. Tunable per query.
- **RRF:** `Σ 1 / (k + rank_i(d))` across sources. Rank-based, no calibration needed. Robust default when scales are hard to compare.

**Common gotchas.**
- Forgetting length normalisation → long-doc bias.
- Mixing normalised and raw scores in the same sum → silent domination.
- Not storing the embedding model name with the vectors → catastrophic when you swap models.
- Chunk boundaries slicing a table, code block, or key phrase → retrieval blindspots.
- Assuming cosine ∈ `[0, 1]` — it's `[-1, 1]`, and negative similarities do occur.
