# rag-search-engine

Building a retrieval stack from scratch — sparse ranking, dense retrieval, and their fusion — as the foundation for a full RAG system. The point is to *build* the algorithms, not wire up a framework: each phase's limitation motivates the next.

- **Sparse** — inverted index, TF-IDF, BM25
- **Dense** — `all-MiniLM-L6-v2` embeddings, cosine similarity, sentence chunking
- **Hybrid** — weighted-sum fusion (`α`) and Reciprocal Rank Fusion
- Python 3.14, `uv`, argparse CLIs, cached indexes/embeddings

## Dataset

~10k movies (~26 MB) at `data/movies.json`, shaped `{"movies": [{"id", "title", "description", ...}]}`. `data/` and `cache/` are gitignored — datasets and derived artifacts stay out of version control.

## The phases

### Sparse — keyword search

Inverted index (`token → doc_ids` + per-doc term frequencies/lengths), cached to `cache/*.pkl`. Ranked by **BM25**, which improves on TF-IDF with two knobs:

- `k1` (1.5) — term-frequency **saturation**: the 10th mention isn't 10× the 1st.
- `b` (0.75) — **length normalisation**: long docs don't win on size alone.

**Where it breaks:** matches surface tokens only. `"british bear"` can't find *"a cub from the UK"*. → dense retrieval.

### Dense — semantic search

Encode query and docs as 384-dim vectors, rank by cosine similarity (magnitude-invariant, so only direction matters). Two strategies: one embedding per movie, or **chunked** — split descriptions into sentence windows, embed each, take the best chunk score per movie. Chunking preserves signal a single doc-level embedding would average away.

**Where it breaks:** too forgiving. Rare terms, proper nouns, and exact-match intent get diluted by conceptually-close-but-wrong results. → fuse the two.

### Hybrid — weighted sum

Sparse and dense fail differently, so combine them. BM25 scores are unbounded; cosine is `[-1, 1]` — not comparable. Min-max normalise each side per query, then:

```
hybrid(d) = α · bm25_norm(d) + (1 - α) · semantic_norm(d)
```

`α = 1` pure BM25, `α = 0` pure semantic, `α = 0.5` balanced. Downside: `α` is an unlearned global knob, and normalisation is lossy — it stretches even a batch of uniformly weak scores to `[0, 1]`.

### Hybrid — Reciprocal Rank Fusion

Score normalisation is messy: it's skewed by outliers and by how peaked each distribution is. RRF skips scores entirely and fuses on **rank**:

```
rrf(d) = Σ  1 / (k + rank_i(d))
```

Summed across the sources that surfaced `d`; `k` (60) dampens how much the top ranks dominate.

**Why RRF over weighted sum:**
- No per-query normalisation — reads only the ordering each retriever is actually good at.
- No `α` to tune; `k` is a mild, well-behaved dampener.
- Immune to outliers — rank 1 is rank 1 whether it won by a landslide or a hair.
- Single-source docs just contribute one term — no "fill the missing side with 0.0" bias.

**Trade-off:** RRF discards magnitude ("barely relevant at rank 3" ≡ "overwhelmingly relevant at rank 3"). When scores are genuinely well-calibrated, weighted sum can use that and RRF can't — but across heterogeneous retrievers, robustness usually wins.

*Example* (`"a bear from peru"`, `k=60`): *Paddington* tops out at `0.033` — rank 1 on **both** sides (`1/61 + 1/61`). The two retrievers agreeing is what carries it, not any raw score.

## Running it

Python 3.14 + [`uv`](https://docs.astral.sh/uv/). First semantic/hybrid run downloads the model and builds embeddings into `cache/` (one-time).

```bash
uv sync

uv run python cli/keyword_search_cli.py search "british bear"
uv run python cli/semantic_search_cli.py search_chunked "british bear" --limit 10
uv run python cli/hybrid_search_cli.py weighted-search "british bear" --alpha 0.5 --limit 10
uv run python cli/hybrid_search_cli.py rrf-search "british bear" -k 60 --limit 10
```

## Roadmap

- **Phase 2 — Productionise.** Loader abstraction (PDF/JSON/MD/HTML), structure-aware chunkers, a real vector store (Chroma → Qdrant), and an eval harness (recall@k, faithfulness) before swapping models.
- **Phase 3 — Agentic.** Reranking (cross-encoder or LLM-as-judge) and generation via OpenRouter.
