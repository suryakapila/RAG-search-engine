
# rag-search-engine

Building a retrieval stack from scratch — sparse ranking, dense retrieval, and their fusion — as the foundation for a full RAG system. The point is to *build* the algorithms, not wire up a framework: each phase's limitation motivates the next.

- **Sparse** — inverted index, TF-IDF, BM25
- **Dense** — `all-MiniLM-L6-v2` embeddings, cosine similarity, sentence chunking
- **Hybrid** — weighted-sum fusion (`α`) and Reciprocal Rank Fusion
- **Query enhancement** — LLM spell-fix / rewrite / expansion before retrieval
- **Reranking** — a precision pass over the RRF candidate pool: LLM per-doc, LLM batch, and a local cross-encoder
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

### Query enhancement — fix the query before you search

Retrieval can only work with the words it's given. A typo (`"shawshenk redemtion"`), a vague phrasing, or a query whose vocabulary doesn't match the corpus all sink recall before ranking even runs. So `rrf-search --enhance` runs the query through an LLM first:

- **`spell`** — high-confidence typo fixes only, nothing rewritten.
- **`rewrite`** — recast as a specific, searchable query ("that bear movie where leo gets attacked" → "The Revenant Leonardo DiCaprio bear attack").
- **`expand`** — append related terms/synonyms to widen recall.

The enhanced query is printed (`Enhanced query (spell): '…' -> '…'`) so the effect is visible, then RRF runs on it. Enhancement trades a network round-trip and some non-determinism for recall — most useful when the raw query is malformed or under-specified.

### Reranking — a precision pass over the top

RRF is a *fusion* step: it never actually reads the query against a document, it only combines the ranks two shallow retrievers assigned. That's great for recall but coarse at the very top — the ordering among the best few is decided by rank arithmetic, not by how well each document answers *this* query. Reranking fixes that by taking a **candidate pool** from RRF (`--rerank-pool × --limit`) and re-scoring just those with a model that reads query and document *together*, then truncating to `--limit`. Expensive per item, so it only ever touches the shortlist.

Three methods, trading cost against quality:

- **`individual`** (LLM-as-judge) — one API call per candidate: "rate this movie 0–10 for this query." Highest-fidelity single judgments and gives a real magnitude, but it's `N` calls, slow, and rate-limited (hence `--rerank-sleep`).
- **`batch`** (LLM, one shot) — a single call ranks the whole pool and returns an ordering of ids. Far cheaper and faster than per-doc, but the model must juggle every candidate in one context (weaker on long lists) and emit parseable JSON — so it needs id-validation and a fallback to the original order, and it yields only a *rank*, no score.
- **`cross_encoder`** (local model) — `cross-encoder/ms-marco-TinyBERT-L2-v2` scores each `(query, doc)` pair directly. No API, no rate limits, deterministic; outputs an unbounded relevance logit (negatives are normal — only the ordering matters).

**Cross-encoder vs. LLM reranking.** A cross-encoder is a small transformer *trained for relevance*: query and document go in as one joint sequence, a scalar comes out. An LLM reranker *repurposes* a general model to judge relevance through a prompt. The differences that matter:

| | Cross-encoder | LLM (individual / batch) |
|---|---|---|
| Where it runs | Local, offline | API round-trip(s) |
| Cost | Free after a one-time ~download | Tokens per call |
| Latency | Whole pool in one batched forward pass (ms) | `N` calls / one call |
| Determinism | Deterministic | Stochastic; batch can drop or hallucinate ids |
| Output | Relevance logit per doc | 0–10 score (individual) / ordering only (batch) |
| Strength | Fine-grained query–doc token interaction | World knowledge, intent, instruction-following ("keep it *family* friendly") |
| Weakness | Bound to its training domain (MS MARCO web text, not movie plots) | Slower, pricier, needs output validation |

In short: the cross-encoder is the fast, cheap, reliable default and scales to reranking large pools; LLM reranking is the reach-for-it option when relevance genuinely needs reasoning or instruction-following that a purpose-built ranker can't express. The classic production shape is **retrieve broadly (RRF) → rerank the shortlist (cross-encoder) → optionally an LLM pass on the very top**.

## Running it

Python 3.14 + [`uv`](https://docs.astral.sh/uv/). First semantic/hybrid run downloads the model and builds embeddings into `cache/` (one-time).

```bash
uv sync

uv run python cli/keyword_search_cli.py search "british bear"
uv run python cli/semantic_search_cli.py search_chunked "british bear" --limit 10
uv run python cli/hybrid_search_cli.py weighted-search "british bear" --alpha 0.5 --limit 10
uv run python cli/hybrid_search_cli.py rrf-search "british bear" -k 60 --limit 10

# fix the query before searching
uv run python cli/hybrid_search_cli.py rrf-search "the shawshenk redemtion" --enhance spell --limit 10

# rerank the RRF candidate pool (pool = --rerank-pool × --limit)
uv run python cli/hybrid_search_cli.py rrf-search "family movie about bears" --rerank-method cross_encoder --limit 10
uv run python cli/hybrid_search_cli.py rrf-search "family movie about bears" --rerank-method batch --limit 10
uv run python cli/hybrid_search_cli.py rrf-search "family movie about bears" --rerank-method individual --limit 10 --rerank-sleep 3
```

Reranking needs an `OPENROUTER_API_KEY` (for `individual`/`batch`); `cross_encoder` runs fully offline after the model downloads on first use.

## Roadmap

- **Phase 2 — Productionise.** Loader abstraction (PDF/JSON/MD/HTML), structure-aware chunkers, a real vector store (Chroma → Qdrant), and an eval harness (recall@k, faithfulness) before swapping models.
- **Phase 3 — Agentic.** Query enhancement and reranking (cross-encoder + LLM-as-judge) are in (see above); **generation** over the reranked context via OpenRouter is next.
