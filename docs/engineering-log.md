# Engineering log

A running record of why each phase of `rag-search-engine` moves the way it does. Complements git history: commits capture *what changed*, this file captures *why I chose it and what I'd have done differently*.

Roadmap: keyword search → semantic search → chunking → hybrid search → LLMs → reranking.

Entries are newest first. Each entry answers four questions:

1. **What was wrong / what did I want?** — the trigger, not the change.
2. **What did I consider?** — alternatives, even one-line dismissals.
3. **What did I choose and why?** — decision + the reason in one sentence.
4. **What did I learn?** — filled in *after* the code is running. This is where the value lives.

---

## 2026-07-28 — "Aspirational dead code": how a helpful refactor hid a two-session bug

### What was wrong / what I wanted

The `bm25search "animated family"` test had been failing by exactly ±0.01 on two of three top results since 2026-07-26. The BM25 entry that day left it as an open question with a wrong-but-plausible hypothesis (the `+1` in the IDF form). It sat there for two days until a reference `InvertedIndex` implementation surfaced during a code review — same formulas, same corpus, same output *only if* the tokenization pipeline emitted the same tokens.

### What I considered

- **IDF variant.** The Lucene-style `log(... + 1)` vs classical Robertson-Sparck-Jones. Testing would have shifted all scores in the same direction — but the actual failure had one score shifting up and another shifting down. Ruled out.
- **Document length definition.** Whether `doc_lengths` counted raw tokens or post-filter tokens. Actively tried this on 2026-07-27 — scores diverged further, so the reference matched my implementation here.
- **Numerical precision at rounding boundaries.** Plausible cover story: raw scores near `X.X5` land differently under `.2f` formatting. Would explain the asymmetric shift but not the persistence across queries.
- **Tokenization pipeline.** The actual culprit. Only surfaced once a reference tokenizer was pastable side-by-side.

### What I chose and why

**Root cause:** `preprocess_text` (lowercase + strip `string.punctuation`, including the apostrophe) was applied to *documents* but not to the *stopwords list*. So a document token `"don't"` became `"dont"` on its way to filtering; the stopword list still held `"don't"`. No match → the contraction survived filtering → got stemmed → became a real index term. Multiply by 45 contractions across 5000 movies with many contraction occurrences per movie and you get:

- `doc_lengths` inflated by 5-30 per document.
- `avgdl` correspondingly shifted.
- `term_frequencies` bloated with garbage terms like `"dont"`, `"isnt"`, `"hes"`, `"weve"`.
- Every downstream BM25 score drifted by a small consistent amount, hitting different rounding-display boundaries for different queries.

**Fix:** apply `preprocess_text` to each stopword at load time, and switch the return type from list to set for O(1) lookup:

```python
def load_stopwords():
    with open(STOPWORDS_PATH, "r") as file:
        return {preprocess_text(line) for line in file.read().splitlines()}
```

Term count dropped 85577 → 85543 (exactly the contraction-stem count). All three previously-failing scores now match the reference to the digit.

### What I learned

- **"Dead code" is sometimes aspirational, not accidental.** During a scoped cleanup on 2026-07-26, I deleted this block from `filter_stopwords`:
  ```python
  p_stopwords = []
  for word in stopwords:
      p_stopwords.append(preprocess_text(word))
  ```
  because `p_stopwords` was built and never used — the `if word not in stopwords` check right below it referenced the raw list. That looked exactly like the fingerprint of a for-loop that got written and forgotten. It wasn't. The original author had *intended* to preprocess stopwords, wrote the transformation, then wired the wrong variable into the check. What I removed as "unused" was a broken but recoverable signal of intent. The two-line comparison was:
  ```
  if word not in stopwords:       # what the code did (wrong)
  if word not in p_stopwords:     # what the author meant (correct)
  ```
  Only one variable name apart. Impossible to see without context.
- **Rule to internalize.** When about to delete a local that's assigned-but-unread, check whether it was *supposed to feed* the nearby comparison. "Dead code" and "wrong variable name" look identical at a glance; the difference is what was *just above* the mistake, not what's missing.
- **The ±0.01 signature was more precise than the hypothesis.** Asymmetric shifts (one score up, one down) told me on day one that the bug wasn't a global formula difference. I had that observation in the log and still spent two days investigating global-formula hypotheses because it was easier to reason about. When your evidence points to "local integer noise," look at the tokenizer before touching the math.
- **Preprocessing symmetry is a general invariant.** Whenever you apply a transform on one side of a comparison, apply it on the other. This isn't specific to stopwords — the same class of bug lurks anywhere text meets text: query vs. index terms, doc vs. metadata, chunk boundaries vs. delimiters. Named lookups (`in`) against externally-authored files (stopword lists, category names, tag vocabularies) are the highest-risk spots because the file's author can't see your preprocessing.

### What this unlocks

- The ±0.01 that was pending an "open question" tag in the BM25 entry is resolved — I've backfilled that entry with a resolution note pointing here.
- Semantic search (phase 2) won't hit the same class of bug — the embedding pipeline doesn't strip apostrophes; the model tokenizer handles contractions natively. Which is one more concrete reason for the "LM does its own preprocessing" observation in the phase-2 log entry: two different preprocessing pipelines are a feature, not duplication to eliminate.

---

## 2026-07-28 — Word chunking → sentence chunking for semantic retrieval

### What was wrong / what I wanted

The first `chunk()` implementation split by word count — pick `chunk_size = 200` words, slice, done. That's fine when chunks feed a keyword index (a stemmed token is a stemmed token regardless of where the split lands), but embeddings expect *semantic units*. Splitting mid-sentence — often mid-clause — hands the encoder half a thought and asks it to represent the whole one. The resulting vector is a garbled average of "the boy walked into the" and "forest and saw a tiger."

### What I considered

- **Fixed word count** (what `chunk()` already does) — simple, but every semantic boundary is coincidental.
- **Regex sentence split** — `re.split(r"(?<=[.!?])\s+", text)`, then group into N-sentence windows. Cheap, no dependencies, correct for well-punctuated prose.
- **Proper sentence tokenizer** (NLTK `sent_tokenize`, spaCy) — handles abbreviations (`Mr. Smith` doesn't split), ellipses, dialogue quotes. Correct where the regex fumbles, but drags in NLTK punkt or a full spaCy model just to move a boundary.
- **Recursive splitter** (LangChain-style — try paragraphs, then sentences, then clauses, then words) — adaptive but overkill for movie descriptions that are already short.

### What I chose and why

**Regex sentence split, then group into windows of `max_chunk_size = 4` sentences with `overlap = 1`.**

- Movie descriptions are 5–20 sentences; the regex handles this vocabulary well enough that upgrading to a real tokenizer would be premature.
- 4 sentences is roughly one paragraph — a natural unit of local meaning without exceeding the encoder's ~256-token window.
- 1 sentence of overlap preserves cross-boundary context: if the answer to a query hinges on "she was his sister," that sentence appears in both surrounding chunks.
- Deferred the proper tokenizer until an actual query fails because of an abbreviation split. Might never happen.

### What I learned

- **The bug lives in the loop bound, not the loop shape.** My first version was `for i in range(0, len(sentences), step):` — which happily emitted a "tail" chunk containing only the last sentence, already present in the previous chunk. On the "animated family"–style corpus (5000 movies, avg description ~15 sentences), this added **1635 redundant chunk embeddings** on top of the correct 72909. Only caught it because a test asserted the exact total; without that assertion, 2.2% garbage would have shipped invisibly. Fix: `range(0, max(1, len(sentences) - overlap), step)` — cap the start-index at "the point past which any new chunk is entirely covered by the previous chunk's overlap."
- **`while` vs `for` was a red herring.** I initially suspected the shape of the loop was the problem. It wasn't — the same off-by-one exists in either form. The fix is dimensional (change the bound), not structural (change the control flow). Worth remembering: when a loop feels wrong, check the bound before rewriting the shape.
- **Chunking is exactly where cache-poisoning bugs live.** A wrong count doesn't raise. The `.npy` writes successfully with 74544 vectors instead of 72909; every downstream search runs against a subtly-corrupted index; nothing complains. The engineering-log entry from yesterday flagged "cache invalidation is where the bugs live" — this is the same category of failure, one phase earlier than expected.
- **Sentence-regex limitations I'm accepting for now:** abbreviations (`Dr.`, `Mr.`, `U.S.`), decimal numbers (`3.5`), and ellipses will all be split. Movie descriptions rarely feature any of these in ways that hurt retrieval — but if a phase-4 hybrid-search evaluation shows semantic recall dropping on titles with medical/legal jargon, that's the culprit.

### What this unlocks

- Chunk-level embeddings replace whole-description embeddings for retrieval — meaning a query can match on a specific plot point buried three sentences into a paragraph, not just movies whose overall description skews toward the query.
- `semantic_chunk` is now a boundary function that phase 4 (hybrid search) will also feed BM25 with — same chunking, two indexes. Consistent chunk boundaries across both retrievers is a prerequisite for any RRF-style fusion.

---

## 2026-07-27 — Embeddings are expensive: adopting a load-or-create cache pattern *(phase 2, in progress)*

### What was wrong / what I wanted

The moment I moved from keyword search to embeddings, the per-command cost profile inverted:

- **Keyword-phase indexing** (inverted index over 5000 movies) built in seconds, in-process, no external dependency. Rebuilding on every CLI invocation was fine.
- **Embedding-phase indexing** (`SentenceTransformer.encode` over 5000 movie descriptions) takes ~30s locally, would be several dollars if this were an OpenAI-embedding project, and pulls model weights over the network. Rebuilding on every `search`/`verify_embeddings`/`embed_chunks` invocation is not viable — even for a learning project it'd make iteration painful.

### What I considered

- **Rebuild every run.** What the BM25 CLI did. Cheap for inverted indices, unusable for embeddings.
- **In-memory cache with lru_cache or similar.** Doesn't survive process exit. CLI processes are one-shot, so this is worthless.
- **`load_or_create_embeddings` pattern** — check for a cached `.npy` on disk, load if present and the doc count still matches, otherwise call `build_embeddings` and save.
- **File format choice: `.npy` vs `.pkl` vs `.parquet` vs plain JSON.** `.pkl` works but drags in pickle's security surface for a data blob. `.parquet` needs pyarrow and its columnar layout is wasted on a single 2D dense matrix. JSON turns each float into 20-character decimal strings — memory dies. `.npy` wins: numpy handles it with no extra deps, `np.load` can mmap for zero-copy reads, and it round-trips exact dtype/shape.

### What I chose and why

**Disk-backed `load_or_create_embeddings` + `.npy` format.** The trigger is: embeddings are *deterministic* given `(model, input)` and *expensive* to recompute, which is the exact profile that makes caching worth the complexity. Two files: `movie_embeddings.npy` (the vectors) and later `chunk_embeddings.npy` + `chunk_metadata.json` for the chunked variant. `.npy` is the least-effort format that doesn't leave real performance on the table.

Validity check is currently `len(cached) == len(documents)` — cheap sentinel, good enough for a fixed corpus. A real system would hash the model name + input to key the cache; deferred.

### What I learned

- **Keyword search never forced this discipline.** BM25 indexing is so cheap the whole "does this cache still match reality?" question didn't come up. Embeddings introduce it on day one — you can't iterate on `search` without a load path.
- **The API/local distinction is a red herring for the pattern.** Even fully local (`sentence-transformers` on CPU), 30s startup makes rebuild-every-time unbearable. The pattern is about *cost per invocation*, not who's charging for it. If I ever swap the local model for an OpenAI-embeddings call, the same cache file works unchanged — only the model name in the key would need to move.
- **Cache invalidation is where the bugs live.** The current `len(cached) == len(documents)` check misses: model version changes, chunker-parameter changes, description edits that keep the same count. Every one of those will silently return stale results. Noting for later — probably move to a hash-keyed cache directory once phase 3 (chunking) parameters start proliferating.
- **`.npy` has failure modes worth naming up front.** Concrete ones I've hit or will hit:
  - **No content versioning.** If I swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` (both ~384-dim families), the shape stays valid but the vectors mean different things. The freshness check passes; searches silently return garbage.
  - **No incremental append.** Adding one movie means re-encoding all 5000. Fine for a fixed corpus; wrong tool the day the corpus goes live.
  - **Not human-inspectable.** Can't `less` or `jq` a `.npy` file — every debug requires round-tripping through Python.
  - **Torn writes.** Hit this concretely: `chunk_embeddings.npy` wrote successfully, then the `chunk_metadata.json` write blew up (`"wb"` vs `json.dump`), leaving the cache in a half-state that the freshness check couldn't detect — it only checks presence, not consistency. Standard fix pattern is write-to-`.tmp` then `os.replace()` when both files succeed. Not applied yet; noting so I don't lose a second five-minute encode learning the same lesson.

### What this unlocks

- Iteration on `search` doesn't pay the embedding cost each time — the loop is edit → run → observe → edit, not edit → wait 30s → run → wait 30s → observe.
- Same pattern is already reused in `ChunkedSemanticSearch.load_or_create_chunk_embeddings`. Confirms it generalizes; probably promotes to a utility once a third caller appears (rule of three).

---

## 2026-07-27 — BM25 → semantic search: closing the vocabulary gap

### What was wrong / what I wanted

BM25 is only as good as the literal token overlap between query and document. That's a hard ceiling — no amount of `k1`/`b` tuning helps when the words themselves don't match. The failure cases that made this concrete for the movie corpus:

- `"jungle boy"` finds nothing about **Mowgli** or **Tarzan** — those descriptions don't use the query's words.
- `"reincarnation"` misses movies that say "born again" or "a second life."
- `"family friendly"` misses descriptions written as "suitable for all ages."
- `"grief"` misses "coping with loss."

BM25 sees these as unrelated queries. A human ranking movies for the same query would obviously prioritize the semantically-matching ones — but the retriever can't, because it's operating on strings, not meanings.

### What I considered

- **Query expansion with WordNet/synonym lists** — cheap on top of BM25, but brittle (WordNet doesn't know "jungle boy" ≈ "Mowgli") and turns every query into a scoring shotgun.
- **SPLADE / learned sparse retrieval** — best-of-both-worlds; keeps the inverted-index infra, adds semantic term weights via a transformer. Right answer for production; too heavy for a learning project's phase 2.
- **Dense retrieval end-to-end** — encode every document into a fixed vector, encode the query the same way, rank by cosine similarity. The canonical "semantic search" approach.

### What I chose and why

**Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2` and cosine similarity.**

- 384-dim vectors: small enough that 5000 movies fit in ~7 MB, big enough to preserve useful structure.
- Pretrained on ~1B general-purpose sentence pairs — no domain fine-tuning needed for a movie corpus.
- Runs on CPU at ~7 batches/sec. Painful once (embed the corpus) but reusable everywhere.
- Model outputs are already L2-normalized (visible in `verify_model` output: `Normalize({})` layer). Which means **cosine similarity = dot product** — one matmul over the whole corpus is faster than iterating and dividing by norms.

### What I learned

- **Semantic retrieval doesn't kill keyword retrieval.** The two are complementary — BM25 crushes exact-phrase and rare-keyword queries where the query and doc use the same jargon; embeddings win where they don't. This is exactly why phase 4 is *hybrid* search, not "replace BM25."
- **The normalized-model shortcut is worth exploiting.** Because `all-MiniLM-L6-v2` outputs unit vectors, `similarities = self.embeddings @ query_embedding` (one numpy matmul, shape `(5000,) @ (384,) = (5000,)`) is the correct implementation. Any Python-level cosine loop with per-vector `norm()` calls is doing 15,000 redundant sqrt operations — a slower and less faithful version of the same math.
- **Vector search stayed linear on purpose.** A brute-force scan over 5000 vectors runs in µs. ANN structures (FAISS, hnswlib) pay off around ~1M docs, not thousands. Reaching for FAISS at this scale is premature and adds an approximation-quality dial I don't need.
- **Same shape as BM25 output.** Both `bm25_search` and `semantic search` return `list[(doc_id, title, score)]`. That symmetry makes phase 4 fusion (RRF, weighted sum) a data problem, not a shape problem.
- **The embedding model does its own preprocessing — mine would be counterproductive.** All the tokenise → lowercase → stopword-filter → Porter-stem pipeline I hand-built for BM25 is *the wrong thing* to feed a sentence-transformer. The model's tokenizer expects raw text: it handles casing, subword splitting, and punctuation as part of what it learned. Feeding it `"cinderella lose slipper prince"` (BM25-preprocessed) throws away the exact signals — word order, function words like "her" and "to," morphology — that give the encoder something to work with. Concretely: `all-MiniLM-L6-v2` embeds `"Cinderella loses her slipper at the prince's ball."` closer to `"a princess drops her shoe running from the palace"` than to `"cinderella lose slipper prince"`. Two different indexing pipelines for two different retrievers is a feature of hybrid search, not duplication to eliminate.

### What this unlocks

- Phase 3 (chunking) has a real motivation: semantic search works better on paragraph-sized units than on whole descriptions, because the vector for a 200-word doc smears many topics into one embedding.
- Phase 4 (hybrid) can be a fair comparison — the same query flows into two retrievers with the same input/output shape, and RRF or a weighted combine works out of the box.

---

## 2026-07-26 — Naive TF-IDF → BM25

### What was wrong / what I wanted

Naive TF-IDF has three well-known weaknesses that were going to bite once the corpus (5000 movies, descriptions ranging from a paragraph to several) hit real queries:

- **TF is unbounded.** A description that repeats "dog" 20 times scores 4× one that says it 5 times, even though for a movie the extra repetition doesn't imply 4× the relevance.
- **Long documents accumulate score by mass.** More words → more chances to match → higher raw TF-IDF, regardless of whether the doc is actually *more* about the query.
- **No principled length normalization.** TF-IDF has no "b" knob to say "penalize long docs, but not linearly."

### What I considered

- **TF-IDF + cosine similarity** — L2-normalizes for length but doesn't cap TF saturation. Half a fix.
- **BM25 (Okapi)** — solves both, two tunable knobs (`k1` for saturation, `b` for length norm), same inverted-index infrastructure I already had.
- **BM25F** — per-field weighting (title × 3, description × 1). Genuinely tempting for movies where title is a stronger signal than description, but adds indexing complexity.
- **Language models (Query Likelihood + Dirichlet)** — competitive with BM25 in the IR literature, but very rarely used outside academic settings; I'd learn less about *why* the numbers move.

### What I chose and why

**BM25 with defaults `k1 = 1.5`, `b = 0.75`.**

- Reused the existing inverted index — only the scoring function changed.
- `k1` caps TF's contribution at a plateau instead of letting it grow linearly. Concretely, the second occurrence of a term matters much more than the tenth.
- `b` scales length normalization: a doc that's 3× the average has its TF component shrunk, but not to zero.
- IDF form: `log((N - df + 0.5) / (df + 0.5) + 1)` (Lucene / BM25+ variant). Picked over classical Robertson-Sparck-Jones because it can't go negative for very common terms, which was a real risk with a small corpus.
- BM25F deferred. Right call to make once I know whether title-hits dominate; premature otherwise.

### What I learned

- **Getting the formula right was easy; matching another implementation's output to 2 decimal places was not.** A test asserted specific scores for query `"animated family"` — top result matched, next two were off by exactly ±0.01 in opposite directions. That asymmetry ruled out a global formula error and pointed at numerical noise landing on `.005` rounding boundaries.
- **First hypothesis (doc length = raw token count) was wrong.** Rebuilding the index with pre-stopword-removal lengths made scores diverge *further* from the reference. Confirmed the reference counts post-filter lengths, same as I do.
- **Practical takeaway:** for a hand-rolled search engine, exact score matching against an external ranker is not a stable target. Top-K ordering is. If a test needs exact numeric scores, it's testing the implementation, not the retrieval quality.
- **Open question left for later:** whether swapping the IDF back to classical (`log((N-df+0.5)/(df+0.5))`, no `+1`) would close the ±0.01. Worth trying only if a downstream phase actually cares.
- **RESOLVED 2026-07-28** — the ±0.01 was never the IDF variant; it was a preprocessing asymmetry in the stopword filter. `preprocess_text` was applied to document tokens (stripping apostrophes, so `"don't"` → `"dont"`) but *not* to the stopword list loaded from `data/stopwords.txt`. All 45 English contractions in that file (`don't`, `it's`, `he's`, `won't`, `i've`, …) therefore never matched — every occurrence in the corpus leaked through as a real token, inflating `doc_lengths`, `avgdl`, and `term_frequencies` by a small consistent amount. Fixing `load_stopwords` to return `{preprocess_text(line) for line in file}` closed the gap completely: Gakuen Alice / Day of the Animals / Fantastic Mr. Fox scores now match the reference to exactly 7.35 / 7.13 / 6.92, and the index shrank from 85577 to 85543 terms — exactly the contraction-stem count. See the 2026-07-28 "Aspirational dead code" entry above for the deeper lesson.

### What this unlocks

- The `bm25search` CLI command now returns ranked `(doc_id, title, score)` tuples, which is the shape phase 2 (semantic search) can be compared against — same input, different scorer.
- `k1` and `b` are exposed as constants and CLI flags, so parameter sweeps are one-liner scripts once I have relevance judgments.
