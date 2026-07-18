# BENCHMARKS_REAL.md — the first honest numbers

_2026-07-16. Real models, a real public benchmark, third-party relevance
judgments. **These supersede every accuracy number in `BENCHMARKS.md`,
`COMPARISON.md`, `PARITY.md` and `README.md`**, all of which were produced by the
deterministic hash embedder scoring synthetic vectors against synthetic vectors
— a hash function grading itself._

## The harness is calibrated

The single most important number here is not RRO's:

| | nDCG@10 |
|---|---|
| our BM25 on nfcorpus, **stemmed** | **0.3283** |
| published BEIR BM25 on nfcorpus | **~0.325** |
| our BM25, unstemmed (`Analyzer::default()`) | 0.3115 |

Our lexical floor reproduces the literature's lexical floor **to within 1%**.
That is what makes everything below arguable rather than self-reported. A harness
whose baseline doesn't match published work is measuring something else, and its
headline number is worthless no matter how good it looks.

**This was wrong twice, both times in the flattering direction.**

1. The BM25 arm originally faked "lexical only" by handing `hybrid_search` a zero
   vector — fusion still ran and blended a degenerate dense ranking into the
   lexical one, scoring **0.159**, less than half the real baseline. A baseline
   broken *low* makes every arm above it look good. Fixed to call the estate's
   real `lexical_search`.
2. It then ran **unstemmed** (0.3115), because `Estate::open()` →
   `EstateConfig::default()` → `Analyzer::default()` is the *legacy* pipeline and
   nobody had read it. Stemming moved it to **0.3283** — from 4% under the
   published figure to within 1% of it. The analyzer is now an explicit knob
   (`RRO_EVAL_ANALYZER=legacy|stemming`, default `stemming`) rather than an
   inherited default.

The pattern in both: **the harness was quietly miscalibrated low, which makes
every arm above it look better than it is.** That is the most dangerous direction
for a bug to point, and therefore the one to check first.

## nfcorpus (BEIR) — 3,633 docs, 323 judged queries, graded qrels 0..2

Embedder: Qwen3-Embedding-4B (f16, llama.cpp `:8090`, 2560-d).
Reranker: llama-nemotron-rerank-1b-v2 (vLLM `:8092`). `recall_k=100`, `top_k=10`.

| arm | nDCG@10 | Recall@10 | MRR@10 | wall ms/query | vs prev |
|---|---:|---:|---:|---:|---|
| `bm25` — lexical only | 0.3115 | 0.1519 | 0.5188 | 42.2 | — |
| `dense` — ANN only | **0.4124** | 0.2013 | 0.6252 | 43.6 | +32.4% |
| `hybrid` — dense+BM25, RRF-fused | 0.3902 | 0.1903 | 0.6132 | 46.3 | **−5.4%** |
| `hybrid+rerank` — + cross-encoder | **0.4288** | 0.2152 | 0.6264 | 1148.7 | +9.9% |
| `rro` — the full pass via `ask()` | **0.4288** | 0.2152 | 0.6264 | 1128.0 | **+0.0%** |

Reproducible: `bm25` and `hybrid` land on **identical** figures across two
independent runs; `dense` moved 0.4119→0.4124 (ANN tie-breaking).

## Latency: engine vs model — they are not the same thing

The engine emits a per-stage breakdown (`flow.stage` events). Averaged over the
323 `rro` passes:

| stage | ms | what it is |
|---|---:|---|
| `rrd` | **0.006** | the gate ladder — pre-model, as claimed |
| `embed` | 42.979 | **model** (HTTP + a 4B forward) |
| `recall` | **3.724** | **the engine** (hybrid ANN+BM25 over 3,633 docs) |
| `rerank` | 1081.122 | **model** (100 cross-encoder pairs) |
| `classify` | **0.192** | readiness verdict |

**Engine ≈ 3.92 ms. Model ≈ 1124 ms. 99.65% of the wall clock is the model.**

This corrects a reporting error, not a regression. An earlier version of this
document printed `1167 ms` under a column called `ms/query` as though it were
engine latency — it was an HTTP round-trip plus a 4B forward plus 100
cross-encoder pairs, summed. The README's **1.88 ms p50** claim is *consistent
with `recall`*, which measures 1.659 ms at 300 docs and 3.724 ms at 3,633.

Anything labelled `wall ms/query` above is end-to-end and model-dominated. It is
not a statement about the engine.

### Finding 1 — hybrid fusion HURTS here, and it survived two rescue attempts

`hybrid` (0.3902) is **worse than `dense` alone** (0.4124). Two hypotheses were
raised that this was a bug in the harness rather than a property of the result.
**Both were tested. Both failed.** The finding stands.

**Hypothesis 1 — the lexical arm was crippled by an unstemmed analyzer.**
`Analyzer::default()` is the *legacy* pipeline: word tokens, lowercased,
stopword-filtered, **unstemmed**. `rro-eval` used `Estate::open()` →
`EstateConfig::default()`, so BM25 ran unstemmed on morphology-heavy biomedical
text (statin/statins, cancer/cancers). Real bug; real fix. Re-run with
`RRO_EVAL_ANALYZER=stemming`:

| arm | unstemmed | stemmed |
|---|---:|---:|
| `bm25` | 0.3115 | **0.3283** (+5.4%) |
| `dense` | 0.4124 | 0.4120 (unchanged, as expected) |
| `hybrid` | 0.3902 (−5.4%) | **0.3943 (−4.3%)** |

Stemming lifted BM25 by 5.4% — **and the regression survived.** A better lexical
arm narrowed the gap and did not close it.

**Hypothesis 2 — RRF had no weight, so a weak arm got an equal vote.**
Also a real bug: `reciprocal_rank_fusion` summed `1/(k+rank)` per list with no
per-list scale at all. The mechanism is arithmetic — a BM25-only hit at rank 1
scores `1/61 = 0.01639` and beats a dense hit at rank 2 at `1/62 = 0.01613`, so
the *worse* retriever outvotes the better one on its own turf. That is pinned by
a unit test (`unweighted_fusion_lets_a_weak_list_outvote_a_strong_one`).

A weight was added (`HybridWeights`) and swept — one embed pass, every weight
scored against it (fusion is query-time, so re-weighting is a clone, not a
reindex):

| dense:lexical | nDCG@10 |
|---|---:|
| 1:1 (plain RRF) | 0.3943 |
| 1.5:1 | 0.3980 |
| 2:1 | 0.4049 |
| 3:1 | 0.4083 |
| 5:1 | 0.4102 |
| 8:1 | 0.4114 |
| **dense alone** | **0.4120** |

**The curve converges toward dense from below and never crosses it.** That is the
asymptote, not a plateau: as the dense weight grows, fusion → dense-only. **The
optimal lexical weight on nfcorpus is ≈ 0.** Weighting doesn't rescue fusion; it
just lets you dial the damage down to zero by turning BM25 off.

**Conclusion.** On this corpus, with this embedder, BM25 contributes nothing to
dense and any vote given to it makes results worse. Both "missed optimization"
theories were real bugs worth fixing on their own merits — and neither was the
cause.

**What this is *not* evidence of.** It is one corpus. BM25 earns its keep on
**rare exact tokens** — identifiers, SKUs, error codes, acronyms — and nfcorpus
is natural-language biomedical prose with none of that. BRIGHT (the real target)
is reasoning-heavy and lexically richer.

**The actual fix is not a constant.** A single global weight is the wrong
abstraction: the right weight is a property of *the query*, not of the estate.
That is what RRD's centroid router is for — classify the shape, pick the fusion
strategy per query. Note also that any weight chosen by reading the table above
would be **fit to these 323 evaluation queries** — that is tuning on the test
set, and it is barred. A weight ships only with a train/dev/test split.

### Finding 1c — the ladder now carries confidence intervals and p-values

Every number above was a point estimate. "Dense beats BM25", "fusion hurts",
"rerank helps" were stated from means with no measure of whether 323 queries could
produce them by chance. `rro-eval` now computes, on the same run (the scores are
**paired** — every arm scores the same queries in the same order):

- a **percentile bootstrap** 95% CI on each arm's mean nDCG@10 (10,000 resamples), and
- a **paired sign-flip permutation test** on each ladder step's per-query delta —
  the textbook distribution-free test for paired data.

Both are seeded (`0xC0FFEE`), so the numbers reproduce exactly. Stemmed run,
Qwen3-4B embedder, nemotron reranker:

| arm | nDCG@10 | 95% CI |
|---|---:|---|
| `bm25` | 0.3283 | [0.2942, 0.3633] |
| `dense` | 0.4120 | [0.3756, 0.4490] |
| `hybrid` | 0.3943 | [0.3587, 0.4307] |
| `hybrid+rerank` | 0.4293 | [0.3938, 0.4656] |
| `rro` | 0.4293 | [0.3938, 0.4656] |

| ladder step | Δ nDCG@10 | 95% CI | p | verdict |
|---|---:|---|---:|---|
| `dense` vs `bm25` | **+0.0837** | [+0.063, +0.105] | 0.0001 | **significant** — dense clearly wins |
| `hybrid` vs `dense` | **−0.0177** | [−0.031, −0.004] | 0.0104 | **significant** — fusion genuinely *hurts* (CI excludes 0) |
| `hybrid+rerank` vs `hybrid` | **+0.0350** | [+0.020, +0.051] | 0.0001 | **significant** — the reranker earns its 1.1 s |
| `rro` vs `hybrid+rerank` | +0.0000 | [0, 0] | 1.0000 | n.s. — identical pipeline, as expected |

This upgrades Finding 1 from directional to **statistically significant**: the
"fusion earns nothing" result is not resampling noise — the mean-delta CI is
entirely below zero at p=0.01. And it holds the harness honest in the other
direction: `rro` vs `hybrid+rerank` comes back exactly n.s. (they are the same
pipeline), which is the null result a working significance test must produce.

Still one corpus. The cross-dataset check (a second BEIR/BRIGHT set) is the
remaining Phase-15 work — it needs a second corpus embedded, not just re-scored.

### Finding 1b — the `rro` arm's DEFAULT reranker destroys the result

In the stemmed run the flagship `rro` arm scored **0.3199 — the worst arm
measured**, below `bm25` (0.3283) and far below `hybrid` (0.3943). That is not a
statement about the engine. It is a default:

```rust
match reranker {
    Some(r) => flow.reranker(r).build(),
    None    => flow.build(),   // -> LexicalReranker (BM25)
}
```

`rro-eval` prints `reranker: none` when `RRO_RERANKER` is unset — but `ask()`
still runs the flow's reranker, and `ObjectBuilder::build()` defaults it to
**`LexicalReranker`**. So the "no reranker" arm silently **BM25-re-sorts the
fused candidates**, dragging 0.3943 down to ≈ BM25's own 0.3199. The harness
reported the absence of a reranker while a reranker was actively undoing the
dense ranking.

This is the same failure shape as Findings 1's two hypotheses and the zero-vector
BM25 bug above: **a default nobody chose, silently degrading a measurement, and
looking like a result about the architecture.** Three of them in one harness.
Every default on the measurement path is now suspect until it has been read.

### Finding 2 — the reranker earns quality and costs ~26x

`hybrid+rerank` (0.4288) beats `hybrid` by 9.9% and `dense` by 4.0%, and is the
best arm. It also takes the wall clock from ~44 ms to ~1149 ms — **1081 ms of
that is the cross-encoder**, not the engine. Whether that trade is worth paying
is a product decision, and it needs the number in front of it.

It also *rescues* fusion: the reranker recovers hybrid's regression and passes
dense. The cross-encoder is doing the work the fusion weighting isn't.

### Finding 3 — RRD and the classifier add ZERO ranking lift

`rro` and `hybrid+rerank` are **identical to four decimals** (0.4288 / 0.2152 /
0.6264) and return the same worst queries.

This is expected, and worth saying out loud anyway: RRD **gates** (cost control —
it refuses payloads before any model runs) and the classifier **judges** (a
readiness verdict the reasoner can act on). Neither reorders results. Anyone
reading "the full RRO pipeline" as "retrieves better" is wrong — the full pass
buys refusal, a trust signal, and the connectome map, at `rrd=0.006ms` +
`classify=0.192ms`. It does not buy ranking.

The equality is also a useful consistency check: it proves `ask()` and the
hand-built ladder agree, i.e. the engine's own composition has no hidden
divergence from the arms measured beside it.

### Finding 5 — the ANN default holds on real vectors (1,200-doc caveat)

`recall/src/ann.rs:533` gates recall@10 ≥ 0.95 against exact search, and had only
ever run on synthetic uniform noise. Swept on **real 2560-d Qwen3 embeddings**
(`crates/recall/tests/real_vector_ef.rs`, fed by `RRO_EVAL_EXPORT_VECTORS`):

| ef | recall@10 | µs/query |
|---:|---:|---:|
| 4 | 0.9980 | 196.3 |
| 8 | 0.9980 | 197.0 |
| 16 | 0.9990 | 253.7 |
| 32 | **1.0000** | 383.1 |
| **64** (default) | **1.0000** | **585.5** |
| 256 | 1.0000 | 1094.9 |

**The default `ef_search=64` holds on real data** — that was the open question.
`ef=32` reaches the same recall 35% faster, so there is visible headroom.

**Read this as weak evidence, not a tuning.** 1,200 vectors is small for an ANN
gate: the graph is nearly fully connected, search degenerates toward exhaustive,
and recall is flattered for reasons unrelated to the beam. Passing at `ef=4` is a
statement about the corpus, not the index. The test prints this caveat itself and
it stands until ≥50k real vectors are swept.

Three bugs in the harness were found on the way here, each by distrusting a
green: the timer wrapped the brute-force oracle (so µs/query measured the oracle,
not the beam, and sat flat at ~4.2ms); `search(q,k,ef)` **clamps**
`ef.max(config.ef_search)` (ann.rs:316), so every ef below 64 silently searched
at 64 — the tell was ef=4..64 taking an identical 584µs while ef≥100 responded;
and the first conclusion therefore claimed "passes at ef=4" when it had passed at
64 nine times. The beam is now swept on the CONFIG, not the argument.

### Finding 5b — the ef knee at 50k scale (the honest scale gate)

Finding 5 promised the small-corpus caveat "stands until ≥50k real vectors are
swept." This is that sweep at scale: **50,000 vectors, dim 64**, in 5,000
well-separated 10-point clusters — synthetic, but with genuine neighbourhood
structure (each query is a held-out point inside a blob, so its true top-10 are
that blob's members, unambiguously). The graph is *not* nearly-fully-connected at
50k, so this is the beam doing real work, not exhaustive search in disguise. Run:

```
cargo test -p recall --release ef_search_sweep_50k -- --ignored --nocapture
```

| ef | recall@10 | p50 (µs) | p95 (µs) |
|---:|---:|---:|---:|
| 10 | 0.9400 | 16 | 24 |
| 16 | 0.9750 | 17 | 27 |
| 24 | 0.9900 | 19 | 28 |
| **32** *(knee)* | **1.0000** | 20 | 28 |
| 48 | 1.0000 | 30 | 43 |
| **64** *(default)* | **1.0000** | **37** | 51 |
| 128 | 1.0000 | 77 | 92 |
| 256 | 1.0000 | 157 | 169 |

**The knee is ef ≈ 32; the shipped default `ef_search=64` clears it with 2×
headroom** at ~37 µs p50, and everything past 32 buys latency for zero recall.
This is now a gate assertion, not a note: the sweep fails if the knee ever exceeds
the default (`knee <= AnnConfig::default().ef_search`). The one thing this run is
NOT is real embeddings at 50k — that needs 50k real vectors (nfcorpus alone is
3.6k) and belongs to Phase 15; structured-synthetic is the honest stand-in for
"does the knee hold at scale," and it does.

### Finding 4 — ingest is ~1000x slower than advertised

**10 docs/sec** (3,633 docs in 355 s), against the README's **10.9k docs/sec**.

Nothing regressed. The old number was measured with a microsecond hash embedder;
this one runs a real 4B model over HTTP. Once a real model is in the path, the
forward pass dominates and every wire/engine choice becomes noise. The README's
figure should be read as "how fast the estate can index vectors someone else
already computed."

## What is NOT yet measured

- **BRIGHT** — the reasoning-intensive benchmark (published SOTA is only ~22.1
  nDCG@10). nfcorpus is a warm-up; BRIGHT is the real target.
- **The 0.6/4/8B tier ladder** across candle · llama.cpp · vLLM.
- **Statistical significance.** 323 queries, single run, no CIs. The −5.4% and
  +9.9% deltas are directional, not established.

## Reproduce

```sh
hf download mteb/nfcorpus --repo-type dataset --local-dir eval-data/nfcorpus

RRO_EMBEDDER=llamacpp RRO_EMBEDDER_ENDPOINT=http://127.0.0.1:8090/v1/embeddings \
RRO_RERANKER=vllm    RRO_RERANKER_ENDPOINT=http://127.0.0.1:8092/rerank \
RRO_EVAL_DATA=eval-data/nfcorpus RRO_EMBED_BATCH=64 \
RRO_EVAL_EXPORT_VECTORS=/tmp/real-vectors.jsonl \
  cargo run --release --bin rro-eval
```

---

## Finding 4 — every weightless default is lexical, and they punish the dense half

_2026-07-17. Found by pointing the daemon at a real estate with a real embedder
(Qwen3-Embedding-4B via llama.cpp) and asking it one question — the first dogfood
of the recall spine. It failed immediately, three different ways, all the same
way._

The query was **"why does combining two retrievers make results worse"** against
a document reading *"RRO fusion finding: on nfcorpus the optimal lexical weight
is approximately zero…"*. Note there is almost no lexical overlap: no shared
content word carries the meaning. Retrieving it is exactly what dense embeddings
are *for*.

| default | what it did to that result |
|---|---|
| `LexicalReranker` (the reranker default) | re-scored by BM25: the correct document got **0.0000** and sank to rank 3, beaten by an unrelated doc about a2a protocols that happened to share a word |
| `HeuristicClassifier` (the classifier default) | verdict **`insufficient @ 0.00`** — it measures query-term *coverage*, so a semantically perfect answer with no shared tokens scores zero |
| RRF (the fusion) | scores are `1/(60+rank)` — **rank-based, magnitude-free**. A hit at rank 1 scores 0.0164 whether it is perfect or garbage |

These are not three bugs. They are one: **lexical and rank-based logic sitting in
judgement over semantic retrieval.** Each is defensible in the weightless demo,
where the embedder is a hash function and there is no semantics to destroy. Each
becomes actively wrong the moment a real model is wired in — which is the only
configuration anyone actually ships.

### Consequences

1. **Fixed:** `IdentityReranker` (`RRO_RERANKER=identity`) makes "keep recall's
   ordering" expressible. It was not, before: the rerank stage cannot be omitted,
   only filled, and the fallback was BM25. With it, the same query returns the
   right document at rank 1 in 59 ms.
2. **Open — no relevance gate exists.** Because RRF discards magnitude and the
   readiness verdict is lexical, **there is no way for a caller to distinguish
   "found the answer" from "found the nearest four things, none of which are
   relevant"**. ANN always returns *k* neighbours, however distant. For a memory
   product this is the core use case, and the plumbing for it is missing: `ask`
   embeds server-side but RRF-scores; `query` accepts `score_threshold` but
   requires the *client* to bring the vector. There is no server-side-embed +
   score-thresholded recall.
3. This is the same root cause as Finding 1. RRF throwing away score magnitude is
   why fusion cannot be weighted usefully **and** why relevance cannot be gated.
   It is why Qdrant ships DBSF alongside RRF, and it is the argument for the
   per-query fusion strategy work rather than another constant.

### Not fixed here, deliberately

`LexicalReranker` remains the default. Over a *dense-only* store it **adds**
lexical signal, which is the one case it earns its place. Changing a default that
every existing caller inherits is a measured decision, not a drive-by — it needs
the train/dev/test split, not this page.

---

## Finding 5 — the three engines, PROVEN (the gates CI never ran)

_2026-07-17, GB10. 28 tests were `#[ignore]`-gated on weights and live servers,
and `ci.yml` has never run one — no `--run-ignored` anywhere. Every claim about
candle, llama.cpp and vLLM was an assertion. These are the numbers._

### The vendored candle encoder reproduces the model card to 6 decimals

`card_reference_scores`, the strong gate (MODELS.md only asks for
`king~queen > king~banana`, which is weak — single tokens pass identically under
last-token *or* mean pooling and cannot detect the pooling bug this backend exists
to avoid):

```
got:  [[0.7645573, 0.1414254], [0.13549742, 0.5999547]]
want: [[0.7645568, 0.14142509], [0.13549736, 0.59995496]]
```

A hand-written, cache-free Qwen3 encoder landing on Qwen's published reference
matrix. Supporting gates: `batching_matches_single` = **1.000000** (left-padding
and last-token pooling are correct — this is the one that would catch a padding
bug), MRL@256 stays semantic (0.8322 vs 0.7074), paraphrase 0.7143 vs unrelated
0.2098, query/document asymmetry applied (0.7851 for the same text on both paths).

### Two independent implementations agree — on identical weights

The agreement test needed care: `:8090` serves the **4B**, and candle was running
the **0.6B**. Comparing those would have produced a meaningless number. Fetched
`Qwen3-Embedding-0.6B-f16.gguf` and served it on `:8095` with `--pooling last`, so
both engines run the *same weights at the same precision*:

| | reference matrix |
|---|---|
| published card | `[[0.7646, 0.1414], [0.1355, 0.6000]]` |
| **candle** (vendored Rust) | `[[0.7646, 0.1414], [0.1355, 0.6000]]` |
| **llama.cpp** (C++) | `[[0.7646, 0.1418], [0.1361, 0.6017]]` |

Separation margin: candle 0.4645, llama.cpp 0.4656. Batched-vs-single order
preserved at 0.999985. **Three-way cross-check: two independent implementations
and the published card all agree.** That is stronger evidence than either engine's
own gate, because two implementations of one contract can only agree by both being
right.

### The rerankers lift — and BM25 fails exactly as Finding 4 predicts

| engine | golden@1 |
|---|---|
| BM25 (`LexicalReranker`) | 0.50 |
| **llama.cpp** (nemotron-rerank-1b-v2) | **1.00** |
| **vLLM** (same model) | **1.00** |
| candle (qwen3-reranker-0.6b) | 0.50 |

llama.cpp and vLLM agree on exact ordering (`["d3","d1","d0","d2"]` and
`["d2","d0","d1","d3"]`). BM25's failure is Finding 4 in miniature: for *"How do
plants make food from sunlight?"* it ranks **"Plants need food and sunlight to grow
well in a garden"** first — pure lexical overlap — over **"Photosynthesis is the
process by which plants convert light energy into chemical energy"**.

**candle at 0.50 is the model, not the backend**, and was already diagnosed in the
test: the 0.6B *saturates* — gold 0.989082 loses to 0.989714 by 0.0006, and a
nonsense distractor still scores 0.942. Classic small-cross-encoder behaviour. Its
calibration is near-perfect on the separable case (0.9995 vs 0.000036). The tier
ladder gets decided by BRIGHT at scale, not by n=2.

### The constrained classifier beats the heuristic — at exactly Finding 4's flaw

```
heuristic: ready=true  label=ready
model:     ready=false label=insufficient conf=0.7933
  "The provided context does not contain any information about the capital of China."
=> LIFT: the heuristic was fooled by lexical overlap; the model was not.
```

The readiness third of Finding 4 already has its fix built. The heuristic says
*ready* to context that does not answer the question, because it counts shared
terms. The constrained-decode judge does not.

### The ANN gate, on real vectors for the first time

`ann.rs` gates recall@10 ≥ 0.95, and that gate had only ever run on `lcg` uniform
noise. Real embeddings are anisotropic and concentrate; noise says nothing about
them. On **2,200 real 2560-d Qwen3-4B vectors**:

| ef | recall@10 | µs/query |
|---:|---:|---:|
| 4 | 0.9680 | 305.7 |
| 16 | 0.9880 | 407.7 |
| 64 *(default)* | **0.9990** | 974.9 |
| 100 | 1.0000 | 1244.9 |

**The default `ef_search=64` holds the gate on real vectors: 0.9990.**

⚠️ **It also passes at ef=4, and that is a warning, not a win.** At 2,200 vectors
the graph is nearly fully connected and near-exhaustive search flatters recall —
passing at ef=4 is a statement about the corpus, not the index. A 50k-vector run
is the honest gate. Not treated as tuned until then.

### A bug this phase found in its own instructions

`real_vector_ef.rs` documents `RRO_EMBEDDER=llamacpp cargo run --bin rro-bench --
--export ...` to produce its real vectors. **`rro-bench` hardcoded
`DeterministicEmbedder` and ignored the variable**, writing 384-d *hash* vectors.
The test that exists precisely because "the ANN gate has only ever run on synthetic
vectors" was being fed synthetic vectors by its own setup — and would have
re-published the synthetic gate under the word "real". Caught only because 384 ≠
2560. Fixed; the export now reports `dim=2560`.
