# Reason Ready — Measured Results

**Every number here came out of a real run of `rrf-bench`.** Nothing is
asserted that a run did not produce. Reproduce with:

```sh
cargo run --release --bin rrf-bench -- --docs 50000 --queries 500 --store mem
cargo run --release --bin rrf-bench -- --docs 50000 --queries 500 --store estate
```

## Environment (2026-07-15)

Shared cloud container (Linux x86_64), release profile, default engine
components (deterministic embedder, dim 384). Synthetic corpus: 50,000 docs,
24–64 tokens each, zipf-skewed vocabulary of ~8k distinct terms; 500 hybrid
queries, top-10. **Numbers on dedicated hardware will differ — re-run there.**
External baselines run outside this tree on the same corpus/queries and are
compared on these emitted numbers.

## Ingest — the full machine (embed → index → persist)

| store | wall time | throughput | errors |
|---|---|---|---|
| `mem` (in-memory) | 0.43 s | **115,387 docs/sec** | 0 |
| `estate` (persistent kvs, durable BM25 + vectors + shapes) | 5.63 s | **8,883 docs/sec** | 0 |

Ingestion runs through the whole tokio machine: bounded intake
(backpressure), 256-doc batches, 4 concurrent batches, graceful drain, every
document embedded, BM25-indexed, and (estate) durably written.

## Query — hybrid (dense + BM25, reciprocal rank fusion), top-10

| store | p50 | p95 | p99 |
|---|---|---|---|
| `mem` | 82.3 ms | 85.6 ms | 95.1 ms |
| `estate` | 155.4 ms | 168.4 ms | 180.6 ms |

Sequential, single-client latency over 50k docs with **exact** (full-scan)
dense search. The scan is the known cost: ANN indexing (roadmap Phase 4)
replaces the O(N) scan; the trait boundary means nothing else changes.

## The rigor loop, demonstrated

The first estate run measured **762 docs/sec**. The harness exposed the flaw:
postings stored as one JSON blob per term were re-read and re-written on every
batch — O(N²) on hot terms. Re-authored to the LSM-native layout (one row per
`(term, doc)`; blind puts, prefix-scan reads):

| | before | after | change |
|---|---|---|---|
| estate ingest | 762 docs/sec | **8,883 docs/sec** | **11.7×** |

A second finding from the same runs: the in-memory store cloned every
record's payload before truncating to top-k; scoring first and cloning only
winners cut mem query p50 from 116 ms to 82 ms (−29%).

Measure → find → re-author → re-measure. That is how every performance claim
in this repository gets made.

## Bake-off vs a popular RAG store (2026-07-15, planted-v1 protocol)

**Identical inputs for every row**: the same 50,500 documents and the same
precomputed 384-d vectors (exported via `rrf-bench --export`), same shared
container, release builds, same run window. Baseline: **ChromaDB 1.5.9**
(embedded and HTTP-server modes), a widely used RAG vector store. 500 planted
queries; accuracy@10 = the planted golden doc retrieved.

| system | path | ingest (docs/sec) | accuracy@10 | query p50 |
|---|---|---|---|---|
| **rrf estate** (hybrid, durable) | local | **6,624** | **1.000** | 188.5 ms |
| **rrf estate, full pipeline** (embed→hybrid→rerank→classify per query) | **a2a layer-2 TCP** | **6,480** | **1.000** | 191.0 ms |
| rrf mem (dense-only fallback) | local | 85,358 | 0.936 | 98.1 ms |
| ChromaDB (vector ANN) | embedded | 566 | 0.572 | 3.2 ms |
| ChromaDB (vector ANN) | HTTP | 586 | 0.606 | 4.9 ms |

What the run demonstrated:

- **Ingestion: 11.7× durable-to-durable** (6,624 vs 566), and rrf's number
  *includes* server-side embedding while the baseline received precomputed
  vectors. Over the network: **11.1×** (a2a 6,480 vs HTTP 586). The
  in-memory engine is ~150× on this protocol.
- **Retrieval correctness: 1.000 vs 0.572/0.606.** The hybrid (dense + BM25,
  reciprocal-rank fused) retrieved every planted target; pure-vector ANN
  missed ~40%. rrf's own dense-only path (0.936) shows the split: exact
  scan recovers most of the gap, **hybrid closes it to zero** — the design
  thesis, measured.
- **The a2a layer-2 wire is ~free**: full pipeline remotely at 191 ms vs
  188.5 ms locally (+3 ms), identical accuracy — the "treat remote nodes as
  local" property, demonstrated over TCP.
- **Query latency is the honest loss**: the baseline's ANN answers in 3–5 ms;
  rrf's exact O(N) scan takes ~190 ms at 50k docs — while also running
  rerank + readiness per query. This is precisely P2 (ANN) — the gap is
  quantified, not hidden.

Methodology caveats, stated plainly: hash-based embeddings are adversarial
for HNSW graphs (near-orthogonal vectors), which depresses the baseline's ANN
recall relative to semantic embeddings; the historical "130×" figure was not
produced by this protocol on this container — today's measured multiples are
**11–15× durable, ~150× in-memory**, and the harness (not memory) is now the
arbiter of every future claim.

## P2: the ANN index lands (2026-07-15)

Clean-authored layered small-world graph (`recall::AnnIndex`), integrated
into the estate on the two-phase pattern (durable `vecs` CF is the source of
truth; graph applied post-commit with read-your-writes; rebuilt from durable
vectors on open). Unit gate: recall@10 ≥ 0.95 vs exact (property test).
End-to-end, planted-v1, release, this container:

| scale | metric | exact scan (before) | ANN (after) | change |
|---|---|---|---|---|
| 50k | query p50 (hybrid) | 188.5 ms | **1.40 ms** | **135×** |
| 50k | accuracy@10 | 1.000 | **1.000** | held |
| 100k | query p50 (hybrid) | ~380 ms (extrapolated O(N)) | **2.09 ms** | ~180× |
| 100k | accuracy@10 | — | **1.000** | 500/500 goldens |
| 100k | throughput | ~3 qps | **478 qps** | sequential, full hybrid |

The engine now answers **faster than the popular baseline's pure-vector ANN
(3.2–4.9 ms) while also running BM25 + reciprocal-rank fusion** — and keeps
exact-retrieval accuracy the baseline could not reach (1.000 vs 0.572–0.606).

**The ingest cost, then the fix (same day):** synchronous graph build first
dropped durable ingest 8,883 → 488 docs/sec. Moving graph apply
**out-of-band** (applier thread + pending overlay for read-your-writes +
`quiesce`, the recovered compaction pattern) plus unrolled dot kernels
restored and then beat it:

| | sync build | **out-of-band** |
|---|---|---|
| durable ingest | 488 docs/sec | **10,800–10,953 docs/sec** |
| query p50 (post-quiesce) | 1.40 / 2.09 ms | **1.06 / 1.88 ms** (50k/100k) |
| accuracy@10 | 1.000 | **1.000 @ 100k**; 0.998 @ 50k¹ |
| index catch-up (reported separately) | — | 31 s @ 50k / 71 s @ 100k |

¹ One golden in 500 lost to the fusion cutoff (a doc ranked in *both* lists
can out-fuse a lexical-only rank-1 at the top-k boundary) — a scoring-depth
tuning question, recorded rather than hidden.

Searches during catch-up stay correct via the pending overlay (exact scores
over unapplied vectors, removals masked — property- and integration-tested);
"durably ingested" and "fully indexed" are two moments and both get printed.

## P3: the map resolves the route (2026-07-15)

RELATE-style relations + BFS traversal + `scoped_search` (exact hybrid
inside a routed neighborhood). The gate corpus is *deliberately ambiguous*:
every query's golden doc has a decoy carrying the same anchor term, near-
identical text — but only the golden is RELATEd to the query's project.

| | accuracy@1 (40 ambiguous queries, 1.5k noise floor) |
|---|---|
| flat hybrid (no map) | **0.025** |
| **routed (map → treasure)** | **1.000** |

Content alone cannot tell twins apart; relationships can. This is the
fusion law measured: the map resolves the route, the treasure answers
inside it. Reproduce: `cargo test -p connxism --test routing -- --nocapture`.

## Baselines & the regression gate

Recorded container baselines live in `baselines/` (config + numbers, JSON).
`rrf-bench --baseline <path>` re-runs the same configuration and exits
non-zero on regression beyond tolerance — see
[OBSERVABILITY](OBSERVABILITY.md). Runs stream JSONL events (`--events`)
queryable directly by DuckDB.

## Sprint 9: filters go index-first, vectors go 4× smaller (2026-07-16)

**Payload secondary indexes** (`pidx` CF, one row per (field, typed value,
doc), order-preserving numeric encoding so ranges are index scans): when
every clause of a `Filter` touches an indexed field, the exact matching
id-set is resolved from sorted scans and scored exactly inside it —
filter-first, not post-filter.

| 10k docs, `team = "sec" AND priority >= 8` | latency |
|---|---|
| full scan (unindexed clause forces it) | 150.2 ms |
| **index-resolved** | **15.4 ms (9.8×)** |

Both strategies return identical, brute-force-verified counts; the mixed
test asserts each strategy is actually the one chosen. Reproduce:
`cargo test -p connxism --test filters -- --nocapture`.

**SQ8 scalar quantization** (`recall::quant`, per-vector affine codes;
asymmetric + symmetric dots stay closed-form): the graph holds codes,
searches over-fetch 2×, and hits are **rescored exactly** from the durable
vector column family — quantization is a memory decision, never a silent
accuracy decision.

| 5k × 64d (in-graph gate) | full f32 | SQ8 |
|---|---|---|
| recall@10 vs exact | 0.95+ (gate) | **0.982** |
| vector memory | 1,280,000 B | **380,000 B (3.4×)** |

| 2,048-doc quantized estate (end-to-end) | measured |
|---|---|
| recall@10 vs full-precision ground truth | **0.976** |
| returned scores | exact cosine (≤1e-5 from ground truth, asserted) |

Reproduce: `cargo test -p recall quantized_recall_gate -- --nocapture` and
`cargo test -p connxism --test quantized -- --nocapture`.

## Sprint 10: the query plane goes everywhere (2026-07-16)

The typed query contract (`EstateQuery` + `Filter`) moved into `rrf-core` —
pure data, no storage dependency — so the thin client speaks the **full**
filter DSL over the a2a wire, and the MCP `rrf_query` tool exposes it to
any MCP host. Text-only queries are embedded server-side: clients stay
weightless.

New retrieval strategies, all gated in-tree
(`cargo test -p connxism --test strategies -- --nocapture`):

- **Grouped search** — n groups × m per group, groups ordered by best hit;
  invariants asserted (distinct keys, membership, ordering).
- **Recommend** — steer toward positive examples, away from negatives, on a
  two-cluster corpus: **10/10 of the top-10 land in the positive cluster**,
  examples never returned; unknown positives are a typed error.
- **Discover** — context-pair agreement rerank: cluster-A hits in the top-10
  went **3/10 (neutral) → 7/10 (steered)** — every A member of the fetched
  pool ranked first (7 were all the pool held; the mechanism reranks the
  pool it fetches, honestly).
- **Batch** — one wire round-trip, results identical to one-at-a-time
  (asserted).

Wire gates (`cargo test -p rrf-client`): filter DSL binds over TCP (every
hit satisfies the clause set), lean payloads arrive lean, recommend works
remotely, estate-less nodes refuse `query` with a typed error, and the MCP
binding answers `rrf_query` with DSL end-to-end through a spawned server.

## Sprint 11: weighted sparse joins the fusion (2026-07-16)

`SparseVector` (learned-sparse / custom term weights) is now a first-class
signal: stored as one weighted posting row per (dimension, document) — the
same blind-put LSM-native layout as the BM25 index — searched by exact
accumulated dot product, and RRF-fused with the dense and lexical rankings
in the typed query plane.

Gates (`cargo test -p connxism --test sparse -- --nocapture`):
- ranking AND scores equal brute-force sparse dots (≤1e-5) — 200 docs, 3 queries;
- a planted df=1 dimension retrieves exactly its document; overwrite and
  removal retract the rows (asserted);
- a dense-invisible document surfaces in hybrid results **only** when the
  query carries its sparse signal — three-way fusion measured doing its job;
- sparse-only queries (no text, no dense vector) work standalone.

## Baseline hygiene: environments drift, gates are per-environment (2026-07-16)

Post-Sprint-11 the regression gate flagged estate query p50 (1.06 → 1.42 ms)
while the *unchanged* in-memory path simultaneously "improved" 3.6× — both
signatures of a different container instance, not a code change. Measured:
three identical release runs of the same ANN probe binary on this container
swing **394–670 µs/query (±65%)** from neighbor noise.

Actions taken, in order: reproduced the flag twice (it was consistent
within a session), probed the suspect hot loop A/B (no code-attributable
delta above the noise floor), then **re-recorded both container baselines
in the current environment** — estate: accuracy\@10 0.998, p50 1.42 ms,
8,989 docs/sec durable @ 50k; mem: 0.936, 29.15 ms. Gates re-verified
green against the fresh baselines.

Rule recorded: baselines are per-environment artifacts. Cross-container
comparisons need the variance probe first
(`cargo run --release -p recall --example annprobe`); a delta inside the
measured noise floor is drift, not regression — and accuracy deltas are
never excused this way (accuracy stayed 1.00/0.998 throughout).

## Sprint 12: multi-vector per point (2026-07-16)

Named vector spaces (each name its own dimensionality, one `nvecs` row per
(space, doc) — blind puts, exact cosine by sorted prefix scan) and
late-interaction token vectors (`mvecs`, MaxSim = Σ_q max_d q·d) joined the
estate and the typed query plane (`using` routes the dense half; `multi`
rescores a fetch-deep candidate set). Both ride the a2a wire via serde
defaults — old payloads still parse (gated).

Gates (`cargo test -p connxism --test multivec`):
- named ranking AND scores equal brute force (≤1e-5); title/body spaces
  rank independently;
- per-point named-vector update: dropped name retracts its row, sibling
  space untouched, remove retracts all, per-name dim guard errors;
- a dense-mediocre document with one planted token vector ranks first
  under MaxSim rescore and does NOT under plain dense; rescored score
  equals brute-force MaxSim.

Honest scope note: named-space search is exact (scan), not ANN — right
up to mid-size spaces; per-space graphs are the follow-up.

## Sprint 13: push-stream changefeed over a2a (2026-07-16)

`watch` joins `changes`: one long-lived a2a connection, the node drains the
durable feed from the client's seq cursor and then pushes each new change
the moment its write commits — event-driven via the estate's feed signal
(a write-side notify), with **zero polling on either side**. Cancel = drop
the connection; resume = the same seq cursor the poll verb uses; the token
gate covers streams. `Client::watch(since, callback)` is the Clyffy-side
handle; the transport grew a general `Handler::handle_stream` hook, so
future streamed verbs (query streaming, tailing) ride the same frame path.

Gates (`cargo test -p rrf-flow --test watch`): history drained in order,
live upserts AND a remove arrive as pushed frames on the one connection
within timeout, seqs strictly increase, reconnect from the returned cursor
replays exactly the missed change, unauthorized watch refused.

## Sprint 14: text analyzers (2026-07-16)

The lexical index grew a configurable analyzer pipeline — tokenizer
(word / whitespace / prefix edge-grams) × lowercase × stopwords × a Porter
stemmer authored from the published 1980 algorithm (zero-dep, 46 canonical
spec pairs gated). The analyzer is **part of the index's identity**: fixed
at estate creation, persisted in `EstateInfo`, applied identically to
postings and queries; existing estates deserialize to the exact legacy
pipeline they were indexed with.

Gates (`cargo test -p connxism --test analyzer` + rrf-core units):
- stemming estate matches run/runs/running to the same doc; the legacy
  estate does not stem (both asserted);
- pure-stopword queries return nothing (stopwords never reach postings);
- prefix analyzer serves autocomplete ("con" → connectome doc, "rea" →
  reason doc) straight off BM25;
- overwrite retracts postings through the same analyzer;
- reopen with a different config keeps the persisted analyzer (creation
  wins once, forever).

## Sprint 15: datetime/uuid indexes, highlighter, REBUILD (2026-07-16)

Payload indexes learned time and identity: RFC3339 strings index as
order-preserving epoch keys (`PIDX_DT` — range scans walk chronology with
early stop, offsets compared by instant), UUID strings as 16 raw bytes
(`PIDX_UUID`, 2.25× smaller keys). `Condition::DateRange` rides the DSL
(index-first when the field is indexed, post-filter otherwise — gated
equal). `Estate::rebuild_payload_index` is REBUILD INDEX for payloads and
the migration path for re-typed rows. `Analyzer::highlight` returns
byte-offset spans of the ORIGINAL text, analyzer-aware — a stemmed "run"
query highlights the surface form "running"; the prefix analyzer
highlights by prefix. RFC3339 parsing is zero-dep (days-from-civil,
offsets, fractional seconds — unit-gated on known instants).

Gates: index id-set equals brute-force truth (bounded + half-open +
offset-spelled bounds); uuid equality resolves from typed rows; rebuild
idempotent and erroring on unindexed fields; highlight spans slice the
original text to the expected surface forms.

## Sprint 16: named collections in one estate (2026-07-16)

Collections joined the estate as first-class scoping: membership is one
`coll` CF row per (collection, doc) — blind puts, retracted exactly on
move/remove — auto-registered with exact counts, and
`EstateQuery.collection` (serde default, rides the a2a wire) folds into
the scope/prefilter id-universe machinery with exact scoring inside the
collection. `Estate::drop_collection` fully retracts every member
(postings, vectors, payload/sparse/named rows, changefeed removes) and
deregisters the name.

Gates (`cargo test -p connxism --test collections`): two collections plus
uncollected floaters sharing identical vocabulary never leak into each
other's results at full depth; collection ∩ explicit scope intersects;
unknown collection returns empty; a moved doc leaves one and joins the
other; drop removes exactly its members (estate len, doc lookups, feed
row count, and search all asserted), leaving siblings and floaters
untouched.

## Sprint 17: offset, with_vectors, sampling, similarity matrix (2026-07-16)

Four small A3 rows closed, all wire-riding serde defaults:
`EstateQuery.offset` ranks to `offset+k` depth then pages (gated: each
page equals the full ranking's slice, on the fused pipeline);
`with_vectors` hydrates each winner's stored dense vector onto
`Candidate.vector` (gated equal to the upserted vectors, absent by
default, old payloads parse); `Estate::sample(n, seed)` draws a
deterministic seeded reservoir over the doc CF (reproducible, distinct,
n>corpus → whole corpus); `similarity_matrix(ids)` returns the pairwise
cosine upper triangle over stored vectors, skipping unknown ids (gated
against direct cosine, 1e-6).

## Sprint 18: aliases + per-point payload CRUD (2026-07-16)

Aliases: a single-blob alias map (create/list/switch/delete) resolved
anywhere a collection name is accepted — a repoint is atomic, so the same
live query flips from one collection to another without touching data
(gated). Payload CRUD per point: `set_payload` (merge),
`overwrite_payload`, `delete_payload_keys`, `clear_payload` — each is one
WriteBatch carrying the rewritten doc, exact payload-index
retraction/rewrite, the shape-census adjustment, and a changefeed row.
Gates assert the index-resolved id-sets before/after every op (old rows
gone, new rows live, siblings untouched), one feed row per op, and that
mutating a missing doc errors.
