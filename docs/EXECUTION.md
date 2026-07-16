# Execution — the operating loop

Every unit of work runs the same loop, and no step counts as done without its
verification output:

```
PLAN      state the step, its design, and its verification gate — in writing
EXECUTE   author it (clean, trait-boundaried, evented)
VERIFY    run the gate: tests + clippy + bench/baseline + the specific proof
RECORD    numbers → BENCHMARKS.md / events; design → ADR; status → this file
COMMIT    push green; never stack unverified work
```

Phases and gates live in [PLAN.md](PLAN.md). This file tracks the active
sprint at step granularity.

## Sprint 1 — Prove the flow against a popular RAG (active)

Rationale: the engine's own numbers show promise (115k docs/sec mem,
8.9k docs/sec durable, hybrid p50 63–113 ms @ 50k — `BENCHMARKS.md`), which
per the operator's rule unlocks a public-baseline comparison. The claim to
reproduce, with defined metrics this time: high ingestion multiple, top-rank
retrieval accuracy, and the full pipeline (embed → hybrid recall → rerank →
classify) over the **a2a layer-2 path** performing on par with a popular RAG
store doing *less* work over HTTP.

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | This outline | committed | ✅ |
| 2 | **accuracy@k** in `rrf-bench`: planted golden docs (one unique-marked golden per query; accuracy = golden in top-k) | unit test on planting; metric printed + evented | ✅ estate **1.000**, mem-dense 0.936 (hybrid is the difference) |
| 3 | **a2a remote path**: `rrf-bench --remote <addr>` queries a live `rrf` daemon over layer-2 TCP (full pipeline per query) | remote run returns identical accuracy to local; latency recorded | ✅ remote **1.000** == local; p50 191 ms vs 188 ms local (+3 ms for the wire); ingest 6,480 docs/sec over a2a |
| 4 | **Baseline harness** (outside the tree): same corpus, same precomputed vectors, into ChromaDB embedded + ChromaDB HTTP server | baseline ingest/query/accuracy numbers emitted | ✅ 566–586 docs/sec, acc 0.572–0.606, p50 3–5 ms |
| 5 | **The bake-off**: rrf (local + a2a) vs baseline (embedded + HTTP), identical inputs | results table + methodology in BENCHMARKS.md; no metric asserted without a run | ✅ 11.7× durable ingest, 1.000 vs 0.606 accuracy, +3 ms wire cost; ANN latency gap quantified → P2 |
| 6 | Green close: fmt/clippy/tests, baselines re-gated, commit+push | CI-green tree | ✅ |

**Methodology guards (so the comparison is honest):**
- Identical corpus and identical pre-computed vectors for both systems — this
  compares *engines*, not embedding models.
- rrf runs its **full** pipeline (embed→hybrid→rerank→classify) per query;
  the baseline does plain vector top-k — rrf doing more work at comparable
  latency *is* the claim.
- Accuracy is defined (golden-doc@k on planted queries), not vibes. The
  historical "1.0 accuracy / 130x" numbers are treated as targets to
  re-demonstrate under this defined protocol, never as pre-accepted facts.
- Single shared container, same run window, release builds; environment noted.

## Sprint 2 — P2: Recall at scale (active)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | ANN index (layered small-world graph, diversity heuristic, soft deletes) | recall@10 ≥ 0.95 vs exact (property test) | ✅ |
| 2 | Estate integration, two-phase (durable vecs → post-commit apply → rebuild-on-open) | estate tests incl. persistence/reopen | ✅ |
| 3 | Gate run @50k + @100k | accuracy 1.000 held; p50 **1.40 ms / 2.09 ms** (was 188.5 ms — 135×); baseline re-recorded | ✅ |
| 4 | Ingest cost honesty | 8,883 → 488 docs/sec recorded; fix path = out-of-band graph apply (compaction-style) | ✅ recorded, fix queued |
| 5 | Unrolled dot kernels (ANN traversal + Embedding) | no regression; tests hold | ✅ graph build ~3× faster; query p50 improved (1.40→1.06 ms @ 50k) |
| 6 | Out-of-band graph apply (applier thread + pending overlay + quiesce, per the recovered compaction pattern) | ingest ≥ 5k docs/sec with ANN on; read-your-writes test; accuracy + ms-latency held post-quiesce; catch-up time reported honestly | ✅ ingest **10.8–11k docs/sec** (2× the gate, above pre-ANN); p50 1.06/1.88 ms @ 50k/100k; accuracy 1.000 @ 100k (0.998 @ 50k — one fusion-cutoff miss, noted); catch-up 31 s/71 s reported. Found+fixed: Estate drop stopped the applier (bench/daemon now hold the estate). |
| 7 | Scalar quantization, weighted sparse, payload filters | per-PLAN gates | ⬜ queued next sprint |

## Sprint 3 — P3: Connectome relations (active)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Relations CF: RELATE-style `(from, verb, to)`, both directions, blind puts | unit tests: relate/unrelate/out/in roundtrip | ✅ |
| 2 | Traversal: typed spec (verbs, depth, limit), BFS | traversal tests incl. depth/verb filters | ✅ |
| 3 | Route→recall: traverse → `scoped_search` (exact dense by point-lookup + scoped BM25, RRF-fused) | scoped search returns only in-scope docs | ✅ |
| 4 | **The measured gate**: ambiguous linked corpus — routed disambiguates what flat hybrid cannot | **flat accuracy@1 = 0.025 vs routed = 1.000** (40 queries, 1500-doc noise floor, ANN path live) — printed from the in-tree gate test | ✅ |
| 5 | Green close + docs + push | CI-green tree | ✅ |

Deferred honestly to Sprint 4: optimistic transactions (WriteBatch atomicity
already covers batch writes), second KV backend, full typed query builder.

## Sprint 4 — P4: Connectors & the live flow (active)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Durable changefeed: every upsert/remove appends a feed row **in the same WriteBatch** (atomic with the write); `Estate::changes(since_seq)` | ordering + resume-by-seq asserted in sync tests | ✅ |
| 2 | `connectors` crate: `Driver` trait (resumable cursor batches) + filesystem driver + JSONL-feed driver | drivers exercised end-to-end in-container | ✅ |
| 3 | Sync engine: pull → **RRD distill** (mode+tags land in the estate) → upsert → RELATE `connector→contains→doc` → cursor advance, evented | fs sync test: 7 docs, 7 provenance edges, 7 `mode:document` tags, feed ordered — and RRD's mode votes caught bad driver field-naming (`path`→Location) and forced honest metadata (`title`/`source_path`) | ✅ |
| 4 | **The resume gate**: interrupt a sync, restart — no duplicates, cursor holds | simulated outage on pull #3: 6 docs durable, cursor held at "6"; resume ingested exactly 4; final count 10, feed shows exactly 10 upserts (replay-free) | ✅ |
| 5 | Changes over a2a: `changes` verb (poll-based subscription, seq-resumable, `next_seq` cursor in every reply) | daemon exposes estate via ServeOptions; verb returns paged changes | ✅ |
| 6 | Green close + docs + push | CI-green tree | ✅ |

Deferred honestly: push-streaming subscriptions (poll-based lands first),
IMAP-class driver (needs a live mailbox; the driver trait is its socket).

## Sprint 5 — RRD-first & the evolving shape baseline (closed same day)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `rrd::baseline`: recency-weighted shape distributions per context; speculative prediction (inline-cache), entropy predictability, PSI drift vs committed snapshots, O(1) decay | unit gates: hit-rate → 1.0 on stable stream; mono ≈ 1.0 vs mega < 0.05 predictability; PSI fires on regime change while identity flips slowly; snapshot/restore roundtrip | ✅ |
| 2 | **RRD literally first — ingest**: sync runs stamp→gates→shape→predict *before* embedding; blocked docs never reach the model (`SyncReport::blocked`); L2 tags route on survivor embeddings | sync tests green; ordering enforced in code | ✅ |
| 3 | **RRD literally first — query**: `flow.ask` stage 1 is `rrd` (gate + mode); blocked queries return gated with zero model cost; intent tags on every `RecallResult` | flow compiles + stage evented first | ✅ |
| 4 | Baseline persists in the estate and grows across sessions (`x:rrd:baseline`); predictability exported as estate trend + `connector.synced` event fields | cross-session gate: fresh Rrd restores snapshot, first-prediction hits, hit-rate never regresses across sessions | ✅ |

## Sprint 6 — Turnkey: one engine, one command (closed same day)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Daemon runs all five components as ONE: RRD attached to the flow, baseline restored on boot, committed on shutdown | quickstart run shows stage order rrd→embed→recall→rerank→classify per query | ✅ live run |
| 2 | `scripts/quickstart.sh` — build→boot→smoke over a2a in one command | executed: 13,597 docs/sec ingest over the wire, accuracy 1.000, p50 1.15 ms | ✅ live run |
| 3 | `scripts/mesh.sh N` — N full engines, each an a2a warp point | executed: 3 nodes, all accuracy 1.000 | ✅ live run |
| 4 | Deploy artifacts: Dockerfile (multi-stage), systemd unit (clean SIGTERM = baseline commit), config.env | authored; Docker build not yet CI-verified (no daemon in this container) — flagged honestly | ✅/⚠ |
| 5 | docs/COMPARISON.md — the head-to-head; README rewritten to the engine as it is | reflects only measured claims | ✅ |

## Sprint 7 — P5 ops: snapshots, crash-proof, capability auth (active)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Estate snapshots: `Estate::snapshot_to(path)` via RocksDB checkpoint; a snapshot opens as a full working estate | ✅ point-in-time verified (post-snapshot writes excluded; relations captured) | ✅ |
| 2 | **Kill-9 crash suite**: a child process ingests then `abort()` (no destructors, no flush) — the reopened estate must be consistent (counts, search, feed) and the ANN rebuilds | ✅ in-tree; **30/30 hard-death rounds recovered** (10× loop × 3 rounds) | ✅ |
| 3 | a2a capability auth v1: shared-secret token on the wire (`Message.token`, serde-defaulted); nodes with a token reject non-bearers (ping stays open); RRF_TOKEN env | ✅ authorized/unauthorized/wrong-token over live TCP; `a2a.unauthorized` evented | ✅ |
| 4 | Green close + docs + push | CI-green tree | ✅ |

Deferred honestly: gRPC surface + MCP transport binding (next slice of P5 —
tonic/proto scaffolding deserves its own sprint), full IAM (capability
attenuation per L3, after tokens prove the seam).

## Sprint 8 — Data plane + client + MCP

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `EstateQuery` builder: text/vector/top_k + metadata **filter** + optional scope, executed hybrid with over-fetch post-filter (payload hydration for lexical hits) | ✅ every hit satisfies the filter | ✅ |
| 2 | Facets, filtered count, scroll (cursor-paged listing) on the estate | ✅ facet counts exact (20/40 split); scroll covers all 60 docs, zero overlap | ✅ |
| 3 | `rrf-client` crate: typed async client over a2a (ping/ask/index/changes/map, token-aware) — what Clyffy imports | ✅ against a live node; typed refusals surfaced | ✅ |
| 4 | **MCP binding, real**: `rrf-mcp` stdio server (JSON-RPC 2.0; initialize / tools list+call) bridging any MCP client to a node | ✅ end-to-end: spawned the binary, spoke MCP, full-pipeline answers came back with candidates | ✅ |
| 5 | Green close + docs + push | CI-green tree | ✅ |

## Sprint 9 — Filter DSL + payload indexes + quantization

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `Filter` DSL: must/should/must_not over eq / match-any / numeric range / exists; serde-serializable; merged with the legacy equality form at execution | ✅ clause semantics unit-tested; DSL results equal brute force on both strategies | ✅ |
| 2 | Payload secondary indexes: `pidx` CF (`field \0 typed-value \0 doc`), order-preserving f64 encoding, rows maintained in the same WriteBatch as every upsert/remove, register-then-backfill `create_payload_index` | ✅ overwrite retracts old rows; remove retracts; **9.8× vs full scan @10k, identical counts** | ✅ |
| 3 | Two-strategy execution: filter-first (index-resolved exact id-set → exact scoring inside it) when fully indexed and ≤4096 ids, over-fetch + hydrate + post-filter otherwise | ✅ test asserts each strategy is the one actually chosen | ✅ |
| 4 | Query options: `score_threshold`, `ids_only` lean payload | ✅ threshold prunes, lean strips text+metadata | ✅ |
| 5 | SQ8 scalar quantization: `recall::quant` (per-vector affine codes, asymmetric+symmetric dots), `AnnConfig.quantized`, `EstateConfig { quantized }`, exact rescore from durable vectors | ✅ graph gate recall@10 **0.982** (3.4× smaller); estate gate **0.976** with scores exact ≤1e-5 | ✅ |
| 6 | Green close + docs + push | CI-green tree (fmt/clippy/test: 0 warnings, 40 suites green) | ✅ |

Deferred honestly: `protoc` still absent in this container → gRPC surface
stays deferred (a2a JSON wire + `rrf-client` + MCP remain the integration
paths); geo/datetime/uuid/full-text payload index types; nested filters.

## Sprint 10 — The query plane everywhere + retrieval strategies

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Query contract moved to `rrf-core` (`EstateQuery`, `Filter`, `Condition` are pure data; connxism executes and re-exports) — thin clients need no storage dep | ✅ workspace green after the move; connxism API unchanged for consumers | ✅ |
| 2 | a2a `query` verb: body IS an `EstateQuery`; text-only queries embedded server-side; `recommend` verb beside it; estate-less nodes refuse with typed errors | ✅ over live TCP: DSL binds, lean payloads, typed refusals | ✅ |
| 3 | `Client::query` + `Client::recommend`; MCP `rrf_query` tool (DSL pass-through) | ✅ client tests + MCP end-to-end with a filter clause | ✅ |
| 4 | Grouped search: `query_grouped(q, field, groups, group_size)` | ✅ invariants: distinct keys, ≤ sizes, membership, best-first group order | ✅ |
| 5 | Recommend / Discover: example-steered and context-pair-steered retrieval | ✅ two-cluster gates: recommend 10/10 in the positive cluster (examples excluded); discover 3/10 → 7/10 (all pool positives ranked first) | ✅ |
| 6 | `query_batch` + Euclid/Manhattan metrics on `Embedding` | ✅ batch ≡ sequential (asserted) | ✅ |
| 7 | Green close + docs + push | fmt/clippy/test: 0 warnings, 41 suites green | ✅ |

## Sprint 24 — Prefetch pipelines + index-first facets

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `Prefetch { query, limit }` on `EstateQuery` (recursive, serde-default — rides the wire): each prefetch gathers candidates by its own signal; their union becomes the outer query's id universe (∩ explicit scope), rescored exactly by the outer signal (dense / sparse / MaxSim). Depth-capped | two-stage dense→MaxSim pipeline equals the hand-built equivalent; sparse→dense; union of two prefetches; depth cap errors; old payloads parse | ✅ |
| 2 | Index-first `facet`: run-length over the sorted `pidx` rows for STR/NUM/BOOL-tagged fields (rows sort by typed value — counting distinct values is one prefix scan, no doc reads); DT/UUID/GEO/OTHER tags fall back to the exact doc scan honestly; `Estate::distinct(field)` on top | indexed facet equals the full-scan facet on the same estate (before/after indexing); mixed-tag fields fall back | ✅ |
| 3 | Green close: fmt/clippy/test, PARITY rows (prefetch ✅, facet index-first), BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 23 — Lexical top-k: max-score pruning + postings fast path

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Per-term document frequencies maintained **blind** via a RocksDB merge operator (`tdf` CF, i64 add: +1 per new (term,doc), −1 on retraction) — no read-modify-write, the LSM law holds | df counter equals actual postings count through upsert/overwrite/remove | ✅ |
| 2 | Binary postings values (8-byte tf+len LE; JSON fallback read for pre-existing rows) — kills the per-row JSON parse on the hot path | old-format rows still score (fallback gated) | ✅ |
| 3 | Max-score (Turtle–Flood-class, authored from the concept) lexical top-k: df point-reads → per-term upper bounds → essential-term scans + **point lookups** (term\x00doc) for non-essential terms; heap top-k. Falls back to the full scorer when df stats are absent | **exactness**: pruned top-k ids AND scores identical to the full scorer on selective, common-term, and mixed workloads (randomized corpora) | ✅ |
| 4 | Measured before/after: featbench common-term hybrid + a mixed rare+common lexical bench — recorded honestly whatever the sizes | numbers in BENCHMARKS.md | ✅ |
| 5 | Green close: fmt/clippy/test, commit, push | full workspace green | ✅ |

## Sprint 22 — Regression pass + feature-latency baseline

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Variance probe ×3, then both recorded baseline gates (mem + estate, 50k/500, ±25%) | gates pass, or any regression is investigated before recording | ✅ |
| 2 | `featbench` example: one 50k estate exercising every sprint 11–21 path — indexed-filter, geo radius, sparse-fused, MaxSim-rescored, collection-scoped p50s + watch push-frame delivery latency | honest numbers recorded in BENCHMARKS.md as the Sprint-22 feature baseline | ✅ |
| 3 | README feature list refreshed to match sprints 11–21 reality | README current | ✅ |
| 4 | Green close: fmt/clippy/test, commit, push | full workspace green | ✅ |

## Sprint 21 — Geo: haversine conditions + Z-order index keys

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `rrf-core::geo`: `{lat, lon}` extraction from metadata, haversine (mean-radius great-circle, meters), `Condition::GeoRadius`/`GeoBox` with exact post-filter `matches` | haversine unit gates on known city pairs; condition matches on hand-checked points | ✅ |
| 2 | `PIDX_GEO` typed keys: 26-bit/axis quantization, Z-order (Morton) interleave authored from the concept — monotone per axis, so one `[z(min corner), z(max corner)]` scan covers any box (false positives culled by an exact doc-level post-check) | Morton monotonicity property test; index-resolved id-sets EQUAL brute-force truth | ✅ |
| 3 | Index-first geo filters: radius → bounding box → Z-range scan → exact haversine/box check against stored metadata; wired into `ids_for_condition` (query plane + counts get it free) | seeded city grid: box + radius id-sets equal brute force; query plane returns exactly the truth set; overwrite retracts | ✅ |
| 4 | Honest limits documented: no antimeridian wrap in v1 (boxes must not cross ±180°), poles clamp | stated in code + PARITY | ✅ |
| 5 | Green close: fmt/clippy/test, PARITY geo rows, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 20 — Ops surface: health verb, /metrics + /healthz HTTP, issues

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `Estate::health()` snapshot (docs, feed seq, applier backlog via new `Pending::backlog()`, collections, dims, quantized) + `Estate::issues(threshold)` self-report (dim unset with docs, applier backlog high, feed/doc-count divergence) | issues reports a planted applier backlog; healthy estate reports none | ✅ |
| 2 | `health` a2a verb (uptime + estate snapshot) + `Client::health` | verb answers with live numbers over TCP | ✅ |
| 3 | Zero-dep ops HTTP listener (`serve_ops`): GET `/metrics` (prometheus text), `/healthz` `/livez` `/readyz` (200 ok), 404 else; daemon mounts it via `RRF_OPS_ADDR` | raw-socket GET parses: 200 + `rrf_docs_total <n>` gauges; healthz 200 | ✅ |
| 4 | Green close: fmt/clippy/test, PARITY A5 rows, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 19 — The sprint 12–18 surface over the wire + MCP

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | a2a verbs (token-gated like all verbs): `matrix`, `sample`, `collections`, `drop_collection`, `create_alias`, `aliases`, `delete_alias`, `set_payload`, `overwrite_payload`, `delete_payload_keys`, `clear_payload` (named/sparse search already ride `query` via `using`/`sparse` — gated over the wire, no new verbs) | each verb answers over live TCP | ✅ |
| 2 | `Client` methods for every verb above | wire results equal local calls (matrix pairwise scores, deterministic sample, collection lists, alias redirects, payload visibility) | ✅ |
| 3 | MCP: `rrf_collections` (list/drop/alias actions) + `rrf_payload` (set/overwrite/delete_keys/clear) tools | tool registry lists them; actions bridge to the node | ✅ |
| 4 | Green close: fmt/clippy/test, PARITY mesh/tooling rows, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 18 — Aliases + per-point payload CRUD

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Alias registry (`META_ALIASES` map, atomic single-blob writes): create/list/switch/delete; `EstateQuery.collection` resolves aliases everywhere a collection name is accepted | alias query returns the target's docs; switch atomically redirects the same query to the new target without touching data; delete restores literal behavior | ✅ |
| 2 | Payload ops per point (writer-serialized, one WriteBatch each): `set_payload` (merge), `overwrite_payload`, `delete_payload_keys`, `clear_payload` — pidx rows retracted/rewritten exactly, shape census maintained, changefeed row appended atomically | `ids_where` before/after each op reflects exactly the change; feed grows by one row per op; doc metadata asserted after each op | ✅ |
| 3 | Green close: fmt/clippy/test, PARITY rows (A1 aliases, A2 payload ops), BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 17 — Query-plane completions: offset, with_vectors, sampling, matrix

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `EstateQuery.offset` (serde default): rank to `offset+k` depth, skip, take — pagination on every strategy (scoped, prefiltered, hybrid, named, fused) | page 2 equals the full ranking's `[k..2k]` slice exactly | ✅ |
| 2 | `EstateQuery.with_vectors` + `Candidate.vector` (serde-defaulted contract field): winners hydrate their stored dense vector | returned vectors equal the upserted ones (1e-6); absent by default; old payloads parse | ✅ |
| 3 | Random sampling: `Estate::sample(n, seed)` — deterministic reservoir over the doc CF (no RNG deps) | n distinct existing docs; same seed → same sample; n > corpus → whole corpus | ✅ |
| 4 | Search matrix: `ConnXRecall::similarity_matrix(ids)` — pairwise cosine over stored vectors (upper triangle) | matrix equals direct cosine on the seeded vectors; unknown ids skipped | ✅ |
| 5 | Green close: fmt/clippy/test, PARITY A3 rows, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 16 — Named collections in one estate

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Registry + membership: `coll` CF (`collection \x00 doc` → empty, blind puts), `StoredDoc.collection`, `VectorRecord.in_collection` builder; auto-registered on first use, `Estate::collections()` lists with exact counts | membership rows retract on overwrite (move a→b) and on remove | ✅ |
| 2 | Collection-scoped queries: `EstateQuery.collection` (serde default — rides the a2a wire), folded into the scope/prefilter id-universe machinery; exact scoring inside the collection | two collections + uncollected floaters in one estate never leak into each other's results | ✅ |
| 3 | `Estate::drop_collection`: removes exactly its members (full retraction — postings, vectors, pidx, sparse, named, feed rows), deregisters | estate len drops by exactly the member count; other collections + floaters untouched; changefeed shows the removes | ✅ |
| 4 | Green close: fmt/clippy/test, PARITY A1 named collections, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 15 — Datetime/UUID payload indexes + highlighter + REBUILD

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `rrf-core::time`: zero-dep RFC3339 → epoch-ms (offsets, fractional seconds, leap days); `Condition::DateRange` (gt/gte/lt/lte as RFC3339 strings) with post-filter `matches` | unit gates: known timestamps, offset math, ordering | ✅ |
| 2 | Typed pidx keys: datetime-parsing strings → `PIDX_DT` (order-preserving epoch), UUID-format strings → `PIDX_UUID` (16 raw bytes); symmetric encode on write & query so Eq/Any/Exists just work | key-order unit gates | ✅ |
| 3 | `DateRange` index-first: ordered scan under the DT tag with early stop; `Estate::rebuild_payload_index` (drop + backfill — REBUILD INDEX parity + the migration path for re-typed rows) | filter unit test: exact id-set from the index equals brute force; rebuild keeps queries working | ✅ |
| 4 | `Analyzer::highlight`: byte-offset spans of tokens whose analyzed form matches the analyzed query (stemmed query highlights the inflected surface form; prefix analyzer highlights by prefix) | spans slice the original text to the expected surface forms | ✅ |
| 5 | Green close: fmt/clippy/test, PARITY rows (pidx types, highlighter, REBUILD), BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 14 — Text analyzers (tokenizers, stemmer, stopwords)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `rrf-core::text::Analyzer`: tokenizer (word / whitespace / prefix edge-grams) × lowercase × stopwords × Porter stemmer (authored from the published algorithm, zero-dep) | stemmer unit gates on canonical spec pairs; prefix/whitespace tokenizer unit gates | ✅ |
| 2 | Estate-configurable: `EstateConfig.analyzer` persisted into `EstateInfo` at creation (serde default = legacy word+lower+stop, so existing estates are untouched); BM25 postings AND lexical queries both run through the estate's analyzer | reopen keeps the analyzer; index/query agreement asserted | ✅ |
| 3 | Retrieval gates: stemmed estate matches "run"→"running" doc lexically, legacy estate doesn't; stopwords produce zero postings rows; prefix analyzer serves autocomplete ("con"→"connectome") | in-tree tests | ✅ |
| 4 | Green close: fmt/clippy/test, PARITY analyzer rows, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 13 — Push-stream changefeed subscriptions over a2a

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `Handler::handle_stream` in rrf-net (default: not-a-stream) + serve loop forwards stream frames until the producer closes | existing single-reply verbs untouched (whole suite green) | ✅ |
| 2 | Write-side signal: `Estate` feed `Notify`, fired by `ConnXRecall::upsert`/`remove` after commit — watchers wake event-driven, zero internal polling | watch test observes frames arrive without any poll interval | ✅ |
| 3 | `watch` verb on FlowNode (token-enforced): drains `changes(since)` pages into frames, then awaits the signal; resume-by-seq preserved | frames carry `change` + `next_seq`; seqs strictly increasing | ✅ |
| 4 | `Client::watch(since, on_change)` — long-lived connection, callback per change, cursor returned on stop; dropping the callback cancels | e2e over TCP: live upserts arrive as frames; reconnect with returned cursor sees only new changes; unauthorized watch refused | ✅ |
| 5 | Green close: fmt/clippy/test, PARITY LIVE/KILL row push-stream ✅, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 12 — Multi-vector per point (named spaces + late interaction)

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | Contract: `VectorRecord.named` (name → vector, per-space dims) + `VectorRecord.multi` (token vectors); `rrf_core::maxsim` (Σ_q max_d q·d); `EstateQuery.using` + `EstateQuery.multi` ride the wire via serde defaults | serde roundtrip incl. new fields; old payloads still parse | ✅ |
| 2 | Storage: `nvecs` CF (`name \x00 doc` → f32-LE) + `mvecs` CF (doc → [n][dim][f32…]); per-name dim guard in `EstateInfo.named_dims`; retraction on overwrite/remove via `StoredDoc.named_spaces`/`multi_len` | planted named vector retrieved; overwrite drops removed names; remove retracts; dim mismatch errors | ✅ |
| 3 | `named_search`: exact cosine over one named space (sorted prefix scan) | ranking + scores equal brute force over the space; cross-space isolation (title-hit ≠ body-hit) | ✅ |
| 4 | Late interaction: MaxSim rescore stage in the query plane (`using` routes the dense half; `multi` rescores fetch-deep candidates) | MaxSim scores equal brute force; planted-token doc ranks first under `multi` and does NOT under plain dense | ✅ |
| 5 | Green close: fmt/clippy/test, PARITY rows 34+61 ✅, BENCHMARKS note, push | full workspace green | ✅ |

## Sprint 11 — Weighted sparse vectors + three-way fusion

| # | Step | Verification gate | Status |
|---|---|---|---|
| 1 | `SparseVector` in the core contract (indices/values, merge-join dot); `VectorRecord.with_sparse`; `EstateQuery.sparse` rides the wire via serde | ✅ unit + serde covered by the existing contract tests | ✅ |
| 2 | Sparse postings CF: one row per (dim BE, doc) → f32 weight — blind puts in the same WriteBatch; `StoredDoc.sparse_dims` retracts rows exactly on overwrite/remove | ✅ planted df=1 dimension hits exactly its doc; overwrite and remove retract | ✅ |
| 3 | `sparse_search`: exact accumulated dot via per-dimension sorted prefix scans | ✅ rank order AND scores equal brute force (≤1e-5) on 200 docs × 3 queries | ✅ |
| 4 | Query-plane fusion: sparse ranking RRF-fused with dense+lexical; respects scope/prefilter id universes; sparse-only queries work | ✅ dense-invisible doc surfaces only when the sparse half is present; sparse-only returns exactly the target | ✅ |
| 5 | Green close + docs + push | fmt/clippy/test green across the workspace | ✅ |

## Sprint log

- **S1 opened 2026-07-15.** Sliver/RRD design recovered into ADR-0002 during
  the sprint.
- **S1 closed 2026-07-15.** All six gates ran and passed; results recorded in
  [BENCHMARKS.md](BENCHMARKS.md) §Bake-off. Headlines: hybrid accuracy
  **1.000** (vs 0.572–0.606 baseline on identical inputs), **11.7×** durable
  ingest, a2a wire cost **+3 ms** at identical accuracy. Known loss: exact-
  scan query latency vs ANN (~190 ms vs 3–5 ms @ 50k) — quantified, feeds
  Sprint 2 (P2 ANN).
