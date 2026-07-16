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

## Sprint log

- **S1 opened 2026-07-15.** Sliver/RRD design recovered into ADR-0002 during
  the sprint.
- **S1 closed 2026-07-15.** All six gates ran and passed; results recorded in
  [BENCHMARKS.md](BENCHMARKS.md) §Bake-off. Headlines: hybrid accuracy
  **1.000** (vs 0.572–0.606 baseline on identical inputs), **11.7×** durable
  ingest, a2a wire cost **+3 ms** at identical accuracy. Known loss: exact-
  scan query latency vs ANN (~190 ms vs 3–5 ms @ 50k) — quantified, feeds
  Sprint 2 (P2 ANN).
