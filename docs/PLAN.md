# Reason Ready — The Master Plan

**One engine. Recall (vector memory) · Connectome (relations & the visible
map) · Reason Ready Flow (the readiness-gated pipeline). Clean-authored,
zero external lineage, every claim measured.**

This is the governing document. Reference systems were studied as *capability
inventories only* — every line in this tree is authored for rrf. No code is
ported, no wrappers exist, no upstream is tracked.

## Laws (unchanged, non-negotiable)

1. **Zero lineage.** Capabilities get re-authored; code never gets copied.
2. **Measured claims.** A number without an `rrf-bench` run does not exist.
   Baselines gate regressions; events stream to DuckDB.
3. **Trait boundaries.** Components plug in behind `rrf-core` traits; the flow
   never changes when an implementation does.
4. **Every phase lands green**: tests + clippy + bench + baseline + docs,
   committed and pushed before the next phase starts.

## Where we stand (done & measured)

| Capability | Home | Proof |
|---|---|---|
| Engine contract (traits, types, errors) | `rrf-core` | property tests |
| Deterministic embedder + DevPULSE (Qwen) plug-point | `embedder` | tests + bench |
| In-memory recall (exact cosine) | `recall` | tests + bench |
| BM25 reranker + DevPULSE (Nemotron) plug-point | `reranker` | tests + bench |
| Reason-ready daemon (coverage heuristic, message service) | `classifier` | tests |
| Flow map + estate map rendering (JSON/DOT) | `connectome` | tests |
| RocksDB estate: nodes, warp points, connectors, tags, shapes, trends | `connxism` | integration tests incl. reopen |
| Persistent BM25 inverted index (LSM-native, blind puts) | `connxism` | 762→8,883 docs/sec measured |
| Hybrid search (dense + lexical, reciprocal rank fusion) | `connxism` + contract | tests + bench |
| Ingestion machine (backpressure, batches, observable states, drain) | `rrf-flow` | tests incl. backpressure |
| a2a: in-proc bus + TCP transport | `rrf-net` | tests |
| Full SignalKind set, consistently emitted | `rrf-flow` | events verified |
| DuckDB-ready JSONL event stream | `rrf-core::events` | 199-event run verified |
| Baseline configuration & tracking (regression gate) | `rrf-bench` | gate re-run passed |
| CI, supply-chain policy, MSRV, coverage | `.github`, `deny.toml` | CI green |

Measured on this container: **115k docs/sec** ingest (mem), **8.9k docs/sec**
durable (estate), hybrid p50 63 ms (mem) / 113 ms (estate) @ 50k docs — exact
scan, no ANN yet. See `docs/BENCHMARKS.md`.

## Capability matrix — everything the reference engines offer, represented

Legend: ✅ done · 🔨 phase assigned · ⏸ deliberately later · ❓ needs your input

### Vector engine capabilities → Recall

| Capability (reference example) | rrf representation | Phase |
|---|---|---|
| Exact dense search | `recall::FlatRecall`, `connxism` dense scan | ✅ |
| **ANN graph index (HNSW-class)** | `recall::AnnIndex` — authored clean, trait-swapped | **P2** 🔨 |
| SIMD distance kernels | `rrf-core::simd` (portable + `std::simd` when stable) | P2 🔨 |
| Scalar quantization (int8) | `recall::quant::Scalar` | P2 🔨 |
| Product/binary quantization | follow-on behind same trait | ⏸ P2.5 |
| Sparse vectors + posting lists | `connxism` postings generalized to weighted sparse | P2 🔨 |
| BM25 sparse embedding (bm25_embed) | `embedder::SparseBm25` (embedder-side, like your edge layer) | P2 🔨 |
| Payload (metadata) filtering | filter pushdown into recall traits + estate secondary indexes | P2 🔨 |
| Payload indexes (keyword/int/geo) | `connxism` `idx` CF family | P2 🔨 |
| Facets / counts / scroll | `edge`-style surface on `rrf-flow` (`query/count/facet/info`) | P3 🔨 |
| Collections & optimizer | estates already partition; segment merge/optimize | P5 🔨 |
| WAL + crash recovery | RocksDB WAL now; kill-9 recovery **proof tests** | P5 🔨 |
| Snapshots / backup | `Estate::snapshot()` via RocksDB checkpoints | P5 🔨 |
| Sharding / distributed | after mesh is real | ⏸ P8 |
| GPU accel | behind `gpu` feature, after candle lands | ⏸ P7+ |

### Relational/graph capabilities → Connectome + connXism

| Capability (reference example) | rrf representation | Phase |
|---|---|---|
| KV abstraction w/ backends (mem/rocksdb/…) | `connxism::Db` — second backend proves the seam | P3 🔨 |
| **Records & relations (graph edges, RELATE-style)** | `connxism` `rel` CF: `(from, verb, to)` rows, both directions | **P3** 🔨 |
| **Graph traversal / route resolution (map→treasure)** | `connectome::route`: traverse relations → feed recall — *your hybrid design, re-authored* | **P3** 🔨 |
| Structured query API (SQL-class surface) | typed `Query` builder first (`RRQL` DSL only when the builder is proven) | P3 🔨 / DSL ⏸ P6 |
| Transactions | RocksDB `TransactionDB` (optimistic) in connxism | P3 🔨 |
| Change feeds | event stream ✅ + durable changefeed CF | P4 🔨 |
| **Live queries / subscriptions** | `watch` on estate mutations over a2a (`subscribe` verb) | **P4** 🔨 |
| Auth / IAM | token-scoped capabilities on a2a + gRPC surfaces | P5 🔨 |
| GraphQL/HTTP/RPC APIs | **gRPC (tonic)** first-class; HTTP read surface after | P5 🔨 |
| Observability/telemetry | events + trends ✅; OTLP export later | ✅ / ⏸ |
| **WASM plugin runtime (surrealism-class, the JIT)** | `rrf-plugins`: wasmtime host, capability manifest, warp-callable | **P6** 🔨 |
| ML model storage (surrealml-class) | DevPULSE model registry in estate (`models` CF + weights refs) | P7 🔨 |
| Files/buckets | connector-fed blobs in estate | ⏸ P8 |

### Reason Ready Flow (yours alone — no reference has this)

| Capability | rrf representation | Phase |
|---|---|---|
| Readiness gate | `classifier` ✅ → learned DevPULSE classifier | P7 🔨 |
| Visible reasoning map | `connectome` ✅ → live UI feed over a2a `map` verb | P4 🔨 |
| Connector sync drivers (mail/drive/db → estate) | `connectors` crate: driver trait + cursors (state machinery ✅) | P4 🔨 |
| MCP mesh warp transport | `rrf-net::mcp` — warp points become live jump targets | P5 🔨 |
| DevPULSE models (Qwen embed, Nemotron rerank) | candle forward passes behind `candle` feature | **P7** 🔨 |
| Inference bake-offs (vLLM, llama.cpp, candle, candle-vllm) | `rrf-bench --backend` matrix per ADR-0001 | P7 🔨 |
| Python owns training; Rust owns serving/memory/kernel | ADR-0001 ✅ (decided) | ✅ |

## The phases

**P2 — Recall at scale** (the biggest measured win)
ANN index authored clean (layered small-world graph), SIMD kernels, scalar
quantization, sparse/weighted postings, payload filters + secondary indexes.
*Gate:* recall@10 ≥ 0.95 vs exact on 1M synthetic; p50 < 10 ms @ 1M (estate);
baseline gates re-recorded; property tests for index invariants.

**P3 — Connectome relations** (your map→treasure, real)
Relations CF + RELATE-style API, traversal, route resolution feeding recall,
typed query builder, optimistic transactions, second KV backend proving the
seam. *Gate:* route→recall e2e beats flat hybrid on a linked corpus, measured;
transaction isolation tests.

**P4 — Live flow** — durable changefeeds, live subscriptions over a2a,
connector driver trait + first two drivers (filesystem, IMAP-class), live
connectome UI feed. *Gate:* subscription latency measured; connector sync
resumes from cursor after kill.

**P5 — Surface & ops** — gRPC (tonic), auth capabilities, snapshots,
kill-9 crash-recovery proof suite, optimizer, Docker + frictionless deploy.
*Gate:* recover-from-crash test green 100/100 runs; deploy from zero in one
command.

**P6 — Plugins (the JIT)** — `rrf-plugins` wasmtime runtime: capability
manifests, scoped host imports (query/kv/events), warp-callable modules.
*Gate:* a plugin module runs a scoped flow query; capability escape tests.

**P7 — DevPULSE models & bake-offs** — candle Qwen embedder + Nemotron
reranker forward passes, learned readiness classifier, model registry,
`rrf-bench --backend candle|vllm|llamacpp|candle-vllm` quality+latency matrix.
*Gate:* DevPULSE beats the weightless floor on a golden retrieval set — the
bake-off decides the Clyffy engine with data, not preference.

**P8 — Scale-out** — replication, sharding over the mesh, estate federation.
Only after P2–P7 are proven; distributed correctness is earned, not assumed.

## Open items needing you

1. **silver** — not in any repo I have. Which repo, or describe it → it gets a
   phase.
2. **RRD (your JIT)** — if it's the WASM-runtime shape, P6 is its home; if it's
   something else (query JIT? kernel codegen?), point me at the design.
3. **DevPULSE weights** — when Qwen/Nemotron tuned checkpoints exist, P7 wires
   them; until then the plug-points stay honest about being unloaded.

## Operating rhythm

Every phase: author → test (unit/property/integration) → measure
(`rrf-bench` + baseline gate) → document (BENCHMARKS/OBSERVABILITY/ADR) →
commit → push. No phase is "done" without its gate output pasted into the
docs. That is how this stays an engine and never becomes a circle again.
