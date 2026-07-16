# PARITY — the full union inventory

> ⚠️ **SUPERSEDED — the accuracy numbers below are SYNTHETIC.**
> They were produced by the deterministic hash embedder scoring synthetic
> vectors against synthetic vectors (a hash function grading itself), not by any
> real model. They say nothing about real retrieval. The first honest numbers —
> real models, a public benchmark, third-party judgments, with a BM25 baseline
> calibrated against published BEIR — are in [docs/BENCHMARKS_REAL.md](BENCHMARKS_REAL.md).
> Latency/throughput figures are likewise pre-real: measured ingest with a real
> model is ~1000x slower (10 docs/sec, not 10.9k).



**Everything both reference engines expose — every protocol, endpoint,
statement, function namespace, index type, storage backend, capability —
extracted from their actual source trees on 2026-07-15, deduplicated, and
mapped to its rrf home.** This is the definitive build list: the union of
both engines, plus the RRO-only layer (RRD, readiness, connectome, warp
mesh, DevPULSE), is what "done" means. Legend: ✅ built · 🔨 phase-assigned ·
⬜ inventoried, not yet scheduled.

Method: enumerated from the reference trees (`openapi.json` paths, gRPC
`.proto` services, `expr/statements/*`, `fnc/*`, `kvs/*`, `idx/*`,
`server/src/ntw/*`), not from memory. No code is ported — this list is
*capabilities to author*, per the zero-lineage law.

---

## A. Vector-engine surface (reference: 60 REST endpoints, 5 gRPC services)

### A1. Collection management
| Capability | rrf home | Status |
|---|---|---|
| Create / list / get / update / delete collections | estates ✅ + named collections in one estate (`coll` CF membership, auto-registered, exact counts, leak-proof scoped queries over the wire, drop with full retraction) | ✅ |
| Collection exists / info | `connxism::Estate::info` | ✅ partial |
| Aliases (create/list/switch) | alias map (atomic single-blob writes — switch redirects live queries without touching data; resolved anywhere a collection name is accepted) | ✅ |
| Optimizer status + config (`/optimizations`) | `cf_sizes()` in `HealthReport.cf_bytes` + manual `compact` verb; tuning knobs ⬜ | ✅ status |
| Cluster info / shard keys / move shard | mesh scale-out | ⬜ P8 |

### A2. Points (data plane)
| Capability | rrf home | Status |
|---|---|---|
| Upsert / get / delete points (REST+gRPC, wait/ordering) | `Recall::upsert/remove` + a2a `index` | ✅ core |
| Batch update ops (`points/batch`, UpdateBatch) | ingestion machine batches | ✅ |
| Update / delete named vectors per point | `nvecs` rows retract exactly on overwrite/remove via `StoredDoc.named_spaces` (gated: dropped name retracts, sibling spaces untouched) | ✅ |
| Set / overwrite / delete / clear payload | `set_payload`/`overwrite_payload`/`delete_payload_keys`/`clear_payload` — one WriteBatch each: exact pidx retract/rewrite, shape census, changefeed row (gated via index-resolved id-sets) | ✅ |
| Scroll (paginated listing w/ filter) | `Estate::scroll` (cursor-paged) | ✅ |
| Count (exact/approx w/ filter) | `Estate::count` (filtered + free total) | ✅ |

### A3. Search & query plane
| Capability | rrf home | Status |
|---|---|---|
| Search / SearchBatch (dense KNN + filter + params) | ANN graph + pending overlay + filters | ✅ |
| Universal Query / QueryBatch / prefetch pipelines | `EstateQuery` (typed contract in `rro-core`, executed by the estate **and over the a2a wire / MCP**) + `query_batch` + recursive `Prefetch` stages (union → exact outer rescore by any signal, depth-capped, gated vs hand-built) | ✅ |
| Hybrid (dense + sparse/lexical fusion, RRF) | `hybrid_search` (BM25+dense, RRF-fused) | ✅ |
| SearchGroups / QueryGroups (group by payload field) | `query_grouped` (n groups × m per group, best-first) | ✅ |
| Recommend / RecommendBatch / RecommendGroups (pos/neg examples) | `recommend` (avg-positive − avg-negative steering, examples excluded; a2a verb + client) | ✅ core |
| Discover / DiscoverBatch (context pairs steering) | `discover` (pair-agreement rerank over the fetched pool) | ✅ core |
| Search matrix (pairs/offsets similarity matrix) | `similarity_matrix(ids)` — pairwise cosine over stored vectors, unknown ids skipped (gated vs direct cosine) | ✅ |
| Facet (value counts over payload field) | `Estate::facet` — index-first run-length over sorted `pidx` rows for string/bool fields (zero doc reads, gated equal to the scan); canonical-key types fall back to the exact scan honestly; `distinct()` on top | ✅ |
| Random sampling | `Estate::sample(n, seed)` — deterministic seeded reservoir over the doc CF (gated: reproducible, distinct, exact edge cases) | ✅ |
| Score threshold / offset / with_payload / with_vectors selectors | `EstateQuery::threshold` + `ids_only` + `offset` (exact pagination on every strategy) + `with_vectors` (`Candidate.vector`) | ✅ |

### A4. Index & storage internals
| Capability | rrf home | Status |
|---|---|---|
| Distance metrics: Cosine ✅, Dot, Euclid, Manhattan | `rro-core::Embedding` (cosine/dot SIMD kernels; euclidean/manhattan) | ✅ |
| HNSW-class ANN graph (m, ef_construct, ef) | `recall::AnnIndex` (recall@10 ≥ 0.95 gated; out-of-band build) | ✅ |
| Plain (exact) index fallback | `FlatRecall` / estate scan | ✅ |
| Quantization: scalar u8 / PQ / binary / 1.5-bit+2-bit (TQ) | `recall::quant` SQ8 + exact rescore from durable vectors (recall@10 0.976 measured, 3.4× smaller) | ✅ scalar; PQ/binary 🔨 P2.5 |
| Sparse vectors + sparse index (inverted, on-disk variants) | `SparseVector` contract + weighted postings CF (one row per (dim, doc), exact accumulated dot, RRF-fused with dense+lexical) | ✅ |
| Multi-vector per point (named vectors, late-interaction/ColBERT-style) | named spaces (`nvecs` CF, per-name dims, exact cosine) + token vectors (`mvecs` CF) with MaxSim rescore in the query plane (`using` / `multi`, over the wire) | ✅ exact; per-space ANN 🔨 |
| Payload field indexes ×8: keyword, integer, float, bool, geo, text (full-text), datetime, uuid | `pidx` CF, order-preserving typed keys (keyword/int/float/bool/**datetime** (RFC3339→epoch keys, chronological range scans)/**uuid** (16-byte keys) ✅, 9.8× vs scan measured) | ✅ core; geo/full-text 🔨 |
| Filtering DSL (must/should/must_not, match/range/geo/nested, filtered KNN) | `Filter` (must/should/must_not × eq/any/range/date_range/**geo_radius**/**geo_box**/exists), filter-first via `pidx` or post-filter | ✅; nested 🔨 |
| Text index w/ tokenizers (word/whitespace/prefix/multilingual, stemmer, stopwords) | `Analyzer` pipeline (word/whitespace/prefix-edge-gram × lowercase × stopwords × Porter stemmer, authored from the published algorithm), persisted per estate — postings and queries always agree | ✅ core; multilingual ⬜ |
| Geo index (radius/box/polygon) | `PIDX_GEO` Z-order keys (Morton, 26 bits/axis, monotone-per-axis gated) — one range scan covers any box, exact haversine/box re-check at the doc level; no antimeridian wrap in v1 (documented); polygon ⬜ | ✅ radius/box |
| WAL + flush/ack semantics | RocksDB WAL ✅ + explicit `Estate::flush` (memtables + WAL sync; `flush` verb / `Client::flush`) + `fsync_writes` config (synced write options per batch) | ✅ |
| Segments + optimizer (merge, vacuum, indexing thresholds) | RocksDB background compaction ✅ + manual `Estate::compact` (full-range, per CF; `compact` verb) + per-CF live SST bytes in health | ✅ core |
| On-disk vs in-RAM storage toggles (vectors/payload/index) | estate storage profiles | ⬜ P5 |
| GPU-accelerated index build | `gpu` feature post-candle | ⬜ P7+ |
| gridstore-style blob storage | estate blob CF | ⬜ P8 (files) |

### A5. Snapshots / cluster / ops
| Capability | rrf home | Status |
|---|---|---|
| Snapshots: create/list/download/upload/recover (full + per-collection + per-shard) | `Estate::snapshot_to` (checkpoint; opens as a working estate) | ✅ core |
| Distributed: raft consensus, shards, replicas, transfers, recovery (`/cluster/*`) | warp-mesh scale-out | ⬜ P8 |
| `/metrics` (prometheus), `/healthz` `/livez` `/readyz` | zero-dep ops HTTP listener (`serve_ops`, daemon: `RRO_OPS_ADDR`) — prometheus 0.0.4 gauges incl. per-collection; probes 200 (gated over a real socket) + `health` a2a verb / `Client::health` | ✅ |
| `/issues` (self-reported problems) | `Estate::issues(threshold)` — applier backlog, dim unset, feed/doc divergence — surfaced in the `health` verb and `rro_issues_total` (gated: fires on backlog, clean when drained) | ✅ |
| Telemetry endpoint | events/trends ✅ (DuckDB-native) | ✅ different-and-better |
| API keys / RBAC / JWT | capability tokens on a2a ✅ (L3 v1); RBAC/JWT 🔨 | ✅ v1 |
| Strict mode / resource limits | `Quotas` (max_docs, max_payload_bytes, max_top_k, max_batch) — typed `RroError::Quota` at the write/query boundaries, reported in health, clean wire refusals; daemon `RRO_STRICT=1` | ✅ |

---

## B. Relational-engine surface (reference: SurrealQL + multi-protocol server)

### B1. Query language — 23 statements
`ACCESS, ALTER, CREATE, DEFINE, DELETE, FOREACH, IF/ELSE, INFO, INSERT,
KILL, LIVE, OPTION, OUTPUT/RETURN, REBUILD, RELATE, REMOVE, SELECT, SET/LET,
SHOW, SLEEP, UPDATE, UPSERT, USE`

| Capability | rrf home | Status |
|---|---|---|
| CRUD: CREATE/INSERT/SELECT/UPDATE/UPSERT/DELETE | typed query builder (`rro-engine`) | 🔨 P3 |
| **RELATE** (graph edges) + graph traversal in SELECT (`->edge->node`) | `Estate::relate/traverse` + routed `scoped_search` (gate 1.000 vs 0.025) | ✅ |
| DEFINE ×17: access, analyzer, api, bucket, config, database, event, field, function, index, model, module, namespace, param, sequence, table, user | estate catalog (subset; see per-row mapping in C) | 🔨 P3–P6 |
| LIVE / KILL (live queries) | poll (`changes`) ✅ + push-stream `watch` over a2a: event-driven frames (estate feed signal, zero polling), seq-resume, token-gated, `Client::watch` (KILL = drop the connection) | ✅ |
| SHOW CHANGES (changefeeds) | durable feed CF, atomic with writes + `feed_stats()` (first/next seq, retained rows) in `info` | ✅ |
| Transactions (BEGIN/COMMIT/CANCEL) | RocksDB TransactionDB | 🔨 P3 |
| INFO (ns/db/table/index introspection) | `info` a2a verb / `Client::info`: identity, analyzer, dims, payload indexes, collections, aliases, quotas, health, feed stats (gated over TCP) | ✅ |
| REBUILD INDEX | `Estate::rebuild_payload_index` (drop + backfill; the typed-key migration path, gated) | ✅ payload; postings 🔨 |
| Permissions-per-field/table, record-level auth | auth layer | 🔨 P5 |
| Full DSL parser (`RRQL`) | only after the builder proves the semantics | ⬜ P6 |

### B2. Function library — 32 namespaces
`api, args, array, bytes, count, crypto, duration, encoding, file, geo,
http, math, object, parse, rand, record, schema, script, search, sequence,
session, set, sleep, string, time, type, util, value, vector, mod, not,
operate`

| Capability | rrf home | Status |
|---|---|---|
| Core value fns (array/object/string/math/time/type/parse/…) | builder expression layer | 🔨 P3 (as needed by builder) |
| `vector::*` (similarity/distance math) | `rro-core::Embedding` ✅ + SIMD P2 | ✅ partial |
| `search::*` (score/highlight/offsets) | scores ✅ + `Candidate.highlights` byte-offset spans (analyzer-aware) ✅ | ✅ |
| `crypto::*` (argon2/bcrypt/pbkdf2/blake3/md5…) | auth layer deps | 🔨 P5 |
| `http::*` (outbound calls from queries) | connector drivers instead (deliberate) | ⬜ different-by-design |
| `script::*` (embedded JS) | **WASM plugins instead** (`rrf-plugins`) | 🔨 P6 |
| `file::*` + buckets | estate blob/files | ⬜ P8 |
| `session::*`, `record::*`, `schema::*` | estate context fns | 🔨 P3/P5 |

### B3. Storage & indexes
| Capability | rrf home | Status |
|---|---|---|
| KV abstraction with backends: mem, rocksdb, surrealkv-class, tikv-class, indxdb (browser), FDB-class | `connxism::Db` seam (rocksdb ✅, mem 🔨 P3; distributed backends ⬜ P8) | ✅/🔨 |
| Full-text index: analyzers (tokenizers/filters/stemmers), BM25 scoring, highlighter, offsets | postings ✅ BM25 + `Analyzer` ✅ + highlights **on candidates over the wire** (`EstateQuery.highlight` → `Candidate.highlights`, offset-exact, gated over TCP) | ✅ |
| HNSW + DiskANN vector trees | **excised in the reference by the author's own design — replaced by Recall** | ✅ by architecture |
| Index planner / query optimizer (streaming + legacy) | query planning in builder | 🔨 P3/P5 |
| Sequences | estate sequence CF | 🔨 P3 |
| Changefeeds (cf) | changefeed CF | 🔨 P4 |
| Catalog (ns/db/tables/defs) | estate catalog CF | 🔨 P3 |

### B4. Protocols & server surface
| Capability | rrf home | Status |
|---|---|---|
| HTTP REST (`/sql`, `/key/*` CRUD, import/export, health, version, sync) | HTTP read surface | 🔨 P5 |
| WebSocket RPC (bidirectional, live query delivery) | a2a TCP ✅ + WS binding | 🔨 P4/P5 |
| **GraphQL** | after typed builder | ⬜ P6 |
| **MCP endpoint (the reference serves MCP natively!)** | `rro-mcp` stdio server ✅ (tools: ask/query/index/changes/**collections**/**payload**, end-to-end tested); HTTP-SSE transport ⬜ | ✅ core |
| ML endpoints (model upload/exec: surrealml-class) | DevPULSE model registry | 🔨 P7 |
| Auth: signin/signup, JWT, root/ns/db/record users, IAM roles | capability tokens | 🔨 P5 |
| Import/export (SQL dump) | estate export/import | 🔨 P5 |
| Client-ip / headers / CORS hardening | HTTP layer | 🔨 P5 |
| OS signals (SIGHUP/INT/QUIT/TERM) | ✅ **done, evented** | ✅ |
| Telemetry: OTLP traces/metrics | events ✅ + OTLP exporter | ⬜ P5 |
| WASM plugin runtime (surrealism-class: WIT, capability manifests, `.surli`-style packages) | `rrf-plugins` (wasmtime) | 🔨 **P6** |
| GUI/browser embedding (indxdb backend) | not a goal for the engine | ⬜ n/a |

---

## C. RRO-only (neither reference has these — the moat)
| Capability | rrf home | Status |
|---|---|---|
| **RRD — reason-ready object JIT** (modes→slivers lattice, per-shape plans, RROs) | `rrd` crate | 🔨 **in progress now** |
| RRD session triggers: fire on conversation start / idle-resume; intent detection → mode switch ("we need to be in X mode"); expert state absorbs the task list | `rrd::trigger` | 🔨 now |
| Readiness gate (classifier daemon) | ✅ heuristic → RRO-based P4 → learned P7 | ✅ core |
| Connectome visual map (flow + estate) | ✅ | ✅ |
| Warp points / layer-2 a2a (local≡remote, +3 ms measured) | ✅ TCP; MCP mesh P5 | ✅ core |
| Griff — the operator-voice layer (plain-language nudging for non-technical operators; never silently rewards bad specs) | Clyffy-side consumer of readiness + connectome (out of this repo's scope; contract: RROs + readiness are its inputs) | ⬜ interface only |
| DuckDB-native event stream + baseline gates | ✅ | ✅ |
| DevPULSE embedder/reranker/classifier | plug-points ✅, forward passes P7 | 🔨 P7 |
| Ingestion machine w/ observable states + backpressure | ✅ | ✅ |

---

## D. Direct dependencies observed (appendix, informational)

Deps are implementation *choices*, not obligations — rrf selects its own and
keeps the tree lean (current: tokio, serde, thiserror, async-trait, tracing,
uuid, rocksdb, criterion, proptest, tempfile). For the record, the reference
trees' notable direct deps:

- **Vector engine:** actix-web (REST), tonic/prost (gRPC), rocksdb (payload
  storage), tokio, serde, bitpacking/bitvec (postings), charabia (multilingual
  tokenize), geo/geohash, half (f16), memmap2, quantization (own crate),
  prometheus, jsonwebtoken, ndarray, rayon, io-uring, foyer (cache),
  hdrhistogram, arc-swap, dashmap, parking_lot.
- **Relational engine:** axum (HTTP/WS), async-graphql, opentelemetry/OTLP,
  jsonwebtoken + argon2/bcrypt/pbkdf2 (auth), rocksdb + tikv-class +
  own-KV-class backends, fst + logos (parsing/ft), flatbuffers + wasmtime
  (plugins), ort (ONNX ML), geo/geo-types, blake3/md5, rayon, mimalloc/
  jemalloc, dashmap, radix_trie, papaya.

rrf equivalents get chosen per phase gate, never imported wholesale.

---

**This document + PLAN.md are the contract.** PLAN.md owns phase gates;
PARITY.md owns the exhaustive "what"; nothing ships as "parity" until its row
flips ✅ with a measured gate behind it.
