# PARITY — the full union inventory

**Everything both reference engines expose — every protocol, endpoint,
statement, function namespace, index type, storage backend, capability —
extracted from their actual source trees on 2026-07-15, deduplicated, and
mapped to its rrf home.** This is the definitive build list: the union of
both engines, plus the RRF-only layer (RRD, readiness, connectome, warp
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
| Create / list / get / update / delete collections | estates + collections-in-estate (`connxism`) | 🔨 P3 (estates ✅, named collections ⬜) |
| Collection exists / info | `connxism::Estate::info` | ✅ partial |
| Aliases (create/list/switch) | estate alias registry | ⬜ P5 |
| Optimizer status + config (`/optimizations`) | segment maintenance (`connxism`) | 🔨 P5 |
| Cluster info / shard keys / move shard | mesh scale-out | ⬜ P8 |

### A2. Points (data plane)
| Capability | rrf home | Status |
|---|---|---|
| Upsert / get / delete points (REST+gRPC, wait/ordering) | `Recall::upsert/remove` + a2a `index` | ✅ core |
| Batch update ops (`points/batch`, UpdateBatch) | ingestion machine batches | ✅ |
| Update / delete named vectors per point | multi-vector records | ⬜ P2.5 |
| Set / overwrite / delete / clear payload | `connxism` doc metadata ops | 🔨 P3 |
| Scroll (paginated listing w/ filter) | `Estate::scroll` (cursor-paged) | ✅ |
| Count (exact/approx w/ filter) | `Estate::count` (filtered + free total) | ✅ |

### A3. Search & query plane
| Capability | rrf home | Status |
|---|---|---|
| Search / SearchBatch (dense KNN + filter + params) | ANN graph + pending overlay + filters | ✅ |
| Universal Query / QueryBatch / prefetch pipelines | `EstateQuery` typed builder (hybrid/filter/scope) | ✅ core; batch/prefetch 🔨 |
| Hybrid (dense + sparse/lexical fusion, RRF) | `hybrid_search` (BM25+dense, RRF-fused) | ✅ |
| SearchGroups / QueryGroups (group by payload field) | grouped recall | ⬜ P3 |
| Recommend / RecommendBatch / RecommendGroups (pos/neg examples) | recall strategies | ⬜ P4 |
| Discover / DiscoverBatch (context pairs steering) | recall strategies | ⬜ P4 |
| Search matrix (pairs/offsets similarity matrix) | analytics over recall | ⬜ P5 |
| Facet (value counts over payload field) | `Estate::facet` (exact, v1 scan) | ✅ |
| Random sampling | recall sampling | ⬜ P5 |
| Score threshold / offset / with_payload / with_vectors selectors | `EstateQuery::threshold` + `ids_only` (lean payload) | ✅ threshold/payload; offset/with_vectors 🔨 |

### A4. Index & storage internals
| Capability | rrf home | Status |
|---|---|---|
| Distance metrics: Cosine ✅, Dot, Euclid, Manhattan | `rrf-core::Embedding` + SIMD kernels | ✅ cosine/dot → 🔨 P2 rest |
| HNSW-class ANN graph (m, ef_construct, ef) | `recall::AnnIndex` (recall@10 ≥ 0.95 gated; out-of-band build) | ✅ |
| Plain (exact) index fallback | `FlatRecall` / estate scan | ✅ |
| Quantization: scalar u8 / PQ / binary / 1.5-bit+2-bit (TQ) | `recall::quant` SQ8 + exact rescore from durable vectors (recall@10 0.976 measured, 3.4× smaller) | ✅ scalar; PQ/binary 🔨 P2.5 |
| Sparse vectors + sparse index (inverted, on-disk variants) | weighted postings (`connxism`) | ✅ BM25 form → 🔨 P2 weighted |
| Multi-vector per point (named vectors, late-interaction/ColBERT-style) | multi-vector records | ⬜ P2.5 |
| Payload field indexes ×8: keyword, integer, float, bool, geo, text (full-text), datetime, uuid | `pidx` CF, order-preserving typed keys (keyword/int/float/bool ✅, 9.8× vs scan measured) | ✅ core; geo/datetime/uuid/full-text 🔨 |
| Filtering DSL (must/should/must_not, match/range/geo/nested, filtered KNN) | `Filter` (must/should/must_not × eq/any/range/exists), filter-first via `pidx` or post-filter | ✅; geo/nested 🔨 |
| Text index w/ tokenizers (word/whitespace/prefix/multilingual, stemmer, stopwords) | `rrf-core::text` grows analyzer support | 🔨 P3 |
| Geo index (radius/box/polygon) | estate geo CF | ⬜ P3 |
| WAL + flush/ack semantics | RocksDB WAL (✅ via estate) + explicit ack | ✅ base → 🔨 P5 semantics |
| Segments + optimizer (merge, vacuum, indexing thresholds) | estate maintenance tasks | 🔨 P5 |
| On-disk vs in-RAM storage toggles (vectors/payload/index) | estate storage profiles | ⬜ P5 |
| GPU-accelerated index build | `gpu` feature post-candle | ⬜ P7+ |
| gridstore-style blob storage | estate blob CF | ⬜ P8 (files) |

### A5. Snapshots / cluster / ops
| Capability | rrf home | Status |
|---|---|---|
| Snapshots: create/list/download/upload/recover (full + per-collection + per-shard) | `Estate::snapshot_to` (checkpoint; opens as a working estate) | ✅ core |
| Distributed: raft consensus, shards, replicas, transfers, recovery (`/cluster/*`) | warp-mesh scale-out | ⬜ P8 |
| `/metrics` (prometheus), `/healthz` `/livez` `/readyz` | events ✅ + health surface | 🔨 P5 |
| `/issues` (self-reported problems) | estate diagnostics from trends | ⬜ P5 |
| Telemetry endpoint | events/trends ✅ (DuckDB-native) | ✅ different-and-better |
| API keys / RBAC / JWT | capability tokens on a2a ✅ (L3 v1); RBAC/JWT 🔨 | ✅ v1 |
| Strict mode / resource limits | estate quotas | ⬜ P5 |

---

## B. Relational-engine surface (reference: SurrealQL + multi-protocol server)

### B1. Query language — 23 statements
`ACCESS, ALTER, CREATE, DEFINE, DELETE, FOREACH, IF/ELSE, INFO, INSERT,
KILL, LIVE, OPTION, OUTPUT/RETURN, REBUILD, RELATE, REMOVE, SELECT, SET/LET,
SHOW, SLEEP, UPDATE, UPSERT, USE`

| Capability | rrf home | Status |
|---|---|---|
| CRUD: CREATE/INSERT/SELECT/UPDATE/UPSERT/DELETE | typed query builder (`rrf-flow`) | 🔨 P3 |
| **RELATE** (graph edges) + graph traversal in SELECT (`->edge->node`) | `Estate::relate/traverse` + routed `scoped_search` (gate 1.000 vs 0.025) | ✅ |
| DEFINE ×17: access, analyzer, api, bucket, config, database, event, field, function, index, model, module, namespace, param, sequence, table, user | estate catalog (subset; see per-row mapping in C) | 🔨 P3–P6 |
| LIVE / KILL (live queries) | seq-resumable `changes` paging over a2a ✅; push-stream 🔨 | ✅ poll |
| SHOW CHANGES (changefeeds) | durable feed CF, atomic with writes | ✅ |
| Transactions (BEGIN/COMMIT/CANCEL) | RocksDB TransactionDB | 🔨 P3 |
| INFO (ns/db/table/index introspection) | estate info + `INFO`-verb on a2a | ✅ partial |
| REBUILD INDEX | estate reindex task | 🔨 P5 |
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
| `vector::*` (similarity/distance math) | `rrf-core::Embedding` ✅ + SIMD P2 | ✅ partial |
| `search::*` (score/highlight/offsets) | recall result annotations | 🔨 P3 |
| `crypto::*` (argon2/bcrypt/pbkdf2/blake3/md5…) | auth layer deps | 🔨 P5 |
| `http::*` (outbound calls from queries) | connector drivers instead (deliberate) | ⬜ different-by-design |
| `script::*` (embedded JS) | **WASM plugins instead** (`rrf-plugins`) | 🔨 P6 |
| `file::*` + buckets | estate blob/files | ⬜ P8 |
| `session::*`, `record::*`, `schema::*` | estate context fns | 🔨 P3/P5 |

### B3. Storage & indexes
| Capability | rrf home | Status |
|---|---|---|
| KV abstraction with backends: mem, rocksdb, surrealkv-class, tikv-class, indxdb (browser), FDB-class | `connxism::Db` seam (rocksdb ✅, mem 🔨 P3; distributed backends ⬜ P8) | ✅/🔨 |
| Full-text index: analyzers (tokenizers/filters/stemmers), BM25 scoring, highlighter, offsets | postings ✅ BM25 core; analyzers/highlight 🔨 P3 | ✅ core |
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
| **MCP endpoint (the reference serves MCP natively!)** | `rrf-mcp` stdio server ✅ (tools: ask/index/changes, end-to-end tested); HTTP-SSE transport ⬜ | ✅ core |
| ML endpoints (model upload/exec: surrealml-class) | DevPULSE model registry | 🔨 P7 |
| Auth: signin/signup, JWT, root/ns/db/record users, IAM roles | capability tokens | 🔨 P5 |
| Import/export (SQL dump) | estate export/import | 🔨 P5 |
| Client-ip / headers / CORS hardening | HTTP layer | 🔨 P5 |
| OS signals (SIGHUP/INT/QUIT/TERM) | ✅ **done, evented** | ✅ |
| Telemetry: OTLP traces/metrics | events ✅ + OTLP exporter | ⬜ P5 |
| WASM plugin runtime (surrealism-class: WIT, capability manifests, `.surli`-style packages) | `rrf-plugins` (wasmtime) | 🔨 **P6** |
| GUI/browser embedding (indxdb backend) | not a goal for the engine | ⬜ n/a |

---

## C. RRF-only (neither reference has these — the moat)
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
