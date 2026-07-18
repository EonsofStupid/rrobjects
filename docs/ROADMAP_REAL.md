# ROADMAP_REAL.md — everything left, fully mapped, in priority order

**This is the execution SSOT.** `PLAN.md` holds the original vision, `PARITY.md`
the capability inventory, `ROADMAP.md` the old phase status. Where any of them
disagree with this file about *what is built* or *what happens next*, this file
wins. It is re-grounded on code read in July 2026, not on doc claims.

_Reconciled 2026-07-17. Phases 0–4 + shape-as-early-intent done and merged to `main`; Phase 5 next._

---

## The goal, stated exactly

**One engine — RRO — with the vector, graph (connectome), and query capabilities
a memory engine needs, plus RRD + embedder + reranker + classifier, over RocksDB
(`connxism`), and turnkey: fully built and fully tested.** SurrealDB and Qdrant
were **guidance absorbed, not authorities to match** — there is no SurrealDB in
this code; the map layer is `connectome` and the engine is RRO. Then pulled into
clyffy as one config line. Proof last, on real data.

## Where it actually stands

RRO today is **a strong vector engine + the connectome graph layer + its own
query language (RRQL) + GraphQL + the RRD/model spine**. What's still open: the
full data model (transactions ✅, but namespaces and schemafull enforcement are
not built), the RAM-scale story (mmap/segments), and the broader serving surface.
Those are capabilities the engine is still growing into — not a gap against any
other product.

The three things most often asked about:

| | status |
|---|---|
| **RocksDB** | ✅ real — 16 CFs; WAL/crash recovery proven by `abort()`×3 with 500 docs intact. Workload tuning missing (Phase 5). |
| **connXism** | ✅ real, and the strongest part of the codebase. |
| **GraphQL** | ✅ 2026-07-17 — real parser+executor, rides the a2a transport (a language, not a transport). Query-side; mutations follow. |

---

## BUILT — do not re-plan, do not rebuild

Verified by reading code *and call sites*, not `README`/`ASSESSMENT` (both stale).
**283 tests, 0 warnings, 14 crates.**

- **Vector:** HNSW — geometric level draw, `ef` beam, diversity-heuristic neighbour
  selection, soft-delete tombstones (`recall/src/ann.rs`, gates recall@10 ≥ 0.95) ·
  SQ8 with exact rescoring (`recall/src/quant.rs`) · named vector spaces · weighted
  sparse vectors · MaxSim late interaction.
- **Query:** filter DSL — must/should/must_not × eq/any/range/date/geo-radius/
  geo-box/exists, **true haversine** (`rro-core/src/query.rs`) · payload secondary
  indexes with rebuild · collections (leak-tested) · aliases · prefetch pipeline ·
  facets · scroll · grouping.
- **Graph:** RELATE / traverse, dual-row blind-put so both directions are prefix
  scans (`connxism/src/rels.rs`); wire-parity proven against local.
- **Durability:** changefeed written atomically with the change · event-driven
  `watch` (notify armed before drain — no lost updates) · snapshots · **WAL/crash
  recovery** (child process `abort()`, ×3 rounds, 500 docs survive).
- **RRQL:** 2,355 L, 63 tests. Hand-written lexer → parser → AST → lower.
  SELECT/DEFINE/REMOVE/UPDATE/DELETE/RELATE/TRAVERSE/INFO. Lowers to the typed
  `Filter` **or refuses** — no silent degradation.
- **Surface:** ~25 a2a verbs (NDJSON/TCP) · MCP server, 7 tools.
- **Intelligence:** RRD (2,081 L, 31 tests — shape, JIT plan cache, L0/L1/L2 gate
  ladder, centroid semantic router, PSI drift baseline) · constrained-decode
  classifier with logprob confidence · model-registry.
- **Models:** deterministic (CI) · candle Qwen3 (hand-written cache-free encoder) ·
  llama.cpp + vLLM over OpenAI-compatible HTTP · MRL truncation (32–1024,
  truncate-then-normalize, tested) · BM25 / candle cross-encoder / HTTP rerankers ·
  `IdentityReranker`.
- **Turnkey:** `quickstart.sh` works with zero env vars · `fetch-models.sh`
  (byte-exact verified Qwen3 catalog, 2.4 GB baseline).
- **Dogfood:** RRO is Claude's memory — `UserPromptSubmit` recall + capture, daemon
  under systemd, MCP registered (`integrations/claude-code/`).

---

## THE CAPABILITY MATRIX — every remaining gap, mapped to a phase

The columns are capabilities a memory engine wants, grouped by which reference
tool the *idea* is associated with — **not** a scoreboard against them. RRO
implements the ones it has its own way. There is no SurrealDB or Qdrant in the code.

### Map / document capabilities (the connectome side)

| capability | status | where | phase |
|---|---|---|---|
| ACID transactions | ⚠️ 2026-07-17 — multi-op atomic `transaction()` over upsert/remove with verified rollback; RRQL BEGIN/COMMIT + other ops = 5b-2 | `connxism/src/txn.rs` | **5** |
| Namespaces / databases above collections | ❌ | `connxism` | **10** |
| Schemafull `DEFINE` enforcement, `ALTER`, `REMOVE` | ❌ — DEFINE parses; nothing enforces | `rro-ql`, `connxism` | **10** |
| `LIVE` / `KILL` | ⚠️ **parse-only — refuses at execution**, points at `watch` | `rro-engine/src/sql.rs:182` | **10** |
| Record links | ⚠️ partial — RELATE covers the edge case | `rels.rs` | **10** |
| Users / roles (reader/writer/admin) | ✅ 2026-07-17 — per-verb RBAC at the `FlowNode` gate; a reader's `sql` write is refused too | `rro-engine/src/auth.rs`, `handler.rs` | ~~12~~ |
| JWT (local HS256) | ✅ 2026-07-17 — **hand-rolled** SHA-256 + HMAC (NIST/RFC vectors), local-key HS256, no external crypto/JWT crate, no JWKS/outbound (self-hosted signs+verifies). Expired/wrong-sig rejected, constant-time MAC compare | `rro-engine/src/auth.rs` | ~~12~~ |
| Namespace-scoped permissions | ✅ 2026-07-17 — token `ns` claim; a scoped token is refused on a node serving another namespace (the same key cannot bypass it) | `rro-engine/src/auth.rs` | ~~12~~ |
| HTTP REST doorway | ✅ 2026-07-17 — thin HTTP/1.1 front door (no new crate, no hyper/axum): builds the a2a `Message` and calls the one `FlowNode::handle`, so HTTP is byte-identical to a2a by construction. `POST /query` · `POST /ask` · `GET /health` · probes · `POST /v/{verb}` escape hatch. Bearer token → the same capability gate. Gate: `POST /query` ≡ a2a `query` ≡ `estate.recall().query()` | `rro-engine/src/http.rs` | ~~11~~ |
| WebSocket RPC | ❌ streaming (`watch`/`live`) doorway — follow-on | `rro-engine/src/http.rs` | **11** |
| **GraphQL** | ✅ 2026-07-17 — query **+ mutation** surface (parser+executor) over the a2a transport, NOT a bolted-on HTTP server. GraphQL is a language, not a transport. `graphql` verb + `rro_graphql` MCP. `mutation { upsert / delete }` write through the same estate the reads query; a reader's mutation is refused like a reader's `sql` write. Introspection/subscriptions = follow-on | `rro-engine/src/graphql.rs` | ~~11~~ |
| Import / export | ❌ | `rro-http` | **11** |
| Distributed / cluster | ✅ 2026-07-17 — **self-reliant failover, no openraft** (3 stages). (1) replication: `replicate` verb + `Replica` follower rebuilds a leader's estate from the stream alone (convergent, idempotent, cursor-resumable, byte-identical vectors). (2) synchronous quorum-ack: `Cluster` learns follower positions from poll cursors (the poll IS the ack); `durability:"quorum"` `tx` blocks until a quorum holds it; an under-replicated write refuses to falsely ack. (3) `Lease` (heartbeat liveness) + `elect` (highest-cursor promotion). **Gate met**: kill the leader under write load → the highest-cursor survivor promotes and holds every acked write (majority-intersection); the laggard would have lost data. Known bound (documented, accepted): quorum-log replication, NOT linearizable consensus — no split-brain proof under partition | `rro-engine/src/{replica,cluster}.rs`, tests `replication`/`quorum_ack`/`failover` | ~~13~~ |
| Full-text search + analyzers | ✅ | `index.rs`, `text::Analyzer` | — |
| Geospatial | ✅ true haversine | `query.rs` | — |
| Graph traversal | ✅ | `rels.rs` | — |
| Change feeds | ✅ | `model.rs`, `keys.rs` | — |

### Vector capabilities

| capability | status | where | phase |
|---|---|---|---|
| **Filter-aware HNSW** — predicate applied *during* traversal | ✅ 2026-07-17 — `search_filtered`, three-way strategy by exact cardinality | `recall/src/ann.rs`, `connxism/src/query.rs` | ~~4~~ |
| Cardinality estimation | ✅ 2026-07-17 — the resolved id-set length IS the exact cardinality | `query.rs` | ~~4~~ |
| mmap vectors / segments / O(1) startup | ❌ **graph is RAM-resident and rebuilds O(N log N) on open — capacity ≈ RAM** | `recall/src/ann.rs`, `estate.rs` | **6** |
| Immutable segments + background optimizer | ❌ | `connxism` | **6** |
| PQ / BQ quantization | ❌ — SQ8 is the only quantizer | `recall/src/quant.rs` | **6** |
| DBSF (distribution-based score fusion) | ✅ 2026-07-17 — `distribution_score_fusion` + `FusionMode`, wired to `EstateQuery::fusion_mode`, magnitude-preservation unit-proven | `connxism/src/index.rs` | ~~7~~ |
| Shard keys / scatter-gather | ❌ | `rro-net` | **13** |
| Replication / raft | ❌ | `rro-net` | **13** |
| REST + gRPC surface | ❌ | `rro-http` | **11** |
| HNSW + `ef` tuning | ✅ 2026-07-17 — swept at **50k** (structured): knee ef≈32, default 64 clears it 2× with headroom; gated `knee <= default` | `ann.rs` (`ef_search_sweep_50k`), `BENCHMARKS_REAL.md` §5b | — |
| SQ8 quantization | ✅ | `quant.rs` | — |
| Named + sparse + multivector | ✅ | `query.rs` | — |
| Payload indexes, collections, aliases | ✅ | `filter.rs`, `keys.rs` | — |
| Snapshots | ✅ | `estate.rs` | — |
| Discovery / recommendation | ✅ | handler verbs | — |
| Grouping | ✅ | `strategies.rs` | — |

### RRO-only — the reason this engine exists

| capability | status | phase |
|---|---|---|
| RRD gate ladder + centroid semantic router | ✅ | — |
| RRD baseline: shape prediction, predictability, PSI drift | ✅ | — |
| **Shape as early intent** — COSTAR fields → distinct slivers, 97%-gated speculation, cross-session-stable ids | ✅ 2026-07-17 | — |
| Constrained-decode readiness classifier | ✅ | — |
| Embedder + reranker in-binary, 3 engines each | ✅ built · ✅ **proven (Phase 3)** | **3** |
| MRL truncation | ✅ inference · ❌ training | **9** |
| Matryoshka / quantization-aware **training** | ❌ greenfield | **9** |
| TOON recall→LLM encoding | ❌ zero in tree | **8** |
| Claude memory dogfood | ✅ | — |

---

## FINDINGS that shape the plan (measured, not asserted)

1. **Fusion earns nothing on nfcorpus, and it is not a bug.** Two rescue attempts
   failed. Stemming lifted BM25 0.3115→0.3283 — and moved our baseline from 4%
   under the published BEIR figure to within 1% of it — but the regression
   survived. Weighting swept 1:1→8:1 (0.3943→0.4114) and **converges toward dense
   (0.4120) from below, never crossing**: optimal lexical weight ≈ 0. (Finding 1.)
2. **Every weightless default is lexical, and they punish the dense half.**
   `LexicalReranker` re-sorts semantic results lexically — live, the right answer
   scored **0.0000**; `HeuristicClassifier` judges by term coverage
   (`insufficient @ 0.00` for a perfect hit); RRF scores are `1/(60+rank)`,
   magnitude-free. One root cause, three symptoms. (Finding 4.)
3. **No relevance gate exists, and none can be built on RRF.** ANN returns *k*
   however distant; RRF discards magnitude; readiness is lexical. Nothing can tell
   "found the answer" from "found the nearest four things". **Same root cause as
   (1)** — which is why Phase 7 is DBSF + per-query routing, not another constant.
4. ~~**The three engines are unproven.**~~ **RESOLVED 2026-07-17** — 23 gates run
   on the GB10, 0 failed. candle reproduces the model card to 6 decimals and
   agrees with llama.cpp on identical weights; both rerankers lift 0.50 → 1.00.
   `scripts/gates.sh` runs them; CI still cannot (no weights). See Finding 5.
   **`ef` remains untuned** — the ANN gate passed on only 2,200 vectors.
5. **Burn training is recoverable, not greenfield.** 2,655 LOC of Qwen3-in-Burn
   with *verified sm_121 gates* lived at `kernel/devpulse-clyffy/`;
   `~/Projects/platform_devpulse` no longer exists. **`~/Projects/qortex-rro-archive.bundle`
   is the likely survivor — recover before re-authoring.**

---

## THE PHASES

**Done:** ~~0 reconcile~~ · ~~1 dogfood~~ · ~~2 identity~~ · ~~3 prove the engines~~ · ~~4 filter-aware HNSW~~ — merged, CI green.

### ~~3 — Prove the three engines~~ ✅ DONE (2026-07-17)

`scripts/gates.sh` → **5 suites, 23 tests, 0 skipped, 0 failed.** The candle
encoder reproduces Qwen's published card to 6 decimals; llama.cpp's independent
C++ implementation on the *same weights* (0.6B GGUF on `:8095`) agrees. Rerankers
lift BM25 0.50 → 1.00 on both llama.cpp and vLLM, with identical ordering. See
`BENCHMARKS_REAL.md` Finding 5.

**Still open from this phase, and not to be forgotten:**
- ⚠️ **`ef` is NOT tuned.** recall@10 = 0.9990 at ef=64 on 2,200 real vectors —
  but it also passes at **ef=4**, which at that corpus size means the graph is
  nearly fully connected and the corpus is flattering the index. A ≥50k real-vector
  run is the honest gate.
- **The roster is unwired.** `nemotron-3-embed-8b` is on disk (15 GB) and
  referenced by zero code; NV-Embed-V2 / NV-ReRank-V2 / nemotron-3-rerank absent.
  ⚠️ *Settle openly:* the roster ask includes Harrier, but
  `TOTALRECALL_MASTER_PLAN.md:278` lists Harrier as **stale**, superseded by the
  07-08 Qwen3 single-lock.
- **The tier ladder (0.6/4/8B) is undecided** — candle's 0.6B reranker saturates
  (0.50, no lift). That gets decided by BRIGHT at scale (Phase 15), not by n=2.

<details><summary>original scope, for the record</summary>

Run the 28 `#[ignore]` gates with real weights on the GB10. Cross-engine agreement
matrix: {candle, llama.cpp, vLLM} × {embed, rerank} × {0.6b, 4b, 8b}. Wire the
roster — `nemotron-3-embed-8b` is **on disk, 15 GB, referenced by zero code**;
NV-Embed-V2 / NV-ReRank-V2 / nemotron-3-rerank are absent.
⚠️ *Settle openly, don't assume:* the roster ask includes Harrier, but
`TOTALRECALL_MASTER_PLAN.md:278` lists Harrier as **stale**, superseded by the
07-08 Qwen3 single-lock.
**Gate:** matrix green and recorded. No accuracy claim ships before this.
</details>

### ~~4 — Filter-aware HNSW~~ ✅ DONE (2026-07-17) *(a correctness bug, and Qdrant's differentiator)*
Cardinality estimation from the existing payload-index stats → predicate applied
during the beam search → three-way strategy choice; `FILTER_OVERFETCH` becomes a
fallback rather than the plan.
**DONE:** three strategies chosen by exact cardinality — exact scoping (≤65,536
matches), filter-aware graph traversal (>65,536), post-filter (unindexed). Bug
reproduced (200k docs, 2.5% filter → 1 result, recall@10=0.10) then fixed to
recall@10 ≥ 0.9. `AnnIndex::search_filtered` admits only allowed nodes to the beam.
Scale tests #[ignore]d (~90s debug), ann-unit test carries the mechanism in CI.

### 5 — The storage layer, done once *(in progress)*
- **5a RocksDB workload fit** ✅ DONE (2026-07-17, PR #10): memtable ceiling +
  `CF_TERMS` prefix bloom + `CF_VECS` BlobDB. Correctness verified; perf within
  noise on a warm synthetic bench (benefit is at-scale, deferred to Phase 15).
- **5b transactions** ✅ core DONE (2026-07-17): multi-op atomic
  `ConnXRecall::transaction(Vec<WriteOp>)` over upsert/remove. `txn::Transaction`
  threads the estate's read-modify-write counters through every statement (the
  hazard: two upserts each re-reading the pre-commit `doc_count` would net +1,
  not +2) and defers the out-of-band graph ops to commit, so rollback needs no
  2PC — a dropped-before-commit transaction leaves the graph untouched because
  nothing durable landed. Every single write is now an implicit one-statement
  transaction on the same path. *Gate met:* `a_failed_transaction_changes_nothing`
  proves rollback leaves count + ids + payload index byte-identical.
  **Follow-on (5b-2):** RRQL `BEGIN`/`COMMIT`/`CANCEL` syntax and `_into` forms
  for the remaining write ops (relate, payload-patch, define) so an arbitrary
  statement sequence is transactional, not just upsert/remove.
- **5c the `Db` seam** — the rocksdb+mem backend `PARITY.md` claims but which was
  never authored (also makes tests hermetic). *Gate:* suite passes on both backends.

### 6 — Scale past RAM
Immutable segments + background optimizer (mirrors the two-phase design already in
place — the architecture is right, the persistence isn't) → mmap vectors + on-disk
graph → PQ/BQ alongside SQ8.
**Gate:** 10M vectors, restart < 5 s, recall@10 ≥ 0.95, RSS well under dataset size.

### 7 — RRD-routed adaptive fusion *(the intelligence)*
Per-query strategy: `Rrf{k,weights}` | `Dbsf` | `Linear{alpha}`. RRD's centroid
router already exists at zero marginal cost — this is wiring, not invention.
**Train/dev/test split mandatory**; publish the sensitivity curve, never a picked
winner (a weight read off Finding 1's table is fit to the eval set).
**Gate:** on a mixed corpus (natural-language + identifier-heavy), routed fusion
beats dense-only **and** every fixed strategy, **on held-out queries**. If it
doesn't, that is the finding and it ships as one.

### 8 — TOON: the recall→LLM encoder
Zero in tree today. **Gate:** measured token reduction vs JSON at equal answer quality.

### 9 — Training *("matryoshka and quantization layers")*
**Recover the Burn tree from `qortex-rro-archive.bundle` before re-authoring 2,655
proven lines.** MRL training (inference truncation already built) ·
quantization-aware training — distinct from the built SQ8 *storage* quantization,
do not conflate · Qwen tuning needs no new config: a checkpoint slots in as a
weights dir.
**Gate:** a tuned checkpoint beats stock Qwen3-0.6B on a held-out set.

### 10 — Data model parity
Namespaces/databases · schemafull `DEFINE` + `ALTER`/`REMOVE` · **wire
`Statement::Live` → the existing `handle_stream`** (parser done; only execution
refuses).
**Gates:** cross-namespace leak test; a schemafull violation is rejected; LIVE delivers.

### 11 — Interfaces: REST → WS → GraphQL
New crate `rro-http` (**COSTAR in the PR** — the only new crate in this plan).
Mirror `ops.rs::serve_ops`: hand-rolled, zero-dep HTTP/1.1 (`openai.rs:9` — "RRO
has no reqwest/hyper/axum anywhere and this does not add one").
**Gate:** REST ≡ WS ≡ GraphQL ≡ `EstateQuery`.

### 12 — Auth
Users/roles + per-verb RBAC → JWT + JWKS → record-level permissions.
**Gates:** role allow/deny per verb; expired/wrong-issuer rejected; a scoped user
cannot read out of scope.

### 13 — Cluster (3-node GB10)
Replicated changefeed (seq-ordered + resumable = the replication log) → read
replicas → shard + scatter-gather → raft (`openraft`, don't hand-roll) →
`/cluster/*` in health.
**Gate:** kill the leader under write load; no acked write lost.

### 14 — The clyffy pull-in
Mirror the registered `trecall` pattern: submodule `deps/rro` → root `exclude` →
`clyffy-storage` feature → `adapters/rro/`.
**Bind to `ContextProvider`, not `VectorStore`** — a 3-method port would strand
RRO's gating, fusion and rerank, which master-plan §3 forbids. `GraphStore::recall`
is **dead** (zero implementors) — the previous plan's binding target did not exist.
RRO replaces `Funnel`; `funnel.rs` demotes to orchestration. Mirror **both**
dispatch levels (`resolve.rs::Vector::connect` *and* `RecallService::from_config`,
plus `clyffy-brain/src/main.rs:168`). **Probe the embedder for `dim`** — the
hardcoded `1024` silently corrupted non-1024-d writes.
**Gate:** clyffy boots on `[storage.connectome] backend = "rro"`, `think()`
recalls, W1 write→restart→survive.

### 15 — Proof
BRIGHT (published SOTA only ~22.1 nDCG@10 — start with `pony`, 7.9k docs/112
queries) + nfcorpus + a TREC/BEIR set · full ablation ladder with per-stage
latency, **including where RRO loses** · every engine × tier on the cluster ·
**bootstrap CIs + paired significance tests** (the current 323-query single run has
none — those deltas are directional only) · session replays.
**Gate:** `BENCHMARKS_REAL.md` is the SSOT; every pre-real number stays superseded.

---

## Order

**3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15.**

3 first because every accuracy claim depends on it and it is currently unproven.
4 next because it is a **correctness** bug, not a feature. 5 and 6 are each
done-once (they touch the same open path and the same ceiling). 7 needs real
models (3) to be measurable. 10–13 are the parity bulk. 14 is the payoff. 15 last —
replays of a half-built system are worth nothing.

## Working rules

- **`main` stays truly functional.** Every phase: `claude/phase-N-*` → PR → CI
  green → merge. Nothing reaches `main` that isn't built, tested and real.
- Every phase ends: 0 warnings · `cargo test --workspace` green · **CI green on the
  PR** · gate **measured** and recorded · merged. The operator confirms
  functionality; success is never self-declared.
- No deprecation without review. Dead code = suspect for incomplete work → finish
  it, or delete only after review says *replaced*. **Keep `deploy/rrf.service`.**
- `RRF` = **Reciprocal Rank Fusion**. Never blanket `s/rrf/rro/`.
- COSTAR (`clyffy/docs/PLANNING_DISCIPLINE.md`) for any new crate. Only Phase 11
  proposes one.
