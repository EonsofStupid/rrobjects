# ROADMAP_REAL.md — everything left, fully mapped, in priority order

_The complete remaining plan from `258b2c1`, authored so a fresh session on the
target box executes without re-deriving anything. Each item: what it is, where it
plugs in, its verification gate (repo law: nothing ships as done until its gate
runs). Ordered by dependency + value. Supersedes the phase status in the older
ROADMAP.md for execution purposes._

## Reality check (read before anything)
- The **engine** (estate, ANN, query plane, a2a wire, ops) is real, gated,
  reusable — ~14,400 loc, 30 test files, green. DO NOT rebuild it.
- The **models** were never wired — every accuracy number is on synthetic
  vectors and is UNVERIFIED. See `docs/MODELS.md`. **Priority #1.**
- **Deploy** is now Podman Quadlets (Docker removed). `deploy/` + §P9.

---

## P7 — REAL MODELS (FIRST; the actual deliverable)

> **P7 STATUS — 2026-07-16: LANDED.** This section's framing ("the models were
> never wired… Priority #1") is now stale. Done: `model-registry` (P7.1);
> the candle Qwen3 encoder (P7.2, matches the model card to ~6dp); llama.cpp +
> vLLM backends (P7.3, candle==vLLM elementwise 0.9999 on the same weights);
> cross-encoder rerankers on all three (P7.4, BM25 0.50 → 1.00 golden@1); the
> first honest benchmark (P7.3/P7.6 → `docs/BENCHMARKS_REAL.md`, BM25 0.3115 vs
> published BEIR ~0.325 — the harness's calibration anchor).
> Still open from this section: **re-tune ANN `ef` on real vectors** (the
> recall@10 ≥ 0.95 gate at `recall/src/ann.rs:533` was measured on SYNTHETIC
> vectors) and the tier ladder / BRIGHT.
Full spec: **docs/MODELS.md**. Ordered work:
1. **`model-registry` crate**: config→boxed-trait selection (`EmbedderKind`,
   `RerankerConfig`), env-driven, features `candle`/`onnx`. Gate: `deterministic`
   still builds weightless; unknown kind → clear error.
2. **Candle Qwen embedder** (feature `candle`): mmap safetensors, tokenizer,
   pooling-per-card, L2-normalize, batched, warm graph. Gate: semantic sanity
   (king~queen > king~banana) + loads real weights.
3. **Re-run bake-off on real embeddings**; honest numbers; **re-tune ANN
   `ef`/graph params** for real 1024-dim vectors; re-record baselines. Gate:
   bake-off table in BENCHMARKS.md marked as superseding all synthetic numbers.
4. **Candle Nemotron reranker**: cross-encoder scoring, batched. Gate: measured
   top-k lift (NDCG@10 / golden@k) vs no-rerank.
5. **Wire both into the daemon** via the registry + `RRO_EMBEDDER`/`RRO_RERANKER`.
   Gate: `RRO_EMBEDDER=candle-qwen` boots + answers real queries over a2a.
6. **Docs sweep**: COMPARISON/README/BENCHMARKS mark pre-real numbers superseded.

---

## P6 — RRQL text DSL + DEFINE/CRUD + GraphQL (in-tree, any environment)
Text query language parsed onto the proven typed machinery. Zero new deps
(hand-rolled parser).
1. Lexer + expression grammar + `SELECT…WHERE` → `Filter`/`EstateQuery`. Gate:
   parsed ≡ hand-built typed query (property test over random ASTs).
2. DEFINE/CREATE/INSERT/UPDATE/UPSERT/DELETE → estate ops. Gate: each ≡ typed API.
3. RELATE/traversal (`->verb->`) → relate/traverse; LIVE/KILL → `watch`;
   INFO/SHOW CHANGES → info/feed. Gate: live delivers; INFO matches.
4. GraphQL schema over collections+fields, same executors. Gate: GraphQL ≡ EstateQuery.
5. `rro_sql` MCP tool + client method. Gate: wire RRQL ≡ local.
Est. 3–5 sprints.

---

## P8 — CLUSTER: replication, sharding, distribution (big, multi-phase)
Substrate: a2a layer (`rro-net`, warp points, tokens) + `deploy/rro-mesh.pod`.
- **P8.1 Replicated changefeed**: follower subscribes to leader `watch`, applies
  changes; feed is seq-ordered+resumable (sprint 4/13) = the replication log.
  Gate: kill follower mid-stream, restart, resume from seq, converge byte-exact.
- **P8.2 Read replicas + routing**: N followers serve reads, writes to leader.
  Gate: any replica's answer ≡ leader's (post-convergence).
- **P8.3 Sharding by key**: partition id space; scatter-gather + merge top-k;
  collections are natural shard bounds. Gate: sharded top-k ≡ single-node top-k.
- **P8.4 Consensus (raft-class)**: leader election + quorum log repl; prefer
  `openraft` over hand-rolling. Gate: kill leader under write load; new leader,
  no acked write lost, ring converges (chaos test).
- **P8.5 Cluster ops**: `/cluster/*` (members, shard map, replica lag) in
  health/info; raft port in the pod; per-node `.container` joining the pod. Gate:
  topology reported over the wire.
Est. 6–10 sprints. AFTER P7, ideally after P6.

---

## P9 — DEPLOY (Podman Quadlets — Docker removed; finish on the box)
In-tree now: `deploy/Containerfile`, `rro.container`, `rro-estate.volume`,
`rro-mesh.pod`, updated `config.env.example`. On a Podman box:
1. `podman build -f deploy/Containerfile -t localhost/rrf:latest .`.
2. Install Quadlets; `systemctl --user daemon-reload`; verify with
   `podman-system-generator --dryrun`.
3. `systemctl --user start rrf`; `/healthz` on :9090 + `ping` on :7878. Gate:
   green health + pong. Then a candle-image variant with weights mounted.
4. Point `scripts/quickstart.sh` / `mesh.sh` at podman. Gate: quickstart smokes.

---

## P-tail — smaller PARITY rows (any environment)
- PQ/binary quantization (SQ8 exists). Gate: recall@10 vs memory measured.
- Transactions beyond atomic WriteBatch (RocksDB TransactionDB). Gate: rollback.
- RBAC/JWT beyond capability tokens. Gate: role allow/deny per verb.
- Multilingual analyzers/stemmers. Gate: language-specific retrieval.
- Geo polygon filters. Gate: point-in-polygon ≡ brute force.
- WASM plugin runtime (+ security review). Gate: sandboxed plugin can't escape.
- HTTP REST / WebSocket RPC / OTLP — alternate transports, same executors. Gate: REST ≡ EstateQuery.

---

## Order of operations for the box
**P7 (real models) → re-measure everything → P6 (RRQL) → P9 finish (podman) →
P8 (cluster) → P-tail.** Do NOT let anything push P7 down again — it is what tells
you whether the engine is worth continuing at all.
