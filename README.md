# Reason Ready Objects

**RRO — Reason Ready Objects. Not just a RAG engine. Intelligence.**
**A turnkey solution granting your AI persistence.**

One embedded, tokio-native retrieval-and-reasoning engine, clean-authored
from a single root: no external database, no vector-store dependency, no
model gateway, no wrappers. It gates and classifies at first touch, retrieves
hybrid, reranks, judges whether what it found is enough to reason on, shows
its work as a graph — and treats remote nodes as local over its own layer-2
protocol.

> **Why RRO and not RRF.** "RRF" is **Reciprocal Rank Fusion** — a standard IR
> term, and the exact thing this engine's own `hybrid_search` does. The product
> name collided with its own algorithm, which is why it was unfindable. After
> the rename every remaining `RRF` in this tree means the *algorithm* and
> nothing else. The objects are the product: **Reason Ready Objects** are the
> typed, grammar-conformant artifacts RRD produces at first touch.

```
        R R D  ─────────────  the instant first thing
   stamp · gates · shape · intent · baseline prediction     (µs, pre-model)
        │
embedder ─▶ recall ─▶ reranker ─▶ classifier ─▶ connectome
(perceive)  (hybrid:   (true      (reason-      (the map an
            ANN+BM25,   relevance)  ready?)      operator reads)
            RRF-fused)
        │
   connXism estate: RocksDB — docs, vectors, postings, relations,
   tags, shapes, trends, changefeed · a2a warp mesh · DuckDB events
```

## Measured

> ⚠️ **The table below is SYNTHETIC and superseded.** Every accuracy figure in it
> came from the deterministic hash embedder — synthetic vectors scored against
> synthetic vectors — and the throughput figures were measured with that same
> microsecond embedder, so they describe the estate's indexing speed, not an
> engine running a real model. Real numbers (Qwen3-Embedding-4B on nfcorpus,
> BM25 baseline calibrated to published BEIR): **[docs/BENCHMARKS_REAL.md](docs/BENCHMARKS_REAL.md)**.
> Headline deltas there: dense 0.4119 nDCG@10, hybrid fusion −5.3% vs dense,
> reranker +9.9% at 27x latency, ingest 10 docs/sec (not 10.9k).

### Pre-real (synthetic) figures, kept for history — see docs/BENCHMARKS.md

| | result |
|---|---|
| Durable ingest (embed + RRD + index + provenance) | **10.9k docs/sec** local · **13.6k** observed over a2a · ~60k raw (pre-built vectors) |
| Query p50 @ 100k docs, full pipeline | **1.88 ms** (532 qps) |
| Dense-only ANN p50 @ 50k | **0.32 ms** |
| Planted-retrieval accuracy@10 | **1.000** |
| Max-score lexical pruning (selective+common vs all-common) | **74×** (0.85 ms vs 63.2 ms), exactness-gated |
| `watch` push-frame delivery (write commit → frame on the wire) | **0.28 ms p50** |
| vs popular RAG baseline, identical inputs | ~19× ingest, ~3× faster queries, 1.000 vs 0.606 accuracy |
| Route→recall on ambiguous corpora | **1.000 vs 0.025** flat |
| a2a wire cost (remote ≡ local) | **+3 ms**, identical accuracy |

## The query plane (all typed, all over the wire, all gated)

One `EstateQuery` speaks every capability — locally, over a2a TCP, and
through MCP: hybrid dense+BM25 fusion · **weighted sparse vectors**
(three-way RRF fusion) · **named vector spaces** per point (independent
dims) · **late interaction / MaxSim** token-vector rescoring · filter DSL
(eq / any / range / **date ranges** by instant / **geo radius + box** on
Z-order keys / exists; must / should / must_not) answered **index-first**
from typed payload indexes (keyword, numeric, bool, datetime, uuid, geo)
· **named collections** with leak-proof scoping + atomic **aliases** ·
groups, recommend, discover, batch · pagination (`offset`), score
threshold, lean payloads, `with_vectors` · similarity matrix ·
deterministic sampling. Text goes through a configurable **analyzer**
(tokenizers, stopwords, Porter stemmer — persisted per estate) with an
offset-exact **highlighter**. Writes stream out as **push** changefeed
frames (`watch`, event-driven, seq-resumable) beside poll paging; per-point
**payload CRUD** keeps every index exactly consistent. Ops: `health` verb,
prometheus **/metrics** + probes, self-reported issues. SQ8 quantization
(exact rescore) when memory matters.

## Turnkey

```sh
./scripts/quickstart.sh          # build → boot (estate+RRD+a2a+events) → smoke over the wire
./scripts/mesh.sh 3              # a local mesh: 3 engines, each a2a-addressable
./scripts/quickstart.sh stop

# Real models (Qwen3 embedder + reranker), fully turnkey:
./scripts/fetch-models.sh        # baseline 0.6b weights (verified byte-exact)
./scripts/fetch-models.sh --list # the catalog: 0.6b (baseline) / 4b / 8b
RRO_REAL=1 ./scripts/quickstart.sh                        # baseline on CPU
RRO_REAL=1 RRO_EMBED_SIZE=4b RRO_DEVICE=cuda:0 ./scripts/quickstart.sh   # scale up
```

The default build is **weightless** (synthetic embedder, dev/CI only). Real
weights are too large to vendor in git, so `scripts/fetch-models.sh` pulls the
Qwen3 family on demand (0.6B baseline → 4B → 8B, embedder + reranker) and
verifies each shard byte-exact; `RRO_REAL=1` wires them into the daemon in one
command. The 0.6B pair is the CPU-runnable baseline and the fine-tuning base;
fine-tuned checkpoints slot in as just-another-weights-dir. Details:
**[docs/MODELS.md](docs/MODELS.md)**.

Deploy: **Podman Quadlets** — `deploy/Containerfile` (build),
`deploy/rro.container` + `deploy/rro-estate.volume` (rootless systemd units),
`deploy/rro-mesh.pod` (cluster node group), `deploy/config.env.example`. The
daemon handles the full signal set, drains cleanly, and commits its RRD baseline
on shutdown so the next boot predicts warm. Install (rootless):
`install -m644 deploy/rro.container ~/.config/containers/systemd/ && systemctl --user daemon-reload && systemctl --user start rro`.

**Model backends** (real Qwen embedder / Nemotron reranker) plug in behind the
`Embedder`/`Reranker` traits, selected by `RRO_EMBEDDER`/`RRO_RERANKER` — the
default build is weightless (synthetic embedder, dev/CI only). Wiring real models
is spec'd exactly in **docs/MODELS.md**; the full remaining plan (models, RRQL,
cluster, deploy) is in **docs/ROADMAP_REAL.md**.

## The workspace

| Crate | Role |
|---|---|
| `rro-core` | The contract: types, traits, events, kernels |
| `rrd` | **The reason-ready JIT**: gate ladder, sliver lattice, plans, RROs, semantic intent router, evolving shape baseline (predict / drift / persist) |
| `embedder` | Perception — deterministic default + DevPULSE (Qwen) plug-point |
| `recall` | Vector memory — ANN graph (recall@10 ≥ 0.95 gated) + exact store |
| `reranker` | True relevance — BM25 default + DevPULSE (Nemotron) plug-point |
| `classifier` | The readiness daemon |
| `connectome` | The visual map (flow + estate), JSON/DOT |
| `connxism` | The kvs-connectome estate: RocksDB, hybrid recall, relations, changefeed, warp points |
| `connectors` | Resumable source drivers + the sync engine (RRD-first) |
| `rro-net` | a2a layer-2: in-proc bus + TCP; MCP mesh binding lands P5 |
| `rro-engine` | The orchestrator, `rro` daemon, `rro-bench` harness |

## Where everything stands

- **docs/COMPARISON.md** — head-to-head vs the reference engines: what only
  RRO has, their home turf measured, and the phased tail.
- **docs/PARITY.md** — the exhaustive capability union, row-by-row status.
- **docs/PLAN.md** / **docs/EXECUTION.md** — phases with gates; the
  plan→execute→verify loop with every sprint's evidence.
- **docs/BENCHMARKS.md** — every number, with the runs that produced it.
- **docs/adr/** — the decisions, including RRD (ADR-0002: gate ladder,
  sliver lattice, shape baseline).

DevPULSE model backends (Qwen embedder, Nemotron reranker, learned
classifier) drop in behind the existing traits — the flow does not change.

---
© 2026 EonsofStupid — Reason Ready. Proprietary; see `LICENSE`.
