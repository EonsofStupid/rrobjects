# Reason Ready

**RRF — Reason Ready Flow. Not just RAG. Intelligence.**

One embedded, tokio-native retrieval-and-reasoning engine that stands entirely
on its own: no external database, no vector-store dependency, no model gateway,
no wrappers. Clean-authored from a single root. It retrieves — then it judges
whether what it found is enough to reason on, and shows you how it got there.
That readiness gate and that visible map are why it is *not just RAG*.

> Recall finds. Reranker orders. Classifier judges readiness. Embedder
> perceives. Connectome shows. One faithful engine, one flow.

## Why

Reason Ready is a bet: that a small, frictionless, fully-owned engine — every
layer authored and tunable end to end — is still the best way to do retrieval.
No HTTP hop between components, no lineage to sanitize, nothing to wrap. It
embeds in-process (tokio, signal-driven) and keeps full node / networking /
**a2a** (agent-to-agent) capability — you lose nothing by embedding it.

## The flow

```
        embedder            recall            reranker         classifier        connectome
text ──▶ perceive ──▶ vector memory ──▶ true-relevance ──▶ reason-ready? ──▶ visual map
         (Qwen)         (Recall)          (Nemotron)        (daemon)         (the UI graph)
           │               │                  │                 │                  │
           └───────────────┴──── DevPULSE models ──┘        readiness         non-technical
                       (trained & tuned in-house)           gate              legibility
```

- **`recall`** — dense vector memory. The retrieval core.
- **`connectome`** — the visual/relational map. Renders how memories and
  reasoning connect so a non-technical viewer can *see* the recall happen. This
  is the engine's sensory surface for the UI.
- **`classifier`** — the *Reason Ready daemon*. Judges whether retrieved context
  is sufficient to reason on, and gates the flow.
- **`reranker`** — true-relevance ordering over recall candidates.
- **`embedder`** — perception: text → dense vectors.
- **`rrf-flow`** — the orchestrator that wires the components into one pass and
  runs the embedded, signal-driven runtime.
- **`rrf-net`** — the a2a / node networking surface. Embedded does not mean
  isolated.

## DevPULSE models

The embedder and reranker are model-backed and swappable behind traits:

- **Embedder** — Qwen-family embedding backbone → tuned into the **DevPULSE
  embedder**.
- **Reranker** — Nemotron-family reranker backbone → tuned into the **DevPULSE
  reranker**.

The workspace ships working default implementations (deterministic embedder,
lexical reranker, heuristic classifier) so the whole flow compiles and runs
**today, with zero weights**. Drop the tuned DevPULSE weights in behind the
`candle` feature as they land — the flow does not change.

## Workspace

| Crate         | Role                                                   |
|---------------|--------------------------------------------------------|
| `rrf-core`    | Shared domain types + the engine traits (the contract) |
| `embedder`    | `Embedder` — DevPULSE (Qwen) + deterministic default   |
| `recall`      | `Recall` — in-memory vector store (cosine), pluggable  |
| `reranker`    | `Reranker` — DevPULSE (Nemotron) + BM25 default        |
| `classifier`  | `Classifier` — the Reason Ready daemon                 |
| `connectome`  | The visual map: graph model + JSON/DOT render          |
| `connxism`    | The kvs-connectome: persistent estate (RocksDB) — nodes, warp points, connectors, hybrid vector+BM25 recall, tags, shapes, trends |
| `rrf-net`     | a2a / node networking surface                          |
| `rrf-flow`    | Orchestrator + ingestion machine + `rrf` daemon + `rrf-bench` |

## Quick start

```sh
# Run the end-to-end demo (index a corpus, ask, see the readiness gate + map).
cargo run --example demo -p rrf-flow

# Run the embedded daemon (tokio, ctrl-c / SIGTERM aware).
cargo run --bin rrf
# …persistently, over an estate:
RRF_ESTATE=./my-estate cargo run --bin rrf

# Measure it (ingest throughput + hybrid query latency, real numbers):
cargo run --release --bin rrf-bench -- --docs 50000 --queries 500 --store estate
```

## Status

Pre-release. The architecture and the end-to-end flow are real, running, and
**measured** (see `docs/BENCHMARKS.md`); the estate (RocksDB), hybrid recall,
ingestion machine, event stream, and baseline gates are live. The full
capability plan — ANN, graph relations, live queries, gRPC, the WASM plugin
runtime, DevPULSE models — is phased with proof gates in **`docs/PLAN.md`**.

---
© 2026 EonsofStupid — Reason Ready. Proprietary; see `LICENSE`.
