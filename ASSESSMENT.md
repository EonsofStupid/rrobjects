# Honest assessment — is this useful, or start over?

> **Superseded in part — 2026-07-16.** This document was written before P7 landed
> and now **UNDER-claims**. Corrections:
> - *"the real embedder/reranker was never built… a stub"* — **no longer true.**
>   `crates/embedder/src/candle_qwen/` is a real, hand-written, cache-free Qwen3
>   encoder that reproduces the model card's published scores to ~6 decimals, and
>   agrees with vLLM **elementwise (cosine 0.9999)** on identical weights.
>   `crates/reranker/` has candle + llama.cpp + vLLM cross-encoders.
> - *"this container 403-blocks HF… it must move"* — done; the work is on the box.
> - *"mean-pool"* (§"The actual next step") is **WRONG for this model**:
>   `1_Pooling/config.json` says `pooling_mode_lasttoken: true`. Mean-pooling would
>   silently degrade every vector. See `crates/embedder/src/candle_qwen/mod.rs`.
>
> What still stands, and is the reason this doc was right to be blunt: every
> accuracy number predating `docs/BENCHMARKS_REAL.md` was synthetic.


_Authored 2026-07-16. Blunt, not a pitch._

## The one thing you asked for first was never done

You asked for a **real embedder (Qwen) and real reranker (Nemotron)**. That was
never built. `crates/embedder/src/devpulse.rs` is a **stub**: it has
`ModelSpec::qwen()`, a `candle` feature gate, and a `TODO(devpulse): load the
Qwen backbone` that returns an error when called. Every "measured accuracy"
number in BENCHMARKS.md (1.000 accuracy@10, the bake-off, etc.) was produced by
the **synthetic deterministic embedder** in `deterministic.rs` — hashed
pseudo-vectors, not semantics. So on the question you actually care about — *does
this engine retrieve well on real text* — **you still have no answer.** That is
the real failure and it is on the build, not on you.

Why it's still not done here: this container's egress policy **403-blocks
huggingface.co** (confirmed in the proxy denial log), and has ~4.9 GB free disk.
Qwen + Nemotron weights (~2.5 GB+) plus a candle/ONNX build do not fit and can't
be fetched. **This environment cannot do it. A different one can.**

## Do you start over? No. Keep the engine.

~14,400 lines of real code, 30 test files, all gated. What's genuinely built and
worth keeping (none of this is fake — it's storage/index/query machinery that
works regardless of what produces the vectors):

- **connxism (4,700 loc)** — the RocksDB estate: hybrid dense+BM25, payload
  secondary indexes (incl. datetime/uuid/geo Z-order), sparse postings,
  named-vector spaces, collections/aliases, changefeed, quotas. Real, tested.
- **recall (970 loc)** — the ANN graph (recall@10 ≥ 0.95 gated vs exact).
  Its *quality* was only ever checked on synthetic vectors, but the graph
  algorithm itself is real.
- **The trait seam is the payoff** — `rro-core/src/traits.rs`: `Embedder`,
  `Reranker`, `Classifier`, `Recall`. Real Qwen/Nemotron drop in **behind these
  traits** without touching the flow, the estate, or the query plane. That is
  exactly why you don't restart: the socket for the real models is already there.

## What is suspect / must be re-earned once real models land

- Every accuracy/recall number in the docs — **re-run on real embeddings**, they
  may look completely different. Treat them as unverified until then.
- The ANN `ef`/graph params were tuned against synthetic vector distributions;
  may need retuning on real 1024-dim Qwen vectors.

## The actual next step (in an environment that can reach HF + has disk/GPU)

1. `cargo add candle-core candle-transformers tokenizers` behind the `candle`
   feature; fill the `TODO` in `devpulse.rs` — load Qwen3-Embedding, run the
   forward pass, mean-pool. (Nemotron reranker the same, behind `Reranker`.)
2. Swap the default embedder for it in `rro-engine`.
3. Re-run the bake-off. **Now** the accuracy number means something.

That is roughly one focused session — **in the right box**, not this one.

## Bottom line

The engine is real and reusable; **do not rebuild it.** The models — the thing
you asked for — are a stub, and this container physically cannot finish them.
Move to an environment with huggingface.co reachable and real disk/GPU, and the
Qwen+Nemotron wiring is a small, well-bounded job because the traits already
exist. Nothing here is thrown away except the *synthetic accuracy claims*, which
were never yours to trust in the first place.
