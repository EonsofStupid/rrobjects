# MODELS.md — the real embedder/reranker: architecture + exact build plan

_The definitive spec for wiring **real** models (Qwen embedder, Nemotron-class
reranker) into RRO. Written so a session on the target box executes it verbatim.
This is the thing that was asked for first and is not yet done: every accuracy
number in the repo today was produced by the deterministic (synthetic) embedder
and is UNVERIFIED for real retrieval until this lands._

## 0. Prerequisites on the target box (this container cannot meet them)

- **Network:** `huggingface.co` reachable (weights), OR weights pre-staged on disk.
  This container 403-blocks HF by egress policy — that is why this is a box task.
- **Disk:** ≥ 15 GB free (Qwen3-Embedding-0.6B ≈ 1.2 GB safetensors; a
  Nemotron/cross-encoder reranker 0.3–4 GB; candle build artifacts several GB).
- **Compute:** CPU works; CUDA or Metal strongly preferred for the reranker.
- **Toolchain:** rust stable, clang/libclang (already used by rocksdb).

## 1. The design goal (non-negotiable)

**Modular AND part of the engine. Truly swappable, max performance, expandable.**
Achieved by three rules:

1. **The trait is the only contract.** `Embedder` / `Reranker` /`Classifier`
   already exist in `rro-core/src/traits.rs`. Nothing in the flow, estate, or
   query plane knows which backend is behind them. Real models drop in here.
2. **Selection is data, not code.** A backend **registry** builds the concrete
   impl from config (`RRO_EMBEDDER=candle-qwen|onnx|remote|deterministic`). Adding
   a new backend = add one enum arm + one constructor; zero flow changes.
3. **Performance lives inside the backend, behind the trait.** Batching, graph
   warmup, mmap weights, device placement, quantization — all internal to the
   candle/onnx impl. The trait stays a clean `embed(&[String]) -> Vec<Embedding>`.

## 2. The backend registry (author this first — new crate `model-registry`)

A tiny crate that turns config into a boxed trait object. Everything else depends
on the trait, never on candle/ort directly, so heavy deps stay optional and the
default workspace build stays weightless.

```
crates/model-registry/
  Cargo.toml         # features: candle (embedder+reranker via candle), onnx (via ort)
  src/lib.rs
```

```rust
// The selection contract — pure data, parsed from env/config.
pub struct EmbedderConfig {
    pub kind: EmbedderKind,      // Deterministic | CandleQwen | Onnx | Remote
    pub weights_path: Option<String>,
    pub dim: Option<usize>,
    pub device: Device,          // Cpu | Cuda(usize) | Metal
    pub batch: usize,
}

pub fn build_embedder(cfg: &EmbedderConfig) -> Result<Arc<dyn Embedder>> {
    match cfg.kind {
        EmbedderKind::Deterministic => Ok(Arc::new(DeterministicEmbedder::new(cfg.dim()))),
        #[cfg(feature = "candle")]
        EmbedderKind::CandleQwen    => Ok(Arc::new(CandleQwenEmbedder::load(cfg)?)),
        #[cfg(feature = "onnx")]
        EmbedderKind::Onnx          => Ok(Arc::new(OnnxEmbedder::load(cfg)?)),
        EmbedderKind::Remote        => Ok(Arc::new(RemoteEmbedder::new(cfg))),
        // arms compiled out without their feature return a clear config error
    }
}
// Same shape: build_reranker(&RerankerConfig) -> Arc<dyn Reranker>.
```

`rro-engine`'s daemon (`bin/rrf.rs`) reads `RRO_EMBEDDER*` / `RRO_RERANKER*` from
env, calls `build_embedder` / `build_reranker`, and hands the results to
`ReasonReadyObject::builder()`. **That is the entire swap mechanism.**

## 3. The Qwen embedder (candle) — exact steps

Target: `Qwen/Qwen3-Embedding-0.6B` (1024-dim; safetensors + tokenizer.json).

1. **Deps** (behind `candle` feature): `candle-core`, `candle-nn`,
   `candle-transformers`, `tokenizers`, `safetensors`, `hf-hub` (or read local
   paths). Pin versions; commit `Cargo.lock`.
2. **Weights**: `hf-hub` snapshot download to `RRO_EMBEDDER_WEIGHTS`, OR accept a
   local dir. Files: `model.safetensors`, `tokenizer.json`, `config.json`,
   `1_Pooling/config.json` (pooling mode), `config_sentence_transformers.json`.
3. **Load** (`CandleQwenEmbedder::load`): mmap safetensors via
   `candle_core::safetensors::MmapedSafetensors`; build the Qwen2/Qwen3 model
   from `candle_transformers::models::qwen2` (or the matching module); load the
   `tokenizers::Tokenizer`; place on `cfg.device`; **warm the graph** with one
   dummy forward so first real query isn't cold.
4. **Forward** (`embed`): tokenize batch (pad to max len, attention mask) →
   forward → **pooling per the model's `1_Pooling` config** (Qwen3-Embedding
   uses last-token / mean per its card — read it, don't guess) → **L2-normalize**
   (the estate's cosine path assumes it) → collect `Embedding`.
5. **Performance:** honor `RRO_EMBED_BATCH`; reuse the tokenizer; keep tensors on
   device; f16/bf16 on GPU; expose `dim()` from config. No allocation per token
   in the hot loop.

### GATE (this is the whole point)
- **Sanity:** cosine("king","queen") > cosine("king","banana"); paraphrases
  score high, unrelated low. Assert in an ignored-by-default test that runs with
  `--features candle` and weights present.
- **Re-run the bake-off** (`rro-bench`) with the real embedder on both RRO and
  the baseline, identical corpus. Record HONEST numbers in BENCHMARKS.md — they
  may differ hugely from the synthetic 1.000. **This is the first real answer to
  "is the engine worth it."**
- **Re-record baselines** and **re-tune ANN `ef`/graph params** on real 1024-dim
  Qwen vectors (the current params were fit to synthetic distributions).

## 4. The Nemotron reranker (candle) — exact steps

Target: a Nemotron-class / cross-encoder reranker (query,doc)→relevance score.

1. Same dep/weights pattern behind the `candle` feature on `crates/reranker`.
2. `CandleNemotronReranker::rerank(query, candidates)`: for each candidate, form
   the cross-encoder input `(query, doc.text)`, tokenize, forward, take the
   relevance logit; sort descending; return re-scored candidates. Batch the
   forward passes (`RRO_EMBED_BATCH`).
3. Wire it as the flow's reranker via the registry when `RRO_RERANKER=candle-nemotron`.

### GATE
- On a labeled set (or the planted-golden bench), the reranker must **lift**
  top-k relevance vs no-rerank (measure NDCG@10 or golden@k before/after).
  Record in BENCHMARKS.md. If it doesn't lift, that's a real finding — report it.

## 5. Expandability (why this shape scales)

- **New backend** (e.g. a local llama.cpp server, a different HF model, an ONNX
  export): add an `EmbedderKind` arm + constructor. Nothing else changes.
- **ONNX path** (`onnx` feature, `ort` crate): for models shipping ONNX; same
  trait, same registry, same gates. Good for CPU-only boxes.
- **Remote path**: `RemoteEmbedder` calls an external endpoint over the existing
  a2a client — lets a thin node borrow a GPU node's model. Already-built wire.
- **Per-collection models** (future): the estate already has named vector spaces
  (sprint 12) — a later step can route different collections to different
  embedders, all behind the same registry.

## 6. Definition of done
- `RRO_EMBEDDER=candle-qwen` boots a node that embeds real text; `deterministic`
  still works for CI. Swapping is one env var. ✅ modular + part of engine.
- Bake-off re-run on real embeddings; numbers recorded honestly; ANN re-tuned;
  reranker lift measured. ✅ the engine is finally *evaluated*.
- Docs (COMPARISON/BENCHMARKS/README) updated to mark all pre-real numbers as
  superseded. ✅ no synthetic claim left standing as if it were real.
