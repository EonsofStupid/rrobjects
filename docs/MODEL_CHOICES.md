# MODEL_CHOICES.md — which model, which runtime, and why

Working notes for picking the real embedder + reranker. Two independent axes:
**which model** (Qwen3 sizes / Nemotron) and **which runtime** (candle / llama.cpp
/ vLLM). They are orthogonal because everything sits behind the `Embedder` /
`Reranker` traits — the registry (`RRO_EMBEDDER` / `RRO_RERANKER`) swaps either
axis without touching the flow (docs/MODELS.md §2). So you can make first-touch
**real today** on one runtime and switch later for free.

## TL;DR
- **Start:** `candle-qwen` embedder + `candle-cross-encoder` reranker, both **0.6B**.
  In-process, no server to babysit, CPU-runnable, numerically gated against the
  model card. This is the fastest path to a real first-touch and the fine-tuning
  baseline.
- **Scale quality:** move to **4B** (GPU). Same trait, one env var + weights.
- **Scale throughput / serving:** switch the runtime to **vLLM** (GPU, batched
  server) or **llama.cpp** (GGUF, CPU/edge) — the HTTP backends are already wired.
- **Nemotron reranker:** viable **only via llama.cpp/vLLM**, never candle (see below).

---

## Axis 1 — runtime (already wired, all three)

| runtime | `RRO_EMBEDDER` / `RRO_RERANKER` | shape | best when |
|---|---|---|---|
| **candle** | `candle-qwen` / `candle-cross-encoder` | in-process, loads safetensors directly | dev, single-node, no ops surface, exact control; the default real path |
| **llama.cpp** | `llamacpp` | external server, GGUF, OpenAI-compatible HTTP | CPU / edge / quantized; run `llama-server --embedding` (embed) or `--reranking` (rerank) |
| **vLLM** | `vllm` | external server, GPU, high-throughput HTTP | GPU serving, batching, many QPS |

Endpoints default to localhost (`RRO_*_ENDPOINT` overrides): llama.cpp embed
`:8090/v1/embeddings`, vLLM embed `:8092/v1/embeddings`, llama.cpp rerank
`:8093/v1/rerank`, vLLM rerank `:8092/rerank`. The HTTP clients are hand-rolled
(no `reqwest`), so the weightless workspace stays dependency-light.

**Trade:** candle = zero moving parts but you own batching/placement in-proc.
HTTP backends = someone else's optimized server, at the cost of running it and a
network hop. Retrieval quality is identical for the same weights; this axis is an
**ops/throughput** decision, not a quality one.

---

## Axis 2 — model

### Embedder — Qwen3-Embedding (the only wired family)
| size | dim | approx | notes |
|---|---|---|---|
| **0.6B** | 1024 | 1.1 GB | baseline; CPU-fine; MRL-truncatable to 32..1024 (`RRO_EMBED_DIM`) |
| **4B** | 2560 | 7.5 GB | quality step; wants a GPU |
| **8B** | 4096 | 14 GB | ceiling; GPU only (f32 on CPU ≈ 32 GB RAM) |

All matryoshka-trained, last-token pooling, instruction-prefixed queries,
L2-normalized — the candle backend already reproduces the card's reference scores
(the `candle_qwen_gate` test is the proof). Bigger = better recall, linearly more
compute. Pick the smallest that clears your quality bar; the 0.6B is a genuinely
strong baseline for its size.

### Reranker — two different architectures, know which you have
1. **Qwen3-Reranker (0.6B / 4B / 8B) — `candle-cross-encoder`, candle-native.**
   It is an `AutoModelForCausalLM` that scores by asking a yes/no question and
   reading the "yes"/"no" logits at the last position (not a scalar head). This
   is what the candle reranker implements and gates. Same sizes/tradeoffs as the
   embedder.
2. **Nemotron-class rerankers (e.g. `llama-nemotron-rerank-1b-v2`) — HTTP only.**
   These are `AutoModelForSequenceClassification` with a scalar relevance head.
   `llama-nemotron-rerank` specifically is `llama_bidirec` **custom_code and
   cannot be loaded by candle at all** — so if you want Nemotron, run it under
   vLLM (`RRO_RERANKER=vllm`) or llama.cpp and point the HTTP backend at it.
   Weigh this only if a benchmark shows it beating Qwen3-Reranker on your data;
   otherwise the candle Qwen3 reranker is the lower-friction default.

**Reranker reality (P7.3/P7.6, honest):** on the small eval the reranker's lift
over no-rerank was real but modest, and the 0.6B reranker showed saturation.
Rerankers earn their ~27× latency only when recall is noisy — measure lift on
*your* corpus before committing (the gate is NDCG@10 / golden@k before/after).

---

## Decision matrix

| you want | embedder | reranker | runtime |
|---|---|---|---|
| real first-touch **today**, one node, no ops | Qwen3-Emb 0.6B | Qwen3-Rerank 0.6B | **candle** |
| best quality, have a GPU | Qwen3-Emb 4B/8B | Qwen3-Rerank 4B | candle (GPU) or vLLM |
| high QPS serving | Qwen3-Emb (any) | Qwen3-Rerank | **vLLM** |
| CPU / edge / quantized | Qwen3-Emb 0.6B GGUF | Qwen3-Rerank GGUF | **llama.cpp** |
| specifically Nemotron rerank | Qwen3-Emb | Nemotron | **vLLM / llama.cpp** (never candle) |
| fine-tuned checkpoint | your dir | your dir | candle (point `RRO_*_WEIGHTS` at it) |

## "First action real" — the concrete move
The perception step (embed) is real the moment you do:
```sh
./scripts/fetch-models.sh                 # baseline 0.6b pair
RRO_REAL=1 ./scripts/quickstart.sh        # candle, real embeddings + rerank, one node
```
That gives you a real embedder + reranker **now**, on candle, while the
llama.cpp-vs-vLLM serving decision stays open — flip `RRO_EMBEDDER=vllm` (and
stand up the server) whenever you make it. Nothing in the flow changes.
