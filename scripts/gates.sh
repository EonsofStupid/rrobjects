#!/usr/bin/env bash
# gates.sh — run the `#[ignore]`d gates: the ones that need real weights and
# live servers, and that CI therefore cannot run.
#
# These are the tests that decide whether the model claims are true. Every one of
# them sat unrun until 2026-07-17 — `ci.yml` has no `--run-ignored`, so "candle,
# llama.cpp and vLLM all work" was an assertion for as long as it had been said.
# This script is how a box with weights turns that into a number.
#
#   ./scripts/gates.sh            # everything reachable with what is running
#   ./scripts/gates.sh --list     # what each gate needs, without running it
#
# Nothing here is skipped silently: a gate whose server or weights are absent
# says so and is counted as SKIPPED, never as passed.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${MODELS:=$HOME/Projects/clyffy/models}"
: "${EMBED_06B:=$MODELS/embedders/qwen3-embedding-0-6b}"
: "${RERANK_06B:=$MODELS/rerankers/qwen3-reranker-0-6b}"

# NOTE on the agreement gate: candle and the HTTP engine must run the SAME
# weights or the comparison is theatre. :8090 serves the 4B; the 0.6B GGUF on
# :8095 is what matches candle's 0.6B safetensors. Pointing this at 8090 would
# compare different models and print a confident, meaningless number.
: "${LLAMACPP_06B:=http://127.0.0.1:8095/v1/embeddings}"
: "${BRAIN:=http://127.0.0.1:8091/v1/chat/completions}"
: "${RERANK_LLAMACPP:=http://127.0.0.1:8093/v1/rerank}"
: "${RERANK_VLLM:=http://127.0.0.1:8092/rerank}"
: "${VECTORS:=/tmp/rro-real-vectors.jsonl}"

up() { curl -sf -m 2 "${1%/v1/*}/v1/models" >/dev/null 2>&1 || curl -sf -m 2 "${1%/*}/models" >/dev/null 2>&1; }
have() { [ -f "$1/model.safetensors" ] && [ -f "$1/tokenizer.json" ]; }

if [ "${1:-}" = "--list" ]; then
  printf '%-22s %s\n' "gate" "needs"
  printf '%-22s %s\n' "candle_qwen_gate"  "weights: $EMBED_06B"
  printf '%-22s %s\n' "engine_agreement"  "weights + an HTTP engine on the SAME model ($LLAMACPP_06B)"
  printf '%-22s %s\n' "rerank_lift"       "$RERANK_LLAMACPP , $RERANK_VLLM , weights: $RERANK_06B"
  printf '%-22s %s\n' "live_brain"        "$BRAIN"
  printf '%-22s %s\n' "real_vector_ef"    "$VECTORS (make with: rro-bench --export)"
  exit 0
fi

pass=0; skip=0; fail=0
run() { # name, condition-msg, then the command
  local name="$1"; shift
  echo "=== $name"
  if ! "$@"; then fail=$((fail+1)); echo "  FAIL $name"; else pass=$((pass+1)); fi
}

if have "$EMBED_06B"; then
  run candle_qwen_gate env RRO_TEST_QWEN_WEIGHTS="$EMBED_06B" \
    cargo test -p embedder --features candle --test candle_qwen_gate -- --ignored --nocapture
else echo "SKIP candle_qwen_gate — no weights at $EMBED_06B"; skip=$((skip+1)); fi

if have "$EMBED_06B" && up "$LLAMACPP_06B"; then
  run engine_agreement env RRO_TEST_QWEN_WEIGHTS="$EMBED_06B" RRO_TEST_LLAMACPP="$LLAMACPP_06B" \
    cargo test -p embedder --features candle --test engine_agreement -- --ignored --nocapture
else echo "SKIP engine_agreement — needs weights AND a same-model engine at $LLAMACPP_06B"; skip=$((skip+1)); fi

if up "$RERANK_LLAMACPP" || up "$RERANK_VLLM"; then
  run rerank_lift env RRO_TEST_RERANK_LLAMACPP="$RERANK_LLAMACPP" RRO_TEST_RERANK_VLLM="$RERANK_VLLM" \
    RRO_TEST_QWEN_RERANK_WEIGHTS="$RERANK_06B" \
    cargo test -p reranker --features candle --test rerank_lift -- --ignored --nocapture
else echo "SKIP rerank_lift — no reranker server"; skip=$((skip+1)); fi

if up "$BRAIN"; then
  run live_brain env RRO_TEST_BRAIN="$BRAIN" \
    cargo test -p classifier --test live_brain -- --ignored --nocapture
else echo "SKIP live_brain — no brain at $BRAIN"; skip=$((skip+1)); fi

if [ -s "$VECTORS" ]; then
  run real_vector_ef env RRO_TEST_VECTORS="$VECTORS" \
    cargo test -p recall --release --test real_vector_ef -- --ignored --nocapture
else
  echo "SKIP real_vector_ef — no vectors at $VECTORS. Make them:"
  echo "  RRO_EMBEDDER=llamacpp RRO_EMBEDDER_ENDPOINT=http://127.0.0.1:8095 \\"
  echo "    cargo run --release --bin rro-bench -- --docs 50000 --export $VECTORS"
  skip=$((skip+1))
fi

# Filter-aware HNSW correctness at scale (200k–300k docs). No server or weights
# needed — just too slow for CI's debug build (~90 s each), so #[ignore]d and run
# in release here. The mechanism is also unit-tested fast in-CI
# (recall::ann::filter_aware_tests); these are the full-scale correctness gates.
run filter_aware_hnsw cargo test -p connxism --release --test filter_aware_hnsw -- --ignored --nocapture

echo
echo "gates: $pass passed, $skip skipped, $fail failed"
[ "$fail" -eq 0 ]
