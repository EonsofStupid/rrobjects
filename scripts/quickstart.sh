#!/usr/bin/env bash
# Reason Ready — turnkey quickstart.
#
#   ./scripts/quickstart.sh            # build, boot, smoke-test over a2a
#   ./scripts/quickstart.sh stop       # stop the daemon
#
# One command yields a running engine: persistent estate, RRD front door,
# ANN-indexed hybrid recall, reranker, readiness classifier, a2a listener,
# DuckDB-ready event stream — then proves it end-to-end over the wire and
# prints the flow stages the engine emitted while answering.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RRO_HOME:-$ROOT/.rro}"
ESTATE="$RUN_DIR/estate"
EVENTS="$RUN_DIR/events.jsonl"
PIDFILE="$RUN_DIR/rro.pid"
ADDR="${RRO_LISTEN:-127.0.0.1:7878}"

stop() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill -TERM "$(cat "$PIDFILE")" && sleep 1
    echo "stopped rro (pid $(cat "$PIDFILE"))"
  else
    echo "no running rro daemon found"
  fi
  rm -f "$PIDFILE"
}

if [[ "${1:-}" == "stop" ]]; then stop; exit 0; fi

mkdir -p "$RUN_DIR"

# ── real models (opt-in) ───────────────────────────────────────────────────
# Weightless by default (synthetic embedder, dev/CI). Set RRO_REAL=1 — or point
# RRO_EMBEDDER/RRO_RERANKER at a candle backend — to boot the real Qwen3 models,
# fetching their weights on first run.
FEATURES=""
if [[ "${RRO_REAL:-}" == "1" ]]; then
  RRO_EMBEDDER="${RRO_EMBEDDER:-candle-qwen}"
  RRO_RERANKER="${RRO_RERANKER:-candle-cross-encoder}"
fi
if [[ "${RRO_EMBEDDER:-}" == candle* || "${RRO_RERANKER:-}" == candle* ]]; then
  FEATURES="--features candle"
  export RRO_DEVICE="${RRO_DEVICE:-cpu}"
  # Which size from the catalog (see fetch-models.sh --list): 0.6b (baseline) | 4b | 8b.
  EMBED_SIZE="${RRO_EMBED_SIZE:-0.6b}"
  RERANK_SIZE="${RRO_RERANK_SIZE:-0.6b}"
  MODELS_DIR="${RRO_MODELS_DIR:-$ROOT/models}"
  if [[ "${RRO_EMBEDDER:-}" == candle* ]]; then
    export RRO_EMBEDDER
    export RRO_EMBEDDER_WEIGHTS="${RRO_EMBEDDER_WEIGHTS:-$MODELS_DIR/qwen3-embedding-$EMBED_SIZE}"
    if ! RRO_MODELS_DIR="$MODELS_DIR" "$ROOT/scripts/fetch-models.sh" --check "embed-$EMBED_SIZE" >/dev/null 2>&1; then
      echo "── fetching embedder weights: embed-$EMBED_SIZE (first run) ────────"
      RRO_MODELS_DIR="$MODELS_DIR" "$ROOT/scripts/fetch-models.sh" "embed-$EMBED_SIZE"
    fi
  fi
  if [[ "${RRO_RERANKER:-}" == candle* ]]; then
    export RRO_RERANKER
    export RRO_RERANKER_WEIGHTS="${RRO_RERANKER_WEIGHTS:-$MODELS_DIR/qwen3-reranker-$RERANK_SIZE}"
    if ! RRO_MODELS_DIR="$MODELS_DIR" "$ROOT/scripts/fetch-models.sh" --check "rerank-$RERANK_SIZE" >/dev/null 2>&1; then
      echo "── fetching reranker weights: rerank-$RERANK_SIZE (first run) ──────"
      RRO_MODELS_DIR="$MODELS_DIR" "$ROOT/scripts/fetch-models.sh" "rerank-$RERANK_SIZE"
    fi
  fi
  echo "models: embedder=${RRO_EMBEDDER:-deterministic}($EMBED_SIZE) reranker=${RRO_RERANKER:-lexical}($RERANK_SIZE) device=$RRO_DEVICE"
fi

echo "── building (release${FEATURES:+, $FEATURES}) ─────────────────────────"
# shellcheck disable=SC2086 # FEATURES is "" or "--features candle"; word-split is intended
cargo build --release $FEATURES --bin rro --bin rro-bench

echo "── booting the engine ─────────────────────────────────────────"
[[ -f "$PIDFILE" ]] && stop
RRO_ESTATE="$ESTATE" RRO_LISTEN="$ADDR" RRO_EVENTS="$EVENTS" RUST_LOG=info \
  "$ROOT/target/release/rro" >>"$RUN_DIR/rro.log" 2>&1 &
echo $! > "$PIDFILE"

for _ in $(seq 1 50); do
  if (exec 3<>"/dev/tcp/${ADDR%:*}/${ADDR#*:}") 2>/dev/null; then exec 3>&-; break; fi
  sleep 0.2
done
echo "engine up: pid $(cat "$PIDFILE"), a2a on $ADDR"
echo "estate:    $ESTATE"
echo "events:    $EVENTS"

echo "── smoke test: full pipeline over a2a (layer-2) ───────────────"
"$ROOT/target/release/rro-bench" --docs 500 --queries 25 --store estate \
  --remote "$ADDR" | grep -E "accuracy|p50|throughput" || true

echo "── flow stages the engine emitted while answering ─────────────"
grep '"flow.stage"' "$EVENTS" | tail -5 || echo "(no stage events yet)"

echo
echo "READY. Ask it something:"
echo "  target/release/rro-bench --docs 0 --queries 1 --remote $ADDR   # or speak a2a JSON on $ADDR"
echo "Stop it:  ./scripts/quickstart.sh stop"
