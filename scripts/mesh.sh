#!/usr/bin/env bash
# Reason Ready — local mesh rollout.
#
#   ./scripts/mesh.sh [N]     # boot N nodes (default 3), each addressable
#   ./scripts/mesh.sh stop    # stop the mesh
#
# Every node is a full engine (estate + RRD + hybrid recall + flow) with its
# own a2a warp point on 127.0.0.1:79XX. Nodes are peers on the layer-2
# protocol: anything that speaks a2a JSON (ask / index / map / changes /
# ping) can treat any of them as local. MCP-transport warp points bind in
# phase P5 on the same addresses.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RRO_HOME:-$ROOT/.rro}/mesh"
BASE_PORT=7901
N="${1:-3}"

if [[ "${1:-}" == "stop" ]]; then
  for pid in "$RUN_DIR"/node-*.pid; do
    [[ -f "$pid" ]] || continue
    kill -TERM "$(cat "$pid")" 2>/dev/null || true
    rm -f "$pid"
  done
  echo "mesh stopped"
  exit 0
fi

mkdir -p "$RUN_DIR"
cargo build --release --bin rro --bin rro-bench >/dev/null

echo "── booting a ${N}-node mesh ───────────────────────────────────"
for i in $(seq 1 "$N"); do
  port=$((BASE_PORT + i - 1))
  RRO_NODE="rro-n$i" RRO_ESTATE="$RUN_DIR/estate-$i" \
  RRO_LISTEN="127.0.0.1:$port" RRO_EVENTS="$RUN_DIR/node-$i.events.jsonl" \
  RUST_LOG=warn "$ROOT/target/release/rro" >>"$RUN_DIR/node-$i.log" 2>&1 &
  echo $! > "$RUN_DIR/node-$i.pid"
done

for i in $(seq 1 "$N"); do
  port=$((BASE_PORT + i - 1))
  for _ in $(seq 1 50); do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then exec 3>&-; break; fi
    sleep 0.2
  done
  echo "node rro-n$i up — warp point tcp://127.0.0.1:$port"
done

echo "── smoke: full pipeline against every node over a2a ───────────"
for i in $(seq 1 "$N"); do
  port=$((BASE_PORT + i - 1))
  acc=$("$ROOT/target/release/rro-bench" --docs 200 --queries 10 --store estate \
        --remote "127.0.0.1:$port" 2>/dev/null | grep -oE 'accuracy@10 \(golden retrieved\)\*\* \| \*\*[0-9.]+' | grep -oE '[0-9.]+$' || echo "?")
  echo "  rro-n$i (127.0.0.1:$port): accuracy@10 = $acc"
done

echo
echo "MESH READY — $N engines, each a2a-addressable. Stop: ./scripts/mesh.sh stop"
