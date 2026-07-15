# Observability — signals, events, baselines

## Signals (consistent by contract)

The daemon handles the full Unix signal set — `SIGHUP`, `SIGINT`, `SIGQUIT`,
`SIGTERM` — and **every receipt is emitted as a `signal.received` event before
shutdown proceeds**, so the analytics stream always records *why* the process
stopped. `serve.start` / `serve.stop` bracket the daemon's life. Non-Unix
platforms fall back to Ctrl-C.

## The event stream (DuckDB-ready)

Every meaningful transition in the engine is emitted through one global sink
(`rrf_core::events`): structured records `{at_ms, kind, fields}` written as
JSONL. Install it with `RRF_EVENTS=/path/events.jsonl` on the daemon or
`--events <path>` on `rrf-bench`.

Current kinds:

| kind | emitted by | fields |
|---|---|---|
| `serve.start` / `serve.stop`   | daemon | node_id, listen / signal |
| `signal.received`              | daemon | signal |
| `ingest.batch`                 | ingestion machine | indexed, errors, batch_ms, total_indexed, docs_per_sec |
| `ingest.finished`              | ingestion machine | received, indexed, errors, batches, docs_per_sec |
| `bench.start` / `bench.result` / `bench.baseline` | rrf-bench | config / headline numbers / verdict |

Query it directly:

```sql
-- ingestion throughput trend across a run
SELECT at_ms, CAST(fields.docs_per_sec AS DOUBLE) AS dps
FROM read_json_auto('rrf-events.jsonl')
WHERE kind = 'ingest.batch'
ORDER BY at_ms;

-- why did the daemon stop?
SELECT fields.signal FROM read_json_auto('rrf-events.jsonl')
WHERE kind = 'signal.received';
```

The estate additionally records durable **trend** series (`Estate::record_trend`
/ `Estate::trend`) inside RocksDB itself — events are the hot stream, trends
are the engine's own long-term memory of its behavior.

## Baseline configuration & tracking

A baseline is a recorded run: **configuration + headline numbers**. The
configuration is part of the baseline — comparisons against a different
config are refused (exit 2), so numbers are never compared across unlike runs.

```sh
# record (per machine, per store)
rrf-bench --docs 50000 --queries 500 --store estate --write-baseline baselines/container-estate.json

# gate (exit 1 on regression beyond tolerance; default ±25%)
rrf-bench --docs 50000 --queries 500 --store estate --baseline baselines/container-estate.json --tolerance 20
```

Checked metrics: ingest docs/sec (higher is better), query p50/p95 ms (lower
is better). `baselines/` carries the recorded container baselines; record your
own on dedicated hardware and tighten the tolerance there.
