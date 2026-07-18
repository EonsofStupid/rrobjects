# connxism: RocksDB → Fjall 3.x port

Target: `crates/connxism` (~5.6k LOC). RocksDB usage is confined to 6 files and —
critically — already funneled through the `Db` wrapper in `estate.rs` (`cf()`,
`get_json`, `put_json`, `write(batch)`, `get_u64`). That wrapper is the seam. We do
**not** build a generic storage trait for a two-engine migration; we port the
wrapper directly and keep call sites stable.

Both backends coexist during the migration behind cargo features
(`kvs-fjall` pulls Fjall in; RocksDB stays the default) so the parity/differential
tests run against each. RocksDB stays the shipping default until Phase 7's gates
pass.

## Fjall 3.1.7 terminology

Fjall 3.1.7 renamed the classic pair. This plan uses the 3.1.7 names:

| concept | RocksDB | Fjall 3.1.7 | older Fjall |
|---|---|---|---|
| the whole store | `DB` | `Database` | `Keyspace` |
| one logical store | column family | `Keyspace` | `Partition` |
| atomic multi-store write | `WriteBatch` | `db.batch()` → `WriteBatch` | — |
| durability | `WriteOptions::set_sync` | `PersistMode::{SyncAll,Buffer}` | — |

Proven against the real API in `crates/connxism/tests/fjall_spike.rs` (4/4,
`--features kvs-fjall`): keyspaces↔CFs + restart durability, atomic cross-keyspace
batch, prefix scan, and the `tdf` RMW replacement.

---

## Phase 0 — Pin the contract (½ day) — invariants the port must preserve

1. **Atomicity unit**: one `WriteBatch` across multiple CFs commits atomically →
   Fjall cross-keyspace `db.batch()` through the shared journal. Like-for-like.
2. **Durability toggle**: `Db.1: bool` (fsync-on-write) → `db.persist(SyncAll)`
   after `batch.commit()` vs. leaving it buffered.
3. **Key ordering**: lexicographic, `SEP = 0x00` prefix encoding in `keys.rs`.
   Fjall is lexicographic; `keys.rs` ports byte-for-byte untouched. (Guard: Fjall
   keys ≤ 65,536 bytes, values ≤ 2^32 — term keys and vector blobs are inside
   both; verify no pathological tag/term input can exceed the key limit, add a
   guard in `keys.rs` if quotas don't already cover it.)
4. **Counter semantics**: `doc_count`, `total_tokens`, `feed_seq`, shape census are
   RMW owned by `Transaction` (read at begin, put at commit). Unchanged.
5. **`tdf` semantics**: associative i64 merge — the one thing Fjall has not. See
   Phase 3.

## Phase 1 — Db wrapper swap (1–2 days)

```rust
// estate.rs
pub(crate) struct Db {
    db: fjall::Database,
    parts: HashMap<&'static str, fjall::Keyspace>,  // opened once from COLUMN_FAMILIES
    fsync: bool,
}
```

- `cf(name)` → keyspace lookup (infallible after open; keep the `Result` to avoid
  churn at 60+ call sites).
- `get_json` / `put_json` / `get_u64` → `keyspace.get()` / `keyspace.insert()`.
  Single-op inserts still journal; prefer batches on hot paths, as today.
- `write(batch)` → the Fjall `WriteBatch` built in `txn.rs`, committed then
  `db.persist(SyncAll)` iff `fsync`.
- Open path: `Database::builder(path).open()` + `db.keyspace(name, opts)` per
  `COLUMN_FAMILIES`, per-keyspace `KeyspaceCreateOptions`.

## Phase 2 — Options translation (1 day). Delete more than you translate.

| RocksDB (current) | Fjall 3.x |
|---|---|
| `write_buffer_bytes` × CF budgeting | per-keyspace memtable size |
| `BlockBasedOptions` + block cache | database-level cache (unified block/blob) |
| LZ4 | LZ4 default; per-keyspace compression |
| Bloom filters | built-in per-keyspace filter policy |
| BlobDB on `vecs`/`nvecs`/`mvecs` | **KV separation** per-keyspace (value log) |
| compaction/flush thread tuning | Fjall background workers; mostly delete |
| `set_merge_operator_associative` on `tdf` | — (Phase 3) |

The BlobDB rationale (compaction must not rewrite vectors) maps directly onto
Fjall's value log — same intent, same knob shape.

## Phase 3 — Replace the `tdf` merge operator (1–2 days, the real work)

Today: blind `merge_cf(tdf, term, delta_i64_le)`, operands composed by
`merge_i64_add`. Fjall has no merge operators (3.1's compaction filters
transform/drop entries; they do not compose operands).

Replacement: **transaction-scoped delta accumulator**, mirroring what `txn.rs`
already does for counters:

- Add `df_deltas: BTreeMap<Vec<u8>, i64>` to `Transaction`.
- Write ops record `±1` per term into the map instead of emitting merge operands.
- At `commit()`: for each entry, `get` current i64 LE from `tdf`, add the delta,
  `insert` into the same batch. On-disk format (i64 LE) unchanged → **no `tdf`
  data migration**, read paths untouched.
- Correctness needs no concurrent writer between the reads and the batch commit.
  connxism is already effectively single-writer (counters read at begin); make it
  explicit with a `tokio::sync::Mutex<()>` writer gate held `begin`→`commit`.
  Recommendation: own the mutex + plain `Database` (keeps `txn.rs`'s design
  authority in our code; avoids coupling to Fjall's tx layer).
- Cost: each df touch is read+write instead of an appended operand. Batched per
  txn (one RMW per distinct term per txn), bounded by vocabulary-per-batch, not
  tokens — measure in Phase 7.

## Phase 4 — Iterators (½ day, mechanical)

- `iterator_cf(cf, From(&prefix, Forward))` + manual `starts_with` break →
  `keyspace.prefix(prefix)`; delete the manual prefix checks (~6 sites).
- `IteratorMode::Start` full scan (`vecs` rebuild) → `keyspace.iter()`.
- Fjall iterators are `DoubleEndedIterator` — free reverse scans if `query.rs`
  ever wants them. Iterators yield `Guard`; resolve with `.key()` / `.value()`.

## Phase 5 — Snapshot / flush / compaction (1 day)

- `snapshot_to` (RocksDB checkpoint) → `db.persist(SyncAll)` then a **directory
  copy** at the quiescent point (callers already snapshot at quiescence; the copy
  must include the journal). Longer-term: logical export (iterate keyspaces →
  archive), which doubles as the cross-engine migration tool.
- `flush()` / `flush_wal(true)` → `db.persist(SyncAll)`.
- Manual full-range compaction → verify Fjall 3's per-keyspace major-compaction
  surface at port time; if absent, drop the endpoint or make it advisory (LSM
  housekeeping is automatic).

## Phase 6 — Migration & rollout

1. No in-place conversion. Ship a `migrate` path: open old RocksDB read-only →
   iterate every CF → batched inserts into Fjall. Reuse `COLUMN_FAMILIES` as the
   manifest. The ANN graph rebuilds from `vecs` (the two-phase design pays off).
2. Keep RocksDB behind `kvs-rocks` for one release; `kvs-fjall` becomes default
   once gated. The `Db` seam makes this a small `cfg` surface, not a trait.
3. Parity gate: run connxism's existing suite against both engines; add a
   differential test that replays a recorded op log into both and diffs full CF
   dumps.

## Phase 7 — Bench gates (before deleting RocksDB)

- Bulk upsert throughput (firehose), realistic vector dims → validates KV
  separation.
- df-heavy ingest (high vocabulary churn) → validates the Phase 3 accumulator.
- Prefix-scan latency on `terms`/`sparse` under load.
- Snapshot time + recovery-from-copy correctness.

## Risks

- **Fjall 3 disk format is young** (Jan 2026); maintainer signalled feature work
  winding down into 2026 (reads as stabilization). We trade RocksDB's decade of
  scar tissue for a cleaner codebase; the op-log differential test is the
  insurance.
- Blocking-I/O discipline unchanged: Fjall is sync like RocksDB — keep the
  existing `spawn_blocking` boundaries.
- **Do not port and redesign simultaneously.** The `tdf` accumulator (Phase 3) is
  the only semantic change; everything else must be behavior-preserving or the
  differential test loses meaning.

**Estimated effort: ~6–9 focused days**, dominated by Phases 3 and 6.
