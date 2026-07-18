//! The Fjall storage backend (`kvs-fjall`).
//!
//! One of the two mutually-exclusive backends behind the KV seam (see
//! [`crate::kv`]). It maps connxism's RocksDB usage onto Fjall 3.1.7:
//!   - **column families → keyspaces** (`Database::keyspace`, one per
//!     [`COLUMN_FAMILIES`] entry),
//!   - **atomic cross-CF `WriteBatch` → `db.batch()`** committed through the
//!     shared journal,
//!   - **`IteratorMode::From`/`Start` → `Keyspace::range`/`iter`** (Fjall
//!     iterators are owned snapshots yielding [`fjall::Guard`]s),
//!   - **BlobDB on the vector CFs → KV separation** per keyspace,
//!   - and the one thing Fjall has no equivalent for — RocksDB's associative
//!     merge operator on `tdf` — is replaced by a **transaction-scoped
//!     read-modify-write** accumulated in the [`Batch`] and applied at
//!     [`Db::write`]. The on-disk `i64 LE` format is unchanged, so `tdf` needs
//!     no data migration and the two backends stay byte-compatible.
//!
//! Correctness of the RMW relies on connxism serialising every write through one
//! writer lock (see `store.rs`/`txn.rs`), so a `tdf` counter never races itself
//! between the read and the batch commit.
//!
//! ## Capability parity with the RocksDB backend
//!
//! Everything RocksDB tunes that Fjall 3.1.7 can express is set here: per-CF
//! point-lookup blooms, filter/index-block **pinning** on the hot CFs, 16 KiB
//! data blocks, LZ4/None compression, KV separation (with explicit GC) on the
//! vector CFs, role-weighted per-keyspace memtable sizing, a bounded WAL, and a
//! consistent (MVCC-pinned) snapshot. Two RocksDB capabilities have **no faithful
//! Fjall 3.1.7 equivalent** and are deliberately not faked:
//!   - **`CF_TERMS` prefix bloom.** RocksDB accelerates BM25 posting-list prefix
//!     scans with a NUL-terminated prefix extractor + memtable prefix bloom.
//!     Fjall's filters are whole-key/point-read only (no prefix filter or
//!     key-transform), so `CF_TERMS` runs unfiltered — BM25 lookups stay correct;
//!     the scan is served by leveled locality + the shared block cache. Closing
//!     this would require forking `lsm-tree`; deferred until real workloads show
//!     it matters.
//!   - **Global cross-keyspace memtable cap.** RocksDB's `db_write_buffer_size` is
//!     a single hard ceiling; Fjall's equivalent is `#[deprecated]`/off in 3.1.7.
//!     Replaced by the shaped per-keyspace budget in `memtable_size_for` + the
//!     WAL cap, so aggregate write memory is still an intentional number.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use fjall::config::{
    BlockSizePolicy, BloomConstructionPolicy, CompressionPolicy, FilterPolicy, FilterPolicyEntry,
    PinningPolicy,
};
use fjall::{
    CompressionType, Database, Keyspace, KeyspaceCreateOptions, KvSeparationOptions, PersistMode,
};
use rro_core::{Result, RroError};

use crate::estate::EstateConfig;
use crate::keys::{self, CF_META, COLUMN_FAMILIES};

/// Values at or above this size land in the keyspace's value log (KV
/// separation) instead of the LSM data blocks — the Fjall analogue of RocksDB's
/// BlobDB `min_blob_size`. A single dense vector (≥ 2 KiB) clears it; the small
/// keys around it do not, so compaction moves pointers, not vectors.
const KV_SEPARATION_THRESHOLD: u32 = 4 * 1024;

/// Data block size — matches the RocksDB backend's 16 KiB blocks (RocksDB
/// `set_block_size(16*1024)`), the unit the shared block cache holds and evicts.
/// Fjall drops to its default block size otherwise; setting it keeps the two
/// backends' cache granularity identical.
const DATA_BLOCK_SIZE: u32 = 16 * 1024;

/// Map a Fjall error into the engine error type.
fn fjall_err(e: fjall::Error) -> RroError {
    RroError::Recall(format!("kvs: {e}"))
}

/// Per-keyspace memtable ceiling, weighted by how write-hot the CF is.
///
/// Fjall's global cross-keyspace write-buffer cap (`max_write_buffer_size`, the
/// analogue of RocksDB's `db_write_buffer_size`) is `#[deprecated]`/off in 3.1.7,
/// so an aggregate budget cannot be enforced there. Instead the budget is shaped
/// here: the recall write-path CFs keep the full `base`; cold/rarely-written CFs
/// take a quarter — so the SUM across all 17 keyspaces is an intentional number,
/// not `base × keyspace_count` fallen out of the CF count.
fn memtable_size_for(name: &str, base: u64) -> u64 {
    let hot = matches!(
        name,
        keys::CF_DOCS
            | keys::CF_VECS
            | keys::CF_NVECS
            | keys::CF_MVECS
            | keys::CF_TERMS
            | keys::CF_PIDX
            | keys::CF_SPARSE
            | keys::CF_TDF
    );
    if hot {
        base
    } else {
        (base / 4).max(1)
    }
}

/// Per-keyspace options mirroring the RocksDB backend's per-CF tuning
/// (`kv/rocks.rs`): a point-lookup bloom, compression, per-CF memtable size, and
/// KV separation on the vector CFs.
fn keyspace_options(name: &str, memtable_bytes: u64) -> KeyspaceCreateOptions {
    // Point-lookup CFs get a bloom (10 bits/key ≈ 1% false positives) and are
    // told to expect hits; range-scanned CFs skip the bloom — a whole-key filter
    // cannot answer a prefix scan, so it would be pure space/cache waste.
    let point_lookup = matches!(
        name,
        keys::CF_DOCS
            | keys::CF_VECS
            | keys::CF_NVECS
            | keys::CF_MVECS
            | keys::CF_META
            | keys::CF_NODES
            | keys::CF_CONNS
            | keys::CF_COLL
            | keys::CF_TDF
    );
    let is_vector = matches!(name, keys::CF_VECS | keys::CF_NVECS | keys::CF_MVECS);

    let filter = if point_lookup {
        FilterPolicy::all(FilterPolicyEntry::Bloom(
            BloomConstructionPolicy::BitsPerKey(10.0),
        ))
    } else {
        FilterPolicy::all(FilterPolicyEntry::None)
    };
    // Dense f32 vectors do not compress meaningfully; paying CPU to not shrink
    // them is a straight loss on the hot read path. Everything else gets LZ4.
    let compression = if is_vector {
        CompressionPolicy::all(CompressionType::None)
    } else {
        CompressionPolicy::all(CompressionType::Lz4)
    };

    let mut opts = KeyspaceCreateOptions::default()
        .max_memtable_size(memtable_bytes)
        .expect_point_read_hits(point_lookup)
        .filter_policy(filter)
        .data_block_compression_policy(compression)
        // 16 KiB data blocks on every CF — the RocksDB backend's `set_block_size`,
        // restored here so both backends cache at the same granularity.
        .data_block_size_policy(BlockSizePolicy::all(DATA_BLOCK_SIZE));

    // Pin filter + index blocks resident for the point-lookup CFs — the recall
    // hot path. RocksDB did this with `cache_index_and_filter_blocks` +
    // `pin_l0_filter_and_index_blocks_in_cache` (all CFs, L0); here it is applied
    // where it pays — the CFs read by exact key — so their blooms/indexes are
    // never evicted from under a burst of point reads. `PinningPolicy::all(true)`
    // pins every level, not just L0. Range-scanned CFs keep Fjall's defaults.
    if point_lookup {
        opts = opts
            .filter_block_pinning_policy(PinningPolicy::all(true))
            .index_block_pinning_policy(PinningPolicy::all(true));
    }

    // BlobDB → KV separation on the vector CFs: values above the threshold live
    // in a value log the LSM only references, so compaction moves pointers, not
    // 10 KiB vectors. Same intent as the RocksDB BlobDB knob. GC thresholds are
    // set explicitly (at Fjall's sane defaults) rather than left implicit, the
    // analogue of RocksDB's `set_enable_blob_gc(true)`: reclaim a value-log file
    // once a third of it is stale.
    if is_vector {
        opts = opts.with_kv_separation(Some(
            KvSeparationOptions::default()
                .separation_threshold(KV_SEPARATION_THRESHOLD)
                .staleness_threshold(0.33)
                .age_cutoff(0.20),
        ));
    }
    opts
}

/// Decode an `i64 LE` counter value (tolerant of short/absent buffers).
fn read_i64(b: impl AsRef<[u8]>) -> i64 {
    let b = b.as_ref();
    let mut a = [0u8; 8];
    a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
    i64::from_le_bytes(a)
}

/// A column-family handle: a borrow of one open keyspace, carrying its canonical
/// name so batched merges can key their accumulator by CF. Same shape as the
/// RocksDB backend's `Cf` so every call site stays backend-agnostic.
#[derive(Clone, Copy)]
pub(crate) struct Cf<'a> {
    ks: &'a Keyspace,
    name: &'static str,
}

/// One raw key/value entry yielded by a store iterator (`iter_from`/`iter_all`).
pub(crate) type KvItem = Result<(Box<[u8]>, Box<[u8]>)>;

/// One staged write in a [`Batch`].
enum Op {
    Put(Keyspace, Vec<u8>, Vec<u8>),
    Delete(Keyspace, Vec<u8>),
    /// A document-frequency delta against `tdf` — applied as a read-modify-write
    /// at commit (Fjall has no merge operator). Carries the CF name so deltas to
    /// the same counter compose within one batch.
    MergeDf(Keyspace, &'static str, Vec<u8>, i64),
}

/// An accumulating atomic write, committed by [`Db::write`]. Unlike RocksDB's
/// `WriteBatch` (which is created from the DB), this is a backend-neutral op log
/// so the recall paths can build one with [`Batch::new`] and no DB handle, then
/// hand it to `Db::write` which translates it into a Fjall batch — folding the
/// `merge_df` deltas into read-modify-writes on the way.
#[derive(Default)]
pub(crate) struct Batch(Vec<Op>);

impl Batch {
    pub(crate) fn new() -> Self {
        Batch(Vec::new())
    }

    pub(crate) fn put_cf(&mut self, cf: Cf, key: impl AsRef<[u8]>, value: impl AsRef<[u8]>) {
        self.0.push(Op::Put(
            cf.ks.clone(),
            key.as_ref().to_vec(),
            value.as_ref().to_vec(),
        ));
    }

    pub(crate) fn delete_cf(&mut self, cf: Cf, key: impl AsRef<[u8]>) {
        self.0
            .push(Op::Delete(cf.ks.clone(), key.as_ref().to_vec()));
    }

    /// Stage a document-frequency delta (`±1`) against `cf`'s counter for `key`
    /// (see the module docs — this becomes an RMW at commit).
    pub(crate) fn merge_df(&mut self, cf: Cf, key: impl AsRef<[u8]>, delta: i64) {
        self.0.push(Op::MergeDf(
            cf.ks.clone(),
            cf.name,
            key.as_ref().to_vec(),
            delta,
        ));
    }
}

/// The open database: the Fjall handle, one keyspace per column family, and the
/// fsync-on-write choice. `Arc`-wrapped so clones are cheap (every
/// `ConnXRecall` holds one).
struct Inner {
    db: Database,
    parts: std::collections::HashMap<&'static str, Keyspace>,
    fsync: bool,
    path: PathBuf,
}

/// Shared handle to the open database — the KV seam, the single place that names
/// Fjall. Cloneable; all clones see one DB.
#[derive(Clone)]
pub(crate) struct Db(Arc<Inner>);

impl Db {
    /// Open (or create) the estate's Fjall database at `path`, applying the same
    /// per-CF tuning the RocksDB backend does: a shared block/blob cache, a
    /// global memtable ceiling, background workers, per-keyspace point-lookup
    /// blooms, compression, and KV separation on the vector CFs.
    pub(crate) fn open(path: &Path, config: &EstateConfig) -> Result<Db> {
        // Each keyspace's memtable is capped below (`max_memtable_size`), so the
        // total write memory is bounded by that × the live keyspaces — the same
        // budget the RocksDB backend states explicitly via `db_write_buffer_size`.
        let write_buffer_bytes = config.write_buffer_bytes as u64;
        // The intentional aggregate memtable budget: the sum of the role-weighted
        // per-keyspace ceilings (`memtable_size_for`). Fjall 3.1.7's global
        // write-buffer cap is deprecated/off, so this shaped sum is the estate's
        // stated write-memory budget instead of `write_buffer_bytes × 17`.
        let memtable_budget: u64 = COLUMN_FAMILIES
            .iter()
            .map(|name| memtable_size_for(name, write_buffer_bytes))
            .sum();

        let db = Database::builder(path)
            // One shared block/blob cache across keyspaces (RocksDB's shared LRU).
            .cache_size(config.block_cache_bytes as u64)
            // Background compaction/flush workers (RocksDB's `background_jobs`).
            .worker_threads(config.background_jobs.max(1))
            // Bound the WAL/journal — it only needs to outlive the memtables it
            // backs, so twice the memtable budget (never below Fjall's 512 MiB
            // default) is the stated ceiling. RocksDB's `max_total_wal_size`.
            .max_journaling_size(memtable_budget.saturating_mul(2).max(512 * 1024 * 1024))
            .open()
            .map_err(fjall_err)?;

        let mut parts = std::collections::HashMap::with_capacity(COLUMN_FAMILIES.len());
        for name in COLUMN_FAMILIES {
            let name = *name;
            // Each keyspace's memtable is sized by how write-hot it is, so the
            // aggregate matches `memtable_budget` above.
            let ks = db
                .keyspace(name, || {
                    keyspace_options(name, memtable_size_for(name, write_buffer_bytes))
                })
                .map_err(fjall_err)?;
            parts.insert(name, ks);
        }
        Ok(Db(Arc::new(Inner {
            db,
            parts,
            fsync: config.fsync_writes,
            path: path.to_path_buf(),
        })))
    }

    pub(crate) fn cf(&self, name: &str) -> Result<Cf<'_>> {
        self.0
            .parts
            .get_key_value(name)
            .map(|(n, ks)| Cf { ks, name: n })
            .ok_or_else(|| RroError::Recall(format!("missing column family `{name}`")))
    }

    /// Raw value read from `cf`.
    pub(crate) fn get_cf(&self, cf: Cf, key: impl AsRef<[u8]>) -> Result<Option<Vec<u8>>> {
        Ok(cf
            .ks
            .get(key)
            .map_err(fjall_err)?
            .map(|v| v.as_ref().to_vec()))
    }

    /// Raw single-key write into `cf` (bypassing a batch).
    pub(crate) fn put_cf(
        &self,
        cf: Cf,
        key: impl AsRef<[u8]>,
        value: impl AsRef<[u8]>,
    ) -> Result<()> {
        cf.ks
            .insert(key.as_ref().to_vec(), value.as_ref().to_vec())
            .map_err(fjall_err)
    }

    pub(crate) fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        cf: &str,
        key: &[u8],
    ) -> Result<Option<T>> {
        let handle = self.cf(cf)?;
        match self.get_cf(handle, key)? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    pub(crate) fn put_json<T: serde::Serialize>(
        &self,
        cf: &str,
        key: &[u8],
        value: &T,
    ) -> Result<()> {
        let handle = self.cf(cf)?;
        self.put_cf(handle, key, serde_json::to_vec(value)?)
    }

    /// Iterate `cf` from `start` forward (a prefix or range seek). Callers apply
    /// their own `starts_with`/range break, matching `IteratorMode::From`.
    pub(crate) fn iter_from<'a>(
        &'a self,
        cf: Cf<'a>,
        start: &[u8],
    ) -> impl Iterator<Item = KvItem> + 'a {
        cf.ks.range(start.to_vec()..).map(guard_to_item)
    }

    /// Iterate every entry of `cf` in key order.
    pub(crate) fn iter_all<'a>(&'a self, cf: Cf<'a>) -> impl Iterator<Item = KvItem> + 'a {
        cf.ks.iter().map(guard_to_item)
    }

    /// Commit a batch atomically, honoring the estate's fsync-on-write choice.
    ///
    /// The `merge_df` deltas are folded into the same batch as read-modify-
    /// writes: composed per counter, read once against the committed store, and
    /// re-inserted. Correct because connxism serialises writers.
    pub(crate) fn write(&self, batch: Batch) -> Result<()> {
        let mut wb = self.0.db.batch();
        // Accumulate df deltas per (CF, key) so repeated merges compose before
        // the single read-modify-write.
        let mut df: BTreeMap<(&'static str, Vec<u8>), (Keyspace, i64)> = BTreeMap::new();
        for op in batch.0 {
            match op {
                Op::Put(ks, k, v) => wb.insert(&ks, k, v),
                Op::Delete(ks, k) => wb.remove(&ks, k),
                Op::MergeDf(ks, name, k, delta) => {
                    let entry = df.entry((name, k)).or_insert((ks, 0));
                    entry.1 += delta;
                }
            }
        }
        for ((_, key), (ks, delta)) in df {
            let current = ks.get(&key).map_err(fjall_err)?.map(read_i64).unwrap_or(0);
            wb.insert(&ks, key, (current + delta).to_le_bytes().to_vec());
        }
        wb.commit().map_err(fjall_err)?;
        if self.0.fsync {
            self.0.db.persist(PersistMode::SyncAll).map_err(fjall_err)?;
        }
        Ok(())
    }

    /// Flush `cf`'s memtable to an on-disk segment (RocksDB's `flush_cf`): rotate
    /// the active memtable and wait for it to land, so the data is durable as a
    /// segment and shows up in `disk_space` — not just fsynced in the journal.
    pub(crate) fn flush_cf(&self, cf: Cf) -> Result<()> {
        cf.ks.rotate_memtable_and_wait().map_err(fjall_err)
    }

    /// Sync the journal (Fjall's write-ahead log) — RocksDB's `flush_wal(sync)`.
    /// Honors the arg: `true` fsyncs the journal (`SyncData`), `false` flushes it
    /// to the OS without an fsync (`Buffer`) — the same contract RocksDB's bool
    /// carries. (`SyncData` rather than `SyncAll`: this is a WAL sync, not a full
    /// data flush.)
    pub(crate) fn flush_wal(&self, sync: bool) -> Result<()> {
        let mode = if sync {
            PersistMode::SyncData
        } else {
            PersistMode::Buffer
        };
        self.0.db.persist(mode).map_err(fjall_err)
    }

    /// Force a full compaction of `cf` — the operator-invoked optimizer pass,
    /// the same as RocksDB's `compact_range_cf`. Best-effort like that endpoint
    /// (which cannot fail the caller); a failure is logged, not propagated.
    pub(crate) fn compact_cf(&self, cf: Cf) {
        if let Err(e) = cf.ks.major_compact() {
            tracing::warn!("fjall major_compact of `{}` failed: {e}", cf.name);
        }
    }

    /// Live on-disk bytes held by `cf`.
    pub(crate) fn cf_sst_bytes(&self, cf: Cf) -> Result<u64> {
        Ok(cf.ks.disk_space())
    }

    /// Take a consistent on-disk snapshot into `path`. Fjall has no checkpoint
    /// primitive, so this persists the journal and copies the database
    /// directory — callers snapshot at quiescent points (see `Estate`).
    pub(crate) fn snapshot_to(&self, path: &Path) -> Result<()> {
        // Pin a consistent MVCC read view for the whole copy. While this snapshot
        // is held, Fjall will not GC/compact away any segment it references, so
        // `copy_dir_all` cannot race a compaction deleting a file mid-read.
        // Combined with the caller quiescing the applier (`Estate::snapshot_to`),
        // this is Fjall's closest equivalent to RocksDB's atomic checkpoint. The
        // snapshot is a cheap seqno pin, released at end of scope.
        let _snapshot = self.0.db.snapshot();
        self.0.db.persist(PersistMode::SyncAll).map_err(fjall_err)?;
        copy_dir_all(&self.0.path, path).map_err(|e| RroError::Recall(format!("snapshot: {e}")))
    }

    pub(crate) fn get_u64(&self, key: &[u8]) -> Result<u64> {
        let handle = self.cf(CF_META)?;
        Ok(self
            .get_cf(handle, key)?
            .map(|b| {
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[..8.min(b.len())]);
                u64::from_le_bytes(a)
            })
            .unwrap_or(0))
    }
}

/// Resolve one iterator [`fjall::Guard`] into an owned key/value pair.
fn guard_to_item(guard: fjall::Guard) -> KvItem {
    let (k, v) = guard.into_inner().map_err(fjall_err)?;
    Ok((k.as_ref().into(), v.as_ref().into()))
}

/// Recursively copy a directory tree (the snapshot's file copy).
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&entry.path(), &to)?;
        } else if ty.is_file() {
            std::fs::copy(entry.path(), to)?;
        }
    }
    Ok(())
}
