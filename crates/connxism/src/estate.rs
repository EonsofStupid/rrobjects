//! The estate: one RocksDB, fully managed.
//!
//! [`Estate`] owns the database and everything registry-shaped: estate info,
//! nodes + warp points, connectors + sync state, tags, shape census, and trend
//! series. The recall store ([`crate::ConnXRecall`]) shares the same database
//! through a cheap clone of the inner handle.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock as StdRwLock};

use recall::{AnnConfig, AnnIndex, Quantizer};
use rocksdb::{ColumnFamily, Options, DB};
use rro_core::{Embedding, Id, Result, RroError};

use crate::keys::{
    self, CF_CONNS, CF_META, CF_NODES, CF_TAGS, CF_TRENDS, CF_VECS, COLUMN_FAMILIES,
};
use crate::model::{now_ms, ConnectorInfo, EstateInfo, NodeInfo, SyncState, TrendPoint, WarpPoint};

/// Shared handle to the open database. Cloneable; all clones see one DB.
#[derive(Clone)]
pub(crate) struct Db(pub(crate) Arc<DB>, pub(crate) bool);

impl Db {
    pub(crate) fn cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.0
            .cf_handle(name)
            .ok_or_else(|| RroError::Recall(format!("missing column family `{name}`")))
    }

    pub(crate) fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        cf: &str,
        key: &[u8],
    ) -> Result<Option<T>> {
        let handle = self.cf(cf)?;
        match self.0.get_cf(handle, key).map_err(rocks_err)? {
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
        self.0
            .put_cf(handle, key, serde_json::to_vec(value)?)
            .map_err(rocks_err)
    }

    /// Commit a batch, honoring the estate's fsync-on-write choice.
    pub(crate) fn write(&self, batch: rocksdb::WriteBatch) -> Result<()> {
        if self.1 {
            let mut wo = rocksdb::WriteOptions::default();
            wo.set_sync(true);
            self.0.write_opt(batch, &wo).map_err(rocks_err)
        } else {
            self.0.write(batch).map_err(rocks_err)
        }
    }

    pub(crate) fn get_u64(&self, key: &[u8]) -> Result<u64> {
        let handle = self.cf(CF_META)?;
        Ok(self
            .0
            .get_cf(handle, key)
            .map_err(rocks_err)?
            .map(|b| {
                let mut a = [0u8; 8];
                a.copy_from_slice(&b[..8.min(b.len())]);
                u64::from_le_bytes(a)
            })
            .unwrap_or(0))
    }
}

/// Associative merge: value = existing + Σ operand (i64 LE deltas).
fn merge_i64_add(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &rocksdb::MergeOperands,
) -> Option<Vec<u8>> {
    let read = |b: &[u8]| -> i64 {
        let mut a = [0u8; 8];
        a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
        i64::from_le_bytes(a)
    };
    let mut acc = existing.map(read).unwrap_or(0);
    for op in operands {
        acc += read(op);
    }
    Some(acc.to_le_bytes().to_vec())
}

/// Map RocksDB errors into the engine error type.
pub(crate) fn rocks_err(e: rocksdb::Error) -> RroError {
    RroError::Recall(format!("kvs: {e}"))
}

/// Map filesystem errors (the vector sidecar) into the engine error type.
pub(crate) fn io_err(e: std::io::Error) -> RroError {
    RroError::Recall(format!("graph sidecar: {e}"))
}

/// The canonical keyword for a schemafull type, or `None` if it is not a type
/// this engine enforces.
pub(crate) fn field_type_of(ty: &str) -> Option<&'static str> {
    match ty {
        "string" => Some("string"),
        "int" => Some("int"),
        "float" => Some("float"),
        "bool" => Some("bool"),
        "datetime" => Some("datetime"),
        "uuid" => Some("uuid"),
        _ => None,
    }
}

/// Whether a metadata value satisfies a declared schemafull type. Datetime and
/// UUID are validated by actually parsing the string (order-preserving encodings
/// the payload index already relies on), so the schema and the index agree.
pub(crate) fn value_matches_type(v: &serde_json::Value, ty: &str) -> bool {
    match ty {
        "string" => v.is_string(),
        "int" => v.is_i64() || v.is_u64(),
        "float" => v.is_number(),
        "bool" => v.is_boolean(),
        "datetime" => v
            .as_str()
            .and_then(rro_core::time::rfc3339_to_epoch_ms)
            .is_some(),
        "uuid" => v
            .as_str()
            .and_then(rro_core::time::parse_uuid_bytes)
            .is_some(),
        _ => false,
    }
}

/// Resource limits enforced at the write and query boundaries. `None`
/// means unlimited. Operational config (like quantization), not index
/// identity — set at open, reported in health.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Quotas {
    /// Estate-wide document cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_docs: Option<u64>,
    /// Per-document metadata size cap (serialized JSON bytes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_payload_bytes: Option<usize>,
    /// Query `top_k` cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_top_k: Option<usize>,
    /// Upsert batch-size cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_batch: Option<usize>,
}

impl Quotas {
    /// Sane strict-mode defaults (the daemon's `RRO_STRICT=1`).
    pub fn strict() -> Self {
        Quotas {
            max_docs: None,
            max_payload_bytes: Some(64 * 1024),
            max_top_k: Some(1024),
            max_batch: Some(4096),
        }
    }
}

/// Open-time choices for an estate.
#[derive(Debug, Clone)]
pub struct EstateConfig {
    /// How the ANN graph stores its vectors: `None` (full f32), `Sq8` (~4×
    /// smaller), or `Bq` (~32× smaller). Lossy modes are **rescored exactly**
    /// from the durable vector column family, so scores at the API stay exact —
    /// quantization is a memory/IO decision, not an accuracy one.
    pub quantizer: Quantizer,
    /// The text analyzer for the lexical (BM25) index. **Fixed at estate
    /// creation** — it is part of the index's identity (postings and
    /// queries must agree on what a token is); reopening ignores this
    /// field in favour of the persisted one.
    pub analyzer: rro_core::text::Analyzer,
    /// Sync every write batch to disk (fsync) before acknowledging.
    /// Durability over throughput; the WAL already survives process
    /// crashes either way — this survives power loss.
    pub fsync_writes: bool,
    /// Resource limits (strict mode). Default: unlimited.
    pub quotas: Quotas,
    /// Maintain per-term document frequencies (blind merges) — the stats
    /// behind max-score lexical pruning (8.3× on selective+common
    /// queries, measured). Costs ~⅓ of ingest throughput on unique-token-
    /// heavy corpora (one extra key write per new term per doc — also
    /// measured). Default ON; turn off to buy ingest, the scorer falls
    /// back to full scans.
    pub lexical_stats: bool,
    /// Shared LRU block cache, bytes.
    ///
    /// RocksDB's default is **8 MiB shared across every column family**, which
    /// for a 16-CF estate evicts the hot blocks continuously. This is the single
    /// biggest read-path knob and the first thing to raise on a dedicated box.
    pub block_cache_bytes: usize,
    /// Per-CF memtable size, bytes. Kept explicit so it is a decision rather
    /// than an inherited default.
    pub write_buffer_bytes: usize,
    /// Background compaction + flush threads. RocksDB defaults to 2, which
    /// stalls writes on a many-core box during ingest.
    pub background_jobs: usize,
    /// Resident budget for the paged ANN vector cache, bytes. The graph's
    /// vectors live in a node-ordered sidecar file (`graph.vectors`) paged
    /// through a bounded cache of this size — so RAM tracks the working set, not
    /// the dataset. The single biggest read-path knob for the vector side on a
    /// large estate; raise it on a box with memory to spare.
    pub graph_cache_bytes: usize,
}

impl Default for EstateConfig {
    fn default() -> Self {
        EstateConfig {
            quantizer: Quantizer::None,
            analyzer: rro_core::text::Analyzer::default(),
            fsync_writes: false,
            quotas: Quotas::default(),
            lexical_stats: true,
            // 256 MiB. RocksDB's default is 8 MiB shared across every CF, which
            // for a 16-CF estate means the hot blocks are evicted continuously.
            // 256 MiB is a working default for a node with GBs to spare and is
            // the first thing to raise on a dedicated box.
            block_cache_bytes: 256 * 1024 * 1024,
            // 64 MiB per CF memtable (RocksDB's own default), kept explicit so
            // it is a decision rather than an inheritance.
            write_buffer_bytes: 64 * 1024 * 1024,
            // Compaction + flush threads. RocksDB defaults to 2, which stalls
            // writes on a many-core box during ingest.
            background_jobs: 4,
            // 256 MiB paged vector cache — bounds vector RAM independently of the
            // dataset, which can be tens of GB. Raise on a memory-rich box.
            graph_cache_bytes: 256 * 1024 * 1024,
        }
    }
}

/// One operator estate: the kvs-connectome over a single RocksDB.
pub struct Estate {
    pub(crate) db: Db,
    /// The ANN graph over the estate's vectors. Two-phase by contract: the
    /// `vecs` column family is the durable source of truth; the graph is
    /// applied **out-of-band** by the applier thread and rebuilt from `vecs`
    /// on open. Searches overlay the pending set for read-your-writes.
    pub(crate) ann: Arc<StdRwLock<AnnIndex>>,
    /// Not-yet-applied graph ops + the applier's signaling.
    pub(crate) pending: Arc<crate::pending::Pending>,
    /// How the graph stores its vectors. Lossy modes drive exact rescore, and BQ
    /// drives a wider over-fetch (its coarse codes need more candidates).
    pub(crate) quantizer: Quantizer,
    /// Fired after every committed changefeed append (upsert/remove), so
    /// push-stream watchers wake event-driven instead of polling.
    pub(crate) feed_notify: Arc<tokio::sync::Notify>,
    /// Resource limits enforced by the recall store.
    pub(crate) quotas: Quotas,
    /// Whether df stats are maintained (drives write-path merges).
    pub(crate) lexical_stats: bool,
    /// Path to the paged ANN vector sidecar (`<estate>/graph.vectors`) — written
    /// by [`Estate::persist_graph`], memory-mapped-free paged on open.
    vectors_path: PathBuf,
    /// True when this open loaded the graph from CF_GRAPH rather than rebuilding
    /// it from the durable vectors — the observable behind the startup gate.
    graph_loaded: bool,
    /// Whether [`Estate::persist_graph`] runs on drop (default true). Turning it
    /// off leaves the on-disk graph blob as it is, so the next open sees whatever
    /// state a crash would have left — the hook the crash-recovery test needs.
    persist_on_drop: std::sync::atomic::AtomicBool,
    applier: Option<std::thread::JoinHandle<()>>,
    info: EstateInfo,
}

impl Estate {
    /// Open (or create) the estate at `path` with default configuration.
    pub fn open(path: impl AsRef<Path>, name: &str) -> Result<Self> {
        Self::open_with(path, name, EstateConfig::default())
    }

    /// Open (or create) the estate at `path`. Rebuilds the ANN graph from
    /// the durable vector column family (the two-phase pattern's recovery
    /// path — the graph is always reconstructible).
    pub fn open_with(path: impl AsRef<Path>, name: &str, config: EstateConfig) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // ---- RocksDB, actually configured -------------------------------
        //
        // Everything below was RocksDB defaults until 2026-07-16, which for a
        // 16-CF estate serving point lookups meant: an 8 MiB block cache shared
        // by every CF, and NO bloom filters — so each point lookup (a doc, a
        // vector, a posting) touched every SST at every level before answering.
        // For an engine whose whole claim is sub-ms recall, that is not a
        // detail; it is the difference between a memory hit and a disk walk.
        //
        // Sized from the config so a laptop and the GB10 are the same code with
        // a different number, not two paths.
        opts.increase_parallelism(config.background_jobs as i32);
        opts.set_max_background_jobs(config.background_jobs as i32);

        // The memtable budget is the sum nobody was computing. `write_buffer_size`
        // is set PER CF (below), and RocksDB keeps up to `max_write_buffer_number`
        // (default 2) live per CF — so the real ceiling is
        // `write_buffer_bytes × max_write_buffer_number × COLUMN_FAMILY_COUNT`.
        // At the defaults that is 64 MiB × 2 × 16 = **2 GiB** of memtables, a
        // number that just fell out of a per-CF knob nobody multiplied out. Cap
        // it explicitly with `db_write_buffer_size`, a hard ceiling across all
        // CFs, so the estate's write memory is a stated budget rather than an
        // accident of the CF count.
        let max_write_buffers = 2u64;
        let memtable_budget =
            (config.write_buffer_bytes as u64) * max_write_buffers * (COLUMN_FAMILIES.len() as u64);
        opts.set_db_write_buffer_size(memtable_budget as usize);

        // One shared block cache across CFs: a per-CF cache partitions memory by
        // guesswork, and the hot set moves with the workload.
        let cache = rocksdb::Cache::new_lru_cache(config.block_cache_bytes);

        let block_opts = |bloom: bool| {
            let mut b = rocksdb::BlockBasedOptions::default();
            b.set_block_cache(&cache);
            b.set_block_size(16 * 1024);
            // Cache index/filter blocks WITH the data, and pin the top level:
            // otherwise the filters get evicted under load and the bloom stops
            // helping exactly when it matters.
            b.set_cache_index_and_filter_blocks(true);
            b.set_pin_l0_filter_and_index_blocks_in_cache(true);
            if bloom {
                // 10 bits/key ~= 1% false positives — the standard point-lookup
                // trade. Only on CFs actually read by exact key.
                b.set_bloom_filter(10.0, false);
            }
            b
        };

        // Per-CF options, matched to how each CF is actually read.
        let descriptors: Vec<rocksdb::ColumnFamilyDescriptor> = COLUMN_FAMILIES
            .iter()
            .map(|cf| {
                let mut cf_opts = Options::default();

                // Point-lookup CFs get a bloom; range-scanned CFs do not (a
                // filter cannot answer a prefix scan, so it would be pure write
                // amplification and cache pressure).
                let point_lookup = matches!(
                    *cf,
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
                cf_opts.set_block_based_table_factory(&block_opts(point_lookup));

                // Vectors are dense f32 that do not compress meaningfully;
                // paying CPU to not shrink them is a straight loss on the hot
                // read path. Everything else (text, postings, JSON) does.
                if matches!(*cf, keys::CF_VECS | keys::CF_NVECS | keys::CF_MVECS) {
                    cf_opts.set_compression_type(rocksdb::DBCompressionType::None);

                    // BlobDB for the vector CFs. A 2560-d f32 vector is ~10 KiB,
                    // and in a plain LSM every value is rewritten by every
                    // compaction that touches its key — so the vectors, which
                    // never change, get copied over and over for the sake of
                    // compacting the small keys around them. BlobDB stores values
                    // above `min_blob_size` in separate blob files that the LSM
                    // only references, so compaction moves 8-byte pointers instead
                    // of 10 KiB payloads. `min_blob_size` is set below a single
                    // vector so every vector lands in a blob; nothing smaller
                    // (there is nothing smaller in these CFs) pays the indirection.
                    cf_opts.set_enable_blob_files(true);
                    cf_opts.set_min_blob_size(4 * 1024);
                    cf_opts.set_enable_blob_gc(true);
                } else {
                    cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                }

                // The BM25 postings CF is read by **prefix scan**: keys are
                // `term \x00 doc_id` and a lexical lookup seeks `term \x00` then
                // iterates. A whole-key bloom cannot help a scan (it answers "is
                // this exact key present", not "does this prefix exist"), so
                // `CF_TERMS` was left with no filter at all — every posting-list
                // read paid full index descent. A **prefix** extractor + memtable
                // prefix bloom fixes exactly that: the bloom answers "could this
                // term have any postings" and skips SSTables and memtables that
                // hold none. The extractor is custom because terms are
                // variable-length — the prefix is everything up to and including
                // the first NUL, not a fixed byte count.
                if *cf == keys::CF_TERMS {
                    cf_opts.set_prefix_extractor(rocksdb::SliceTransform::create(
                        "term_prefix",
                        |key: &[u8]| match key.iter().position(|&b| b == 0) {
                            Some(nul) => &key[..=nul],
                            None => key,
                        },
                        Some(|key: &[u8]| key.contains(&0)),
                    ));
                    cf_opts.set_memtable_prefix_bloom_ratio(0.1);
                }

                cf_opts.set_write_buffer_size(config.write_buffer_bytes);

                // `tdf` carries an associative merge operator so document-
                // frequency counters update as blind merge writes.
                if *cf == keys::CF_TDF {
                    cf_opts.set_merge_operator_associative("i64_add", merge_i64_add);
                }
                rocksdb::ColumnFamilyDescriptor::new(*cf, cf_opts)
            })
            .collect();
        let db = DB::open_cf_descriptors(&opts, path.as_ref(), descriptors).map_err(rocks_err)?;
        let db = Db(Arc::new(db), config.fsync_writes);

        let info = match db.get_json::<EstateInfo>(CF_META, keys::META_ESTATE)? {
            Some(existing) => existing,
            None => {
                let fresh = EstateInfo {
                    id: format!("estate-{:x}", now_ms()),
                    name: name.to_string(),
                    created_at: now_ms(),
                    dim: None,
                    named_dims: std::collections::BTreeMap::new(),
                    analyzer: config.analyzer.clone(),
                };
                db.put_json(CF_META, keys::META_ESTATE, &fresh)?;
                // Fresh estate with stats on: df is maintained from the
                // first write, unlocking the pruned lexical scorer.
                if config.lexical_stats {
                    db.0.put_cf(db.cf(CF_META)?, keys::META_LEXSTATS, 1u64.to_le_bytes())
                        .map_err(rocks_err)?;
                }
                fresh
            }
        };

        // Load the ANN graph. A clean shutdown persists the graph's *structure* to
        // CF_GRAPH (tagged with the changefeed seq it was captured at) and its
        // *vectors* to a node-ordered `graph.vectors` sidecar. If that seq still
        // matches the live feed_seq, nothing has changed the graph since, so we
        // load the structure and page the vectors from the sidecar — opening a
        // 10M-vector estate in read-time, with RAM tracking the working set rather
        // than the dataset. Any mismatch, absence, or decode failure falls through
        // to the rebuild, so the durable vectors remain the sole source of truth
        // and a stale or corrupt cache can never serve wrong results.
        let vectors_path = path.as_ref().join("graph.vectors");
        let ann_config = AnnConfig {
            quantizer: config.quantizer,
            ..AnnConfig::default()
        };
        let (ann, graph_loaded) = match Self::load_persisted_graph(
            &db,
            ann_config.clone(),
            &vectors_path,
            config.graph_cache_bytes,
        )? {
            Some(loaded) => {
                tracing::info!(
                    vectors = loaded.len(),
                    "ann graph loaded — structure + paged vectors"
                );
                (loaded, true)
            }
            None => (Self::rebuild_ann_from_vecs(&db, ann_config)?, false),
        };

        let ann = Arc::new(StdRwLock::new(ann));
        let pending = crate::pending::Pending::new();
        let applier = pending.spawn_applier(ann.clone());

        // Stats are live only when configured AND the estate has carried
        // them since creation (the pruned scorer's precondition).
        let lexical_stats = config.lexical_stats
            && db
                .0
                .get_cf(db.cf(CF_META)?, keys::META_LEXSTATS)
                .map_err(rocks_err)?
                .is_some();
        Ok(Estate {
            db,
            ann,
            pending,
            quantizer: config.quantizer,
            feed_notify: Arc::new(tokio::sync::Notify::new()),
            quotas: config.quotas.clone(),
            lexical_stats,
            vectors_path,
            graph_loaded,
            persist_on_drop: std::sync::atomic::AtomicBool::new(true),
            applier: Some(applier),
            info,
        })
    }

    /// The changefeed signal: notified after every committed upsert/remove.
    /// Watchers await it between [`Estate::changes`] drains — event-driven
    /// push, no polling interval.
    pub fn feed_signal(&self) -> Arc<tokio::sync::Notify> {
        self.feed_notify.clone()
    }

    /// Estate metadata.
    pub fn info(&self) -> &EstateInfo {
        &self.info
    }

    // ---- nodes & warp points -------------------------------------------------

    /// Register (or replace) a node.
    pub fn register_node(&self, mut node: NodeInfo) -> Result<()> {
        node.last_seen = now_ms();
        self.db.put_json(CF_NODES, node.id.as_bytes(), &node)
    }

    /// Add a warp point to an existing node.
    pub fn add_warp_point(&self, node_id: &str, warp: WarpPoint) -> Result<()> {
        let mut node: NodeInfo = self
            .db
            .get_json(CF_NODES, node_id.as_bytes())?
            .ok_or_else(|| RroError::msg(format!("no such node: {node_id}")))?;
        node.warp_points.push(warp);
        self.register_node(node)
    }

    /// Fetch one node.
    pub fn node(&self, id: &str) -> Result<Option<NodeInfo>> {
        self.db.get_json(CF_NODES, id.as_bytes())
    }

    /// All registered nodes.
    pub fn nodes(&self) -> Result<Vec<NodeInfo>> {
        self.scan_json(CF_NODES)
    }

    // ---- connectors ----------------------------------------------------------

    /// Register (or replace) a connector.
    pub fn register_connector(&self, conn: ConnectorInfo) -> Result<()> {
        self.db.put_json(CF_CONNS, conn.id.as_bytes(), &conn)
    }

    /// Fetch one connector.
    pub fn connector(&self, id: &str) -> Result<Option<ConnectorInfo>> {
        self.db.get_json(CF_CONNS, id.as_bytes())
    }

    /// All registered connectors.
    pub fn connectors(&self) -> Result<Vec<ConnectorInfo>> {
        self.scan_json(CF_CONNS)
    }

    /// Update a connector's sync state (cursor advance, status change, counts).
    pub fn update_sync(&self, connector_id: &str, sync: SyncState) -> Result<()> {
        let mut conn: ConnectorInfo = self
            .db
            .get_json(CF_CONNS, connector_id.as_bytes())?
            .ok_or_else(|| RroError::msg(format!("no such connector: {connector_id}")))?;
        conn.sync = sync;
        self.register_connector(conn)
    }

    // ---- tags ----------------------------------------------------------------

    /// Attach tags to a document (idempotent).
    pub fn tag(&self, doc_id: &str, tags: &[String]) -> Result<()> {
        let handle = self.db.cf(CF_TAGS)?;
        for t in tags {
            self.db
                .0
                .put_cf(handle, keys::tag_key(t, doc_id), [])
                .map_err(rocks_err)?;
        }
        Ok(())
    }

    /// All document ids carrying `tag`.
    pub fn docs_by_tag(&self, tag: &str) -> Result<Vec<String>> {
        let handle = self.db.cf(CF_TAGS)?;
        let prefix = keys::tag_prefix(tag);
        let mut out = Vec::new();
        let iter = self.db.0.iterator_cf(
            handle,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );
        for item in iter {
            let (k, _) = item.map_err(rocks_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            if let Some((_, doc)) = keys::split_compound(&k) {
                out.push(String::from_utf8_lossy(doc).into_owned());
            }
        }
        Ok(out)
    }

    /// All distinct tags in the estate.
    pub fn tags(&self) -> Result<Vec<String>> {
        let handle = self.db.cf(CF_TAGS)?;
        let mut out: Vec<String> = Vec::new();
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (k, _) = item.map_err(rocks_err)?;
            if let Some((tag, _)) = keys::split_compound(&k) {
                let tag = String::from_utf8_lossy(tag).into_owned();
                if out.last().map(|t| t != &tag).unwrap_or(true) {
                    out.push(tag);
                }
            }
        }
        Ok(out)
    }

    /// Number of documents carrying `tag`.
    pub fn tag_count(&self, tag: &str) -> Result<u64> {
        Ok(self.docs_by_tag(tag)?.len() as u64)
    }

    // ---- payload secondary indexes --------------------------------------------

    /// Index a metadata field for filter-first queries. Registers the field
    /// **before** backfilling, so writes that land mid-backfill maintain
    /// their own rows (index puts are idempotent — overlap is harmless);
    /// then backfills from every stored document. Idempotent per field.
    pub fn create_payload_index(&self, field: &str) -> Result<()> {
        let mut fields = crate::filter::indexed_fields(&self.db)?;
        if fields.iter().any(|f| f == field) {
            return Ok(());
        }
        fields.push(field.to_string());
        self.db.put_json(CF_META, keys::META_PIDX, &fields)?;

        // Backfill from the durable documents.
        let docs_cf = self.db.cf(crate::keys::CF_DOCS)?;
        let pidx_cf = self.db.cf(crate::keys::CF_PIDX)?;
        let mut batch = rocksdb::WriteBatch::default();
        let mut rows = 0u64;
        for item in self.db.0.iterator_cf(docs_cf, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            let doc: crate::model::StoredDoc = serde_json::from_slice(&v)?;
            if let Some(value) = doc.metadata.get(field) {
                batch.put_cf(pidx_cf, keys::pidx_key(field, value, &doc.id), []);
                rows += 1;
                if rows.is_multiple_of(4096) {
                    self.db
                        .0
                        .write(std::mem::take(&mut batch))
                        .map_err(rocks_err)?;
                }
            }
        }
        self.db.write(batch)?;
        rro_core::events::emit(
            "estate.payload_index",
            serde_json::json!({ "field": field, "rows": rows }),
        );
        Ok(())
    }

    /// The payload-indexed field names.
    pub fn payload_indexes(&self) -> Result<Vec<String>> {
        crate::filter::indexed_fields(&self.db)
    }

    /// REBUILD INDEX: drop every row of `field`'s payload index and
    /// re-backfill from the durable documents. This is also the migration
    /// path when value typing changes (e.g. datetime/UUID strings gaining
    /// typed keys): old rows are swept, new rows carry the current encoding.
    pub fn rebuild_payload_index(&self, field: &str) -> Result<()> {
        if !crate::filter::indexed_fields(&self.db)?
            .iter()
            .any(|f| f == field)
        {
            return Err(rro_core::RroError::Recall(format!(
                "`{field}` has no payload index to rebuild"
            )));
        }
        let pidx_cf = self.db.cf(crate::keys::CF_PIDX)?;
        let prefix = keys::pidx_field_prefix(field);
        let mut batch = rocksdb::WriteBatch::default();
        for item in self.db.0.iterator_cf(
            pidx_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        ) {
            let (k, _) = item.map_err(rocks_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            batch.delete_cf(pidx_cf, k);
        }
        let docs_cf = self.db.cf(crate::keys::CF_DOCS)?;
        let mut rows = 0u64;
        for item in self.db.0.iterator_cf(docs_cf, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            let doc: crate::model::StoredDoc = serde_json::from_slice(&v)?;
            if let Some(value) = doc.metadata.get(field) {
                batch.put_cf(pidx_cf, keys::pidx_key(field, value, &doc.id), []);
                rows += 1;
            }
        }
        self.db.write(batch)?;
        rro_core::events::emit(
            "estate.payload_index.rebuilt",
            serde_json::json!({ "field": field, "rows": rows }),
        );
        Ok(())
    }

    /// The exact id-set matching `filter`, resolved from payload indexes.
    /// `None` when the filter references an unindexed field (callers fall
    /// back to post-filtering).
    pub fn ids_where(&self, filter: &crate::Filter) -> Result<Option<Vec<String>>> {
        crate::filter::ids_where(&self.db, filter)
    }

    /// Count documents matching a DSL filter: index-resolved when every
    /// referenced field is indexed, full scan otherwise.
    pub fn count_where(&self, filter: &crate::Filter) -> Result<u64> {
        if let Some(ids) = crate::filter::ids_where(&self.db, filter)? {
            return Ok(ids.len() as u64);
        }
        let handle = self.db.cf(crate::keys::CF_DOCS)?;
        let mut n = 0u64;
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            let doc: crate::model::StoredDoc = serde_json::from_slice(&v)?;
            if filter.matches(&doc.metadata) {
                n += 1;
            }
        }
        Ok(n)
    }

    // ---- shapes ---------------------------------------------------------------

    /// The shape census: canonical shape key → document count.
    pub fn shapes(&self) -> Result<std::collections::BTreeMap<String, u64>> {
        Ok(self
            .db
            .get_json(CF_META, keys::META_SHAPES)?
            .unwrap_or_default())
    }

    // ---- trends ---------------------------------------------------------------

    /// Record one sample of `metric` at now.
    pub fn record_trend(&self, metric: &str, value: f64) -> Result<()> {
        let handle = self.db.cf(CF_TRENDS)?;
        self.db
            .0
            .put_cf(
                handle,
                keys::trend_key(metric, crate::model::now_ns()),
                value.to_le_bytes(),
            )
            .map_err(rocks_err)
    }

    /// The stored series for `metric`, oldest first.
    pub fn trend(&self, metric: &str) -> Result<Vec<TrendPoint>> {
        let handle = self.db.cf(CF_TRENDS)?;
        let prefix = keys::trend_prefix(metric);
        let mut out = Vec::new();
        for item in self.db.0.iterator_cf(
            handle,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        ) {
            let (k, v) = item.map_err(rocks_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            let at_ns = k[prefix.len()..]
                .try_into()
                .map(u64::from_be_bytes)
                .unwrap_or(0);
            let mut b = [0u8; 8];
            b.copy_from_slice(&v[..8.min(v.len())]);
            out.push(TrendPoint {
                at: at_ns / 1_000_000, // key is ns; the point reads in ms
                value: f64::from_le_bytes(b),
            });
        }
        Ok(out)
    }

    /// All distinct trend metric names.
    pub fn trend_metrics(&self) -> Result<Vec<String>> {
        let handle = self.db.cf(CF_TRENDS)?;
        let mut out: Vec<String> = Vec::new();
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (k, _) = item.map_err(rocks_err)?;
            if let Some((metric, _)) = keys::split_compound(&k) {
                let metric = String::from_utf8_lossy(metric).into_owned();
                if out.last().map(|m| m != &metric).unwrap_or(true) {
                    out.push(metric);
                }
            }
        }
        Ok(out)
    }

    // ---- internals -------------------------------------------------------------

    /// Block until the applier has drained every pending graph op.
    pub fn quiesce(&self) {
        self.pending.quiesce();
    }

    // ---- snapshots ----------------------------------------------------------------

    /// Write a consistent point-in-time snapshot of the whole estate to
    /// `path` (RocksDB checkpoint: hard-links immutable SST files, copies the
    /// WAL — cheap and crash-consistent). The snapshot directory opens as a
    /// fully working estate via [`Estate::open`]; its persisted graph loads
    /// directly (or rebuilds from the durable vectors, as on any open).
    pub fn snapshot_to(&self, path: impl AsRef<Path>) -> Result<()> {
        // Capture the graph first so the checkpoint carries a loadable blob and
        // the snapshot opens in read-time too (best-effort; a rebuild covers any
        // decline). Callers snapshot at quiescent points, which is what
        // `persist_graph` needs.
        let _ = self.persist_graph();
        let checkpoint = rocksdb::checkpoint::Checkpoint::new(&self.db.0).map_err(rocks_err)?;
        checkpoint
            .create_checkpoint(path.as_ref())
            .map_err(rocks_err)?;
        rro_core::events::emit(
            "estate.snapshot",
            serde_json::json!({ "path": path.as_ref().display().to_string() }),
        );
        Ok(())
    }

    // ---- component state --------------------------------------------------------

    /// Persist engine-component state (e.g. the RRD shape baseline) under the
    /// meta column family, namespaced with an `x:` prefix.
    pub fn put_component_json<T: serde::Serialize>(&self, key: &str, value: &T) -> Result<()> {
        self.db
            .put_json(CF_META, format!("x:{key}").as_bytes(), value)
    }

    /// Load engine-component state stored via
    /// [`Estate::put_component_json`].
    pub fn get_component_json<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> Result<Option<T>> {
        self.db.get_json(CF_META, format!("x:{key}").as_bytes())
    }

    // ---- changefeed ------------------------------------------------------------

    /// Read changefeed entries with `seq >= since_seq`, oldest first, up to
    /// `limit`. Feed rows are written atomically with the writes they record;
    /// `seq` is the resume cursor for subscribers.
    pub fn changes(&self, since_seq: u64, limit: usize) -> Result<Vec<crate::model::Change>> {
        let handle = self.db.cf(crate::keys::CF_FEED)?;
        let start = since_seq.to_be_bytes();
        let mut out = Vec::new();
        for item in self.db.0.iterator_cf(
            handle,
            rocksdb::IteratorMode::From(&start, rocksdb::Direction::Forward),
        ) {
            if out.len() >= limit {
                break;
            }
            let (_, v) = item.map_err(rocks_err)?;
            out.push(serde_json::from_slice(&v)?);
        }
        Ok(out)
    }

    /// The **replication** view of the changefeed: each change carried with the
    /// record a follower needs to apply it. Mirrors [`Self::changes`] (same
    /// `since_seq` cursor, same ordering, same limit) but resolves each upsert to
    /// its current dense record so a follower can rebuild the estate from the
    /// stream alone — no back-channel fetch.
    ///
    /// An upsert whose document no longer exists (already removed by a later,
    /// higher-seq change) yields `record: None`; the follower skips it and the
    /// later remove reconciles. This makes the stream **convergent**: replaying
    /// `[since_seq, head]` in order reaches the leader's current state.
    pub async fn replication_batch(
        &self,
        since_seq: u64,
        limit: usize,
    ) -> Result<Vec<crate::model::ReplEntry>> {
        let changes = self.changes(since_seq, limit)?;
        let recall = self.recall();
        let mut out = Vec::with_capacity(changes.len());
        for c in changes {
            let record = match c.op {
                crate::model::ChangeOp::Upsert => {
                    match (
                        recall.vector_of(&c.doc_id).await?,
                        recall.doc(&c.doc_id).await?,
                    ) {
                        (Some(embedding), Some(doc)) => Some(crate::model::ReplRecord {
                            id: c.doc_id.clone(),
                            embedding,
                            text: doc.text,
                            metadata: doc.metadata,
                            collection: doc.collection,
                        }),
                        // Gone (superseded by a later remove) — the follower skips
                        // it; the later entry reconciles.
                        _ => None,
                    }
                }
                crate::model::ChangeOp::Remove => None,
            };
            out.push(crate::model::ReplEntry {
                seq: c.seq,
                op: c.op,
                doc_id: c.doc_id,
                record,
            });
        }
        Ok(out)
    }

    fn scan_json<T: serde::de::DeserializeOwned>(&self, cf: &str) -> Result<Vec<T>> {
        let handle = self.db.cf(cf)?;
        let mut out = Vec::new();
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            out.push(serde_json::from_slice(&v)?);
        }
        Ok(out)
    }
}

impl Drop for Estate {
    fn drop(&mut self) {
        // Clean shutdown: persist the derived graph so the next open loads it in
        // read-time instead of rebuilding in O(N log N). Best-effort — this is the
        // last handle, so no writer can race, but if it fails for any reason the
        // durable vectors still rebuild the graph on open. A crash that skips Drop
        // simply leaves a stale/absent blob, which the seq check rejects → rebuild.
        if self
            .persist_on_drop
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            if let Err(err) = self.persist_graph() {
                tracing::warn!(%err, "graph persist on shutdown failed — open will rebuild");
            }
        }
        // Stop the applier cleanly; unapplied pendings are already durable in
        // the vecs column family and reappear via rebuild-on-open.
        self.pending.stop();
        if let Some(handle) = self.applier.take() {
            let _ = handle.join();
        }
    }
}

impl Estate {
    /// The named collections in this estate, with exact member counts.
    pub fn collections(&self) -> Result<Vec<(String, u64)>> {
        let names: Vec<String> = self
            .db
            .get_json(CF_META, keys::META_COLLECTIONS)?
            .unwrap_or_default();
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            out.push((name.clone(), self.collection_members(&name)?.len() as u64));
        }
        Ok(out)
    }

    /// The member doc ids of one collection (sorted — a prefix scan).
    pub fn collection_members(&self, name: &str) -> Result<Vec<String>> {
        let handle = self.db.cf(keys::CF_COLL)?;
        let prefix = keys::coll_prefix(name);
        let mut out = Vec::new();
        for item in self.db.0.iterator_cf(
            handle,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        ) {
            let (k, _) = item.map_err(rocks_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            out.push(String::from_utf8_lossy(&k[prefix.len()..]).into_owned());
        }
        Ok(out)
    }

    /// Drop a collection: fully remove every member (postings, vectors,
    /// payload/sparse/named rows, changefeed removes) and deregister the
    /// name. Documents outside the collection are untouched.
    pub fn drop_collection(&self, name: &str) -> Result<u64> {
        let members = self.collection_members(name)?;
        let analyzer = self.info.analyzer.clone();
        for id in &members {
            crate::store::remove_blocking(
                &self.db,
                &self.pending,
                &analyzer,
                id,
                self.lexical_stats,
            )?;
            self.pending.push_remove(rro_core::Id::new(id.clone()));
        }
        let mut registry: Vec<String> = self
            .db
            .get_json(CF_META, keys::META_COLLECTIONS)?
            .unwrap_or_default();
        registry.retain(|n| n != name);
        self.db
            .put_json(CF_META, keys::META_COLLECTIONS, &registry)?;
        rro_core::events::emit(
            "estate.collection.dropped",
            serde_json::json!({ "name": name, "members": members.len() }),
        );
        Ok(members.len() as u64)
    }
}

impl Estate {
    /// Random sampling: up to `n` documents drawn by a deterministic
    /// seeded reservoir over the doc column family (same seed, same corpus
    /// → same sample; no RNG dependencies).
    pub fn sample(&self, n: usize, seed: u64) -> Result<Vec<crate::model::StoredDoc>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let handle = self.db.cf(crate::keys::CF_DOCS)?;
        let mut reservoir: Vec<crate::model::StoredDoc> = Vec::with_capacity(n);
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = move |bound: u64| -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound.max(1)
        };
        let mut seen = 0u64;
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            seen += 1;
            if reservoir.len() < n {
                reservoir.push(serde_json::from_slice(&v)?);
            } else {
                let j = next(seen);
                if (j as usize) < n {
                    reservoir[j as usize] = serde_json::from_slice(&v)?;
                }
            }
        }
        Ok(reservoir)
    }
}

impl Estate {
    /// Create (or repoint) an alias to a collection. The whole alias map
    /// writes as one blob, so a repoint (alias switch) is atomic: every
    /// query sees either the old target or the new one, never neither.
    pub fn create_alias(&self, alias: &str, collection: &str) -> Result<()> {
        let mut aliases: std::collections::BTreeMap<String, String> = self
            .db
            .get_json(CF_META, keys::META_ALIASES)?
            .unwrap_or_default();
        aliases.insert(alias.to_string(), collection.to_string());
        self.db.put_json(CF_META, keys::META_ALIASES, &aliases)?;
        rro_core::events::emit(
            "estate.alias",
            serde_json::json!({ "alias": alias, "collection": collection }),
        );
        Ok(())
    }

    /// Declare a schemafull field type on a collection: a record in `collection`
    /// whose metadata carries `field` must match `ty` (a type keyword —
    /// `string`/`int`/`float`/`bool`/`datetime`/`uuid`) or the write is rejected.
    pub fn define_field(&self, collection: &str, field: &str, ty: &str) -> Result<()> {
        if field_type_of(ty).is_none() {
            return Err(RroError::msg(format!("unknown field type `{ty}`")));
        }
        let mut schema = self.schema()?;
        schema
            .entry(collection.to_string())
            .or_default()
            .insert(field.to_string(), ty.to_string());
        self.db.put_json(CF_META, keys::META_SCHEMA, &schema)?;
        rro_core::events::emit(
            "estate.define_field",
            serde_json::json!({ "collection": collection, "field": field, "type": ty }),
        );
        Ok(())
    }

    /// Drop a schemafull field constraint.
    pub fn remove_field(&self, collection: &str, field: &str) -> Result<()> {
        let mut schema = self.schema()?;
        if let Some(fields) = schema.get_mut(collection) {
            fields.remove(field);
            if fields.is_empty() {
                schema.remove(collection);
            }
        }
        self.db.put_json(CF_META, keys::META_SCHEMA, &schema)?;
        Ok(())
    }

    /// The whole schema map: `collection → { field → type keyword }`.
    pub fn schema(
        &self,
    ) -> Result<std::collections::BTreeMap<String, std::collections::BTreeMap<String, String>>>
    {
        Ok(self
            .db
            .get_json(CF_META, keys::META_SCHEMA)?
            .unwrap_or_default())
    }

    /// The alias map (alias → collection).
    pub fn aliases(&self) -> Result<std::collections::BTreeMap<String, String>> {
        Ok(self
            .db
            .get_json(CF_META, keys::META_ALIASES)?
            .unwrap_or_default())
    }

    /// Delete an alias (the underlying collection is untouched).
    pub fn delete_alias(&self, alias: &str) -> Result<()> {
        let mut aliases: std::collections::BTreeMap<String, String> = self
            .db
            .get_json(CF_META, keys::META_ALIASES)?
            .unwrap_or_default();
        aliases.remove(alias);
        self.db.put_json(CF_META, keys::META_ALIASES, &aliases)
    }
}

/// Live + tombstoned node counts of the ANN graph — the compaction gauge.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct GraphNodes {
    /// Total nodes, live plus tombstoned.
    pub total: usize,
    /// Dead nodes (removed or overwritten) still occupying the graph.
    pub tombstones: usize,
}

impl GraphNodes {
    /// Live (non-tombstoned) nodes.
    pub fn live(&self) -> usize {
        self.total - self.tombstones
    }
}

/// Before/after of a [`Estate::compact_graph`] pass.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct GraphCompaction {
    /// Node counts before compaction.
    pub before: GraphNodes,
    /// Node counts after (tombstone-free).
    pub after: GraphNodes,
}

impl GraphCompaction {
    /// Nodes reclaimed (the tombstones that were dropped).
    pub fn reclaimed(&self) -> usize {
        self.before.total.saturating_sub(self.after.total)
    }
}

/// A live snapshot of the estate's operational state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthReport {
    /// Total indexed documents.
    pub docs: u64,
    /// Next changefeed sequence (== total feed rows ever written).
    pub feed_seq: u64,
    /// Graph ops awaiting the out-of-band applier.
    pub applier_backlog: usize,
    /// Named collections with member counts.
    pub collections: Vec<(String, u64)>,
    /// Default dense dimensionality (None until the first upsert).
    pub dim: Option<usize>,
    /// Named vector spaces and their dims.
    pub named_dims: std::collections::BTreeMap<String, usize>,
    /// Whether the ANN graph holds SQ8 codes.
    pub quantized: bool,
    /// Tombstoned (dead) ANN nodes awaiting compaction — the gauge for when a
    /// graph compaction is worth running.
    #[serde(default)]
    pub graph_tombstones: usize,
    /// Live SST bytes per column family (optimizer status).
    #[serde(default)]
    pub cf_bytes: Vec<(String, u64)>,
    /// The configured resource limits.
    #[serde(default)]
    pub quotas: Quotas,
}

/// One self-reported operational concern.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Issue {
    /// Stable machine code (e.g. `applier_backlog`).
    pub code: String,
    /// Human-readable detail.
    pub detail: String,
}

impl Estate {
    /// A live operational snapshot (cheap: counters + registry reads).
    /// Estate info is re-read from the database — `dim`/`named_dims` are
    /// written by upserts after open, so the boot-time copy goes stale.
    pub fn health(&self) -> Result<HealthReport> {
        let info: EstateInfo = self
            .db
            .get_json(CF_META, keys::META_ESTATE)?
            .unwrap_or_else(|| self.info.clone());
        Ok(HealthReport {
            docs: self.db.get_u64(crate::keys::META_DOC_COUNT)?,
            feed_seq: self.db.get_u64(crate::keys::META_FEED_SEQ)?,
            applier_backlog: self.pending.backlog(),
            collections: self.collections()?,
            dim: info.dim,
            named_dims: info.named_dims,
            quantized: self.quantizer.is_lossy(),
            graph_tombstones: self.graph_nodes().tombstones,
            cf_bytes: self.cf_sizes()?,
            quotas: self.quotas.clone(),
        })
    }

    /// Self-reported issues, derived from what the estate already tracks.
    /// `backlog_threshold`: how many queued graph ops count as concerning.
    pub fn issues(&self, backlog_threshold: usize) -> Result<Vec<Issue>> {
        let h = self.health()?;
        let mut out = Vec::new();
        if h.applier_backlog > backlog_threshold {
            out.push(Issue {
                code: "applier_backlog".into(),
                detail: format!(
                    "{} graph ops queued (threshold {backlog_threshold}); searches stay \
                     correct via the pending overlay but latency grows with the backlog",
                    h.applier_backlog
                ),
            });
        }
        if h.docs > 0 && h.dim.is_none() {
            out.push(Issue {
                code: "dim_unset".into(),
                detail: format!("{} docs but no recorded dimensionality", h.docs),
            });
        }
        if h.feed_seq < h.docs {
            out.push(Issue {
                code: "feed_behind".into(),
                detail: format!(
                    "feed_seq {} < doc count {} — the changefeed should record at \
                     least one row per document",
                    h.feed_seq, h.docs
                ),
            });
        }
        Ok(out)
    }
}

impl Estate {
    /// The blind-maintained document frequency of one term (diagnostics;
    /// the pruned lexical scorer's input).
    pub fn term_df(&self, term: &str) -> Result<i64> {
        Ok(self
            .db
            .0
            .get_cf(self.db.cf(keys::CF_TDF)?, term.as_bytes())
            .map_err(rocks_err)?
            .map(|b| {
                let mut a = [0u8; 8];
                a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
                i64::from_le_bytes(a)
            })
            .unwrap_or(0))
    }
}

impl Estate {
    /// Read the persisted ANN graph — structure blob from CF_GRAPH, vectors paged
    /// from the sidecar — but only if it is still current. The blob is
    /// `[feed_seq u64 LE][AnnIndex::to_structure_bytes]`; the seq tag is the
    /// graph's as-of point on the changefeed. When it equals the live `feed_seq`
    /// nothing has mutated the graph since it was captured, so the structure is
    /// exact and the sidecar (validated by length inside `from_paged`) matches it.
    /// Any staleness, absence, decode failure, or missing/mismatched sidecar
    /// returns `None`, and the caller rebuilds from the durable vectors — the
    /// cache is never trusted over the source of truth.
    fn load_persisted_graph(
        db: &Db,
        config: AnnConfig,
        vectors_path: &Path,
        cache_bytes: usize,
    ) -> Result<Option<AnnIndex>> {
        let handle = db.cf(keys::CF_GRAPH)?;
        let Some(blob) = db.0.get_cf(handle, keys::GRAPH_ANN).map_err(rocks_err)? else {
            return Ok(None);
        };
        if blob.len() < 8 {
            return Ok(None);
        }
        let seq = u64::from_le_bytes(blob[..8].try_into().expect("8 bytes"));
        let live_seq = db.get_u64(keys::META_FEED_SEQ)?;
        if seq != live_seq {
            tracing::info!(
                persisted = seq,
                live = live_seq,
                "persisted graph stale — rebuilding"
            );
            return Ok(None);
        }
        let file = match File::open(vectors_path) {
            Ok(f) => f,
            Err(err) => {
                tracing::warn!(%err, "graph vector sidecar missing — rebuilding");
                return Ok(None);
            }
        };
        match AnnIndex::from_paged(&blob[8..], file, config, cache_bytes) {
            Some(ann) => Ok(Some(ann)),
            None => {
                tracing::warn!("persisted graph failed to decode / sidecar mismatch — rebuilding");
                Ok(None)
            }
        }
    }

    /// Build a fresh ANN graph by re-inserting every durable vector. The `vecs`
    /// column family holds exactly the live vectors (an overwrite replaces the
    /// row, a remove deletes it), so the rebuilt graph carries **no tombstones**
    /// — this is both the open-time recovery path and the compaction path.
    fn rebuild_ann_from_vecs(db: &Db, config: AnnConfig) -> Result<AnnIndex> {
        let mut ann = AnnIndex::new(config);
        let handle = db.cf(CF_VECS)?;
        let mut rebuilt = 0u64;
        for item in db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (k, v) = item.map_err(rocks_err)?;
            let id = Id::new(String::from_utf8_lossy(&k).into_owned());
            ann.insert(id, &Embedding(keys::decode_vec(&v)));
            rebuilt += 1;
        }
        if rebuilt > 0 {
            tracing::info!(rebuilt, "ann graph rebuilt from durable vectors");
        }
        Ok(ann)
    }

    /// Compact the ANN graph: rebuild it tombstone-free from the durable vectors
    /// and swap it in. Deletes and overwrites leave dead nodes behind (soft
    /// tombstones — still traversed, still occupying RAM/disk); over a churny
    /// estate's life they dominate, so this is the vector-side optimizer that
    /// reclaims them. Returns the (before, after) node counts.
    ///
    /// The rebuild runs O(N log N) **without** the graph lock; the swap is atomic
    /// under the write lock, and is applied only if no write raced the rebuild —
    /// otherwise the fresh graph would be missing those writes, so it retries.
    /// Best invoked during a quiet window (a maintenance op, like `compact`).
    pub fn compact_graph(&self) -> Result<GraphCompaction> {
        let before = {
            let g = self.ann.read().expect("ann lock");
            GraphNodes {
                total: g.node_count(),
                tombstones: g.tombstones(),
            }
        };
        let ann_config = AnnConfig {
            quantizer: self.quantizer,
            ..AnnConfig::default()
        };
        for _ in 0..4 {
            self.quiesce();
            let seq0 = self.db.get_u64(keys::META_FEED_SEQ)?;
            // Rebuild lock-free; a concurrent write may land in `vecs` meanwhile.
            let mut fresh = Self::rebuild_ann_from_vecs(&self.db, ann_config.clone())?;
            fresh.seal(); // train PQ if applicable, so the fresh graph matches
            self.quiesce();
            // Swap only if nothing changed the vectors during the rebuild.
            if self.pending.backlog() == 0 && self.db.get_u64(keys::META_FEED_SEQ)? == seq0 {
                let after = GraphNodes {
                    total: fresh.node_count(),
                    tombstones: fresh.tombstones(),
                };
                *self.ann.write().expect("ann lock") = fresh;
                rro_core::events::emit(
                    "estate.graph_compact",
                    serde_json::json!({
                        "before_total": before.total,
                        "after_total": after.total,
                        "reclaimed": before.total.saturating_sub(after.total),
                    }),
                );
                return Ok(GraphCompaction { before, after });
            }
        }
        Err(RroError::msg(
            "graph compaction kept racing writes — retry during a quieter window",
        ))
    }

    /// Whether this open loaded the graph from CF_GRAPH (`true`) or rebuilt it
    /// from the durable vectors (`false`). The observable behind "restart loads,
    /// not rebuilds."
    pub fn graph_was_loaded(&self) -> bool {
        self.graph_loaded
    }

    /// Live + tombstoned node counts of the ANN graph.
    pub fn graph_nodes(&self) -> GraphNodes {
        let g = self.ann.read().expect("ann lock");
        GraphNodes {
            total: g.node_count(),
            tombstones: g.tombstones(),
        }
    }

    /// Heap bytes the ANN graph holds for vectors — the RAM tail plus the bounded
    /// page cache, never the on-disk sidecar. Observable behind "RSS tracks the
    /// working set, not the dataset": stays bounded by `graph_cache_bytes` even as
    /// the vector file grows to tens of GB.
    pub fn graph_heap_vector_bytes(&self) -> usize {
        self.ann.read().expect("ann lock").heap_vector_bytes()
    }

    /// Enable/disable persisting the graph on drop (default enabled). Disabling
    /// leaves the on-disk blob untouched at shutdown, so the next open sees the
    /// state a crash would have left — a stale or absent graph that must rebuild.
    pub fn set_persist_graph_on_drop(&self, on: bool) {
        self.persist_on_drop
            .store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// Capture the ANN graph so the next open loads it instead of rebuilding —
    /// the write side of `load_persisted_graph`. The structure (ids, links,
    /// tombstones, SQ8 params) goes to CF_GRAPH tagged with the changefeed seq;
    /// the vectors go to the `graph.vectors` sidecar, streamed so a multi-GB write
    /// never materializes in RAM.
    ///
    /// Must be called at a quiescent point (shutdown, or an operator-driven
    /// snapshot): the graph and the seq it is tagged with have to be captured for
    /// the *same* state, and there is no lock that freezes commits. This holds the
    /// ANN read lock (which stalls the applier) and re-reads `feed_seq` around the
    /// capture; if a write races in, it declines rather than write a mis-tagged
    /// blob — the open path just rebuilds. Persisting is always optional: the
    /// durable vectors are the truth.
    pub fn persist_graph(&self) -> Result<()> {
        self.quiesce();
        // Train the PQ codebook if the graph is still warming up, so it serializes
        // as codes + codebook rather than raw full vectors. No-op for other
        // quantizers and for an already-trained PQ graph. Quiescent, so the brief
        // write lock races nothing.
        self.ann.write().expect("ann lock poisoned").seal();
        let guard = self.ann.read().expect("ann lock poisoned");
        // A nonzero backlog means committed graph ops the applier has not yet
        // folded in — the graph trails the changefeed and cannot be tagged.
        if self.pending.backlog() != 0 {
            tracing::debug!("skipping graph persist: applier backlog nonzero");
            return Ok(());
        }
        let seq_before = self.db.get_u64(keys::META_FEED_SEQ)?;
        let structure = guard.to_structure_bytes();

        // Write the sidecar first (atomic temp + rename), so the structure blob we
        // commit next always references an at-least-as-new sidecar; the length
        // check in `from_paged` rejects any residual skew. Skip the (potentially
        // multi-GB) rewrite when the graph is already fully paged from the current
        // sidecar and has taken no vector writes since — only tombstones/counters
        // changed, and those live in the structure blob.
        if guard.needs_vector_flush() {
            let tmp = self.vectors_path.with_extension("vectors.tmp");
            let file = File::create(&tmp).map_err(io_err)?;
            let mut w = BufWriter::new(file);
            guard.write_vectors_to(&mut w).map_err(io_err)?;
            w.flush().map_err(io_err)?;
            w.into_inner()
                .map_err(|e| io_err(e.into_error()))?
                .sync_all()
                .map_err(io_err)?;
            std::fs::rename(&tmp, &self.vectors_path).map_err(io_err)?;
        }
        drop(guard);

        // If a commit slipped in while we captured (it can bump feed_seq even
        // under the read lock, though the applier could not have folded it in),
        // the structure no longer corresponds to `seq_before` — decline. The
        // sidecar we may have written is harmless: without a matching structure
        // blob the open path rebuilds.
        if self.db.get_u64(keys::META_FEED_SEQ)? != seq_before {
            tracing::debug!("skipping graph persist: feed_seq moved mid-capture");
            return Ok(());
        }
        let mut blob = Vec::with_capacity(8 + structure.len());
        blob.extend_from_slice(&seq_before.to_le_bytes());
        blob.extend_from_slice(&structure);
        let handle = self.db.cf(keys::CF_GRAPH)?;
        self.db
            .0
            .put_cf(handle, keys::GRAPH_ANN, &blob)
            .map_err(rocks_err)?;
        self.db.0.flush_cf(handle).map_err(rocks_err)?;
        rro_core::events::emit(
            "estate.graph_persist",
            serde_json::json!({ "seq": seq_before, "structure_bytes": blob.len() }),
        );
        Ok(())
    }

    /// Flush every column family's memtable and sync the WAL — the
    /// explicit ack point: after this returns, everything acknowledged is
    /// on disk regardless of process or power fate.
    pub fn flush(&self) -> Result<()> {
        for name in COLUMN_FAMILIES {
            self.db.0.flush_cf(self.db.cf(name)?).map_err(rocks_err)?;
        }
        self.db.0.flush_wal(true).map_err(rocks_err)?;
        rro_core::events::emit("estate.flush", serde_json::json!({}));
        Ok(())
    }

    /// Manual full-range compaction of every column family — the
    /// operator-invoked optimizer pass (RocksDB runs its own background
    /// compactions continuously; this forces a full pass now).
    pub fn compact(&self) -> Result<()> {
        for name in COLUMN_FAMILIES {
            self.db
                .0
                .compact_range_cf(self.db.cf(name)?, None::<&[u8]>, None::<&[u8]>);
        }
        rro_core::events::emit("estate.compact", serde_json::json!({}));
        Ok(())
    }

    /// Live SST bytes per column family (the optimizer-status numbers).
    pub fn cf_sizes(&self) -> Result<Vec<(String, u64)>> {
        let mut out = Vec::with_capacity(COLUMN_FAMILIES.len());
        for name in COLUMN_FAMILIES {
            let bytes = self
                .db
                .0
                .property_int_value_cf(self.db.cf(name)?, "rocksdb.total-sst-files-size")
                .map_err(rocks_err)?
                .unwrap_or(0);
            out.push((name.to_string(), bytes));
        }
        Ok(out)
    }
}

/// Changefeed shape (the SHOW-CHANGES numbers).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FeedStats {
    /// The oldest retained sequence number (None when the feed is empty).
    pub first_seq: Option<u64>,
    /// The next sequence to be written (== total rows ever written).
    pub next_seq: u64,
    /// Retained rows.
    pub retained: u64,
}

impl Estate {
    /// The changefeed's shape: oldest retained seq, next seq, row count.
    pub fn feed_stats(&self) -> Result<FeedStats> {
        let handle = self.db.cf(crate::keys::CF_FEED)?;
        let mut first_seq = None;
        let mut retained = 0u64;
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (k, _) = item.map_err(rocks_err)?;
            if first_seq.is_none() && k.len() == 8 {
                first_seq = Some(u64::from_be_bytes(k[..8].try_into().expect("8 bytes")));
            }
            retained += 1;
        }
        Ok(FeedStats {
            first_seq,
            next_seq: self.db.get_u64(crate::keys::META_FEED_SEQ)?,
            retained,
        })
    }
}
