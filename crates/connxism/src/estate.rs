//! The estate: one RocksDB, fully managed.
//!
//! [`Estate`] owns the database and everything registry-shaped: estate info,
//! nodes + warp points, connectors + sync state, tags, shape census, and trend
//! series. The recall store ([`crate::ConnXRecall`]) shares the same database
//! through a cheap clone of the inner handle.

use std::path::Path;
use std::sync::{Arc, RwLock as StdRwLock};

use recall::{AnnConfig, AnnIndex};
use rocksdb::{ColumnFamily, Options, DB};
use rrf_core::{Embedding, Id, Result, RrfError};

use crate::keys::{
    self, CF_CONNS, CF_META, CF_NODES, CF_TAGS, CF_TRENDS, CF_VECS, COLUMN_FAMILIES,
};
use crate::model::{now_ms, ConnectorInfo, EstateInfo, NodeInfo, SyncState, TrendPoint, WarpPoint};

/// Shared handle to the open database. Cloneable; all clones see one DB.
#[derive(Clone)]
pub(crate) struct Db(pub(crate) Arc<DB>);

impl Db {
    pub(crate) fn cf(&self, name: &str) -> Result<&ColumnFamily> {
        self.0
            .cf_handle(name)
            .ok_or_else(|| RrfError::Recall(format!("missing column family `{name}`")))
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

/// Map RocksDB errors into the engine error type.
pub(crate) fn rocks_err(e: rocksdb::Error) -> RrfError {
    RrfError::Recall(format!("kvs: {e}"))
}

/// Open-time choices for an estate.
#[derive(Debug, Clone, Default)]
pub struct EstateConfig {
    /// Hold the ANN graph's vectors as SQ8 codes (~4× smaller memory).
    /// Search results are **rescored exactly** from the durable vector
    /// column family, so scores at the API stay exact — quantization is a
    /// memory decision, not an accuracy decision.
    pub quantized: bool,
    /// The text analyzer for the lexical (BM25) index. **Fixed at estate
    /// creation** — it is part of the index's identity (postings and
    /// queries must agree on what a token is); reopening ignores this
    /// field in favour of the persisted one.
    pub analyzer: rrf_core::text::Analyzer,
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
    /// Whether the graph stores SQ8 codes (search paths rescore exactly).
    pub(crate) quantized: bool,
    /// Fired after every committed changefeed append (upsert/remove), so
    /// push-stream watchers wake event-driven instead of polling.
    pub(crate) feed_notify: Arc<tokio::sync::Notify>,
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

        let db = DB::open_cf(&opts, path.as_ref(), COLUMN_FAMILIES).map_err(rocks_err)?;
        let db = Db(Arc::new(db));

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
                fresh
            }
        };

        // Rebuild the ANN graph from durable vectors.
        let mut ann = AnnIndex::new(AnnConfig {
            quantized: config.quantized,
            ..AnnConfig::default()
        });
        {
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
        }

        let ann = Arc::new(StdRwLock::new(ann));
        let pending = crate::pending::Pending::new();
        let applier = pending.spawn_applier(ann.clone());

        Ok(Estate {
            db,
            ann,
            pending,
            quantized: config.quantized,
            feed_notify: Arc::new(tokio::sync::Notify::new()),
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
            .ok_or_else(|| RrfError::msg(format!("no such node: {node_id}")))?;
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
            .ok_or_else(|| RrfError::msg(format!("no such connector: {connector_id}")))?;
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
        self.db.0.write(batch).map_err(rocks_err)?;
        rrf_core::events::emit(
            "estate.payload_index",
            serde_json::json!({ "field": field, "rows": rows }),
        );
        Ok(())
    }

    /// The payload-indexed field names.
    pub fn payload_indexes(&self) -> Result<Vec<String>> {
        crate::filter::indexed_fields(&self.db)
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
    /// fully working estate via [`Estate::open`]; the ANN graph rebuilds from
    /// its durable vectors as on any open.
    pub fn snapshot_to(&self, path: impl AsRef<Path>) -> Result<()> {
        let checkpoint = rocksdb::checkpoint::Checkpoint::new(&self.db.0).map_err(rocks_err)?;
        checkpoint
            .create_checkpoint(path.as_ref())
            .map_err(rocks_err)?;
        rrf_core::events::emit(
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
        // Stop the applier cleanly; unapplied pendings are already durable in
        // the vecs column family and reappear via rebuild-on-open.
        self.pending.stop();
        if let Some(handle) = self.applier.take() {
            let _ = handle.join();
        }
    }
}
