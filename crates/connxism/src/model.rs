//! The estate ontology: estates, nodes, warp points, connectors, sync state,
//! tags, shapes, and trends.
//!
//! The relationship model the connectome renders:
//! an **operator** shares a **connector** (a third-party source — mail, drive,
//! documents, a database — usually fronting a large data repo); ingestion pulls
//! it into the **estate**; **nodes** (agent endpoints) get **layer-2 a2a warp
//! points** so the operator-facing host works seamlessly behind the scenes.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Milliseconds since the Unix epoch.
pub type EpochMs = u64;

/// Current wall-clock time in epoch milliseconds.
pub fn now_ms() -> EpochMs {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Current wall-clock time in epoch **nanoseconds** (u64 — good to year 2554).
///
/// Trend keys use nanoseconds: two samples in the same millisecond must be two
/// points, not an overwrite.
pub fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Top-level metadata for an estate (one estate == one RocksDB).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstateInfo {
    /// Estate id (stable).
    pub id: String,
    /// Human name.
    pub name: String,
    /// Creation timestamp.
    pub created_at: EpochMs,
    /// Vector dimensionality, fixed by the first upsert.
    pub dim: Option<usize>,
    /// Per-named-space dimensionality, each fixed by the first vector
    /// written under that name.
    #[serde(default)]
    pub named_dims: std::collections::BTreeMap<String, usize>,
    /// The lexical analyzer this estate's postings were built with (fixed
    /// at creation; the serde default keeps pre-analyzer estates on the
    /// legacy pipeline they were indexed with).
    #[serde(default)]
    pub analyzer: rro_core::text::Analyzer,
}

/// How a warp point is reached.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    /// Same process — the in-proc a2a bus.
    Local,
    /// Raw TCP (newline-delimited JSON a2a).
    Tcp,
    /// An MCP mesh endpoint.
    Mcp,
}

/// A layer-2 a2a jump target for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpPoint {
    /// Transport used to reach the node.
    pub transport: Transport,
    /// Transport-specific address (socket addr, MCP URI, or local bus id).
    pub address: String,
    /// Capabilities advertised at this warp point (e.g. `ask`, `map`, `ingest`).
    pub capabilities: Vec<String>,
}

/// An agent/compute endpoint registered in the estate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node id (stable within the estate).
    pub id: String,
    /// Human name.
    pub name: String,
    /// The node's warp points, in preference order.
    pub warp_points: Vec<WarpPoint>,
    /// Last time this node was seen/updated.
    pub last_seen: EpochMs,
}

/// Broad category of a connector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnectorKind {
    /// Mailboxes.
    Mail,
    /// File/drive storage.
    Drive,
    /// Document collections.
    Docs,
    /// A location / filesystem path.
    Location,
    /// A third-party application.
    Application,
    /// A database.
    Database,
    /// Anything else.
    Custom,
}

/// Where a connector's sync currently stands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "detail")]
pub enum SyncStatus {
    /// Registered, never synced.
    Registered,
    /// A sync pass is running.
    Syncing,
    /// Synced and idle.
    Idle,
    /// The last sync failed.
    Error(String),
}

/// Sync bookkeeping for a connector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// Provider-specific resume cursor (page token, WAL LSN, mtime…).
    pub cursor: Option<String>,
    /// Documents ingested from this connector so far.
    pub docs_synced: u64,
    /// Completion time of the last successful sync.
    pub last_sync: Option<EpochMs>,
    /// Current status.
    pub status: SyncStatus,
}

impl Default for SyncState {
    fn default() -> Self {
        SyncState {
            cursor: None,
            docs_synced: 0,
            last_sync: None,
            status: SyncStatus::Registered,
        }
    }
}

/// A third-party information source shared by an operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorInfo {
    /// Connector id (stable within the estate).
    pub id: String,
    /// Human name ("Work mail", "Team drive").
    pub name: String,
    /// Broad category.
    pub kind: ConnectorKind,
    /// Concrete provider slug (free-form: which mail/drive/database it is).
    pub provider: String,
    /// Source URI/locator (redacted-safe; no secrets).
    pub uri: String,
    /// Sync bookkeeping.
    pub sync: SyncState,
    /// Registration time.
    pub registered_at: EpochMs,
}

/// The schema/modality fingerprint of a document: field name → JSON type.
///
/// Shapes let the estate see *what kinds* of content it holds ("mail with
/// from/subject/body", "row with amount:number") without inspecting payloads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Shape(pub BTreeMap<String, String>);

impl Shape {
    /// Fingerprint a metadata map: field name → JSON type name.
    pub fn of(metadata: &rro_core::Metadata) -> Self {
        let mut m = BTreeMap::new();
        for (k, v) in metadata {
            let ty = match v {
                serde_json::Value::Null => "null",
                serde_json::Value::Bool(_) => "bool",
                serde_json::Value::Number(_) => "number",
                serde_json::Value::String(_) => "string",
                serde_json::Value::Array(_) => "array",
                serde_json::Value::Object(_) => "object",
            };
            m.insert(k.clone(), ty.to_string());
        }
        Shape(m)
    }

    /// Canonical key for grouping identical shapes (stable field order).
    pub fn key(&self) -> String {
        let parts: Vec<String> = self.0.iter().map(|(k, t)| format!("{k}:{t}")).collect();
        parts.join(",")
    }
}

/// What a changefeed entry records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeOp {
    /// A document was inserted or overwritten.
    Upsert,
    /// A document was removed.
    Remove,
}

/// One durable changefeed entry (written atomically with the change itself).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Change {
    /// Monotonic sequence number — the resume cursor.
    pub seq: u64,
    /// What happened.
    pub op: ChangeOp,
    /// The document affected.
    pub doc_id: String,
    /// When.
    pub at: EpochMs,
}

/// One **replication** entry: a changefeed row carried together with the payload
/// a follower needs to reproduce the write. The lean [`Change`] carries only the
/// id (enough for a live subscriber to re-fetch); a follower rebuilding a whole
/// estate needs the record itself, so replication ships it inline.
///
/// `record` is `Some` for an [`ChangeOp::Upsert`] whose document still exists at
/// read time, and `None` for a [`ChangeOp::Remove`] (or an upsert already
/// superseded by a later remove — the follower converges to the leader's *current*
/// state, and a later entry reconciles the gap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplEntry {
    /// The changefeed sequence — the follower's resume cursor.
    pub seq: u64,
    /// What happened.
    pub op: ChangeOp,
    /// The affected document id.
    pub doc_id: String,
    /// The record to apply on an upsert. `None` for a remove (or an upsert already
    /// superseded by a later remove).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record: Option<ReplRecord>,
}

/// The wire form of a replicated document: the **dense** record a follower needs
/// to reproduce an upsert. Explicitly dense — the internal `VectorRecord` also
/// carries sparse / named / multi-vector spaces, which Stage-1 replication does
/// not yet ship (a documented follow-on); making that a distinct type keeps the
/// wire honest about what crosses it rather than serializing a fuller record than
/// is actually filled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplRecord {
    /// Document id.
    pub id: String,
    /// The dense embedding.
    pub embedding: rro_core::Embedding,
    /// Raw text.
    pub text: String,
    /// Structured metadata.
    #[serde(default)]
    pub metadata: rro_core::Metadata,
    /// The named collection, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
}

impl ReplRecord {
    /// Rebuild the upsert-ready [`rro_core::VectorRecord`] on the follower. Tags,
    /// shape, token length and postings are re-derived by the estate on upsert —
    /// exactly as they were on the leader — so the mirror is faithful for the
    /// dense path without shipping any of that derived state.
    pub fn into_record(self) -> rro_core::VectorRecord {
        let mut r = rro_core::VectorRecord::new(self.id, self.embedding, self.text);
        r.metadata = self.metadata;
        r.collection = self.collection;
        r
    }
}

/// One point in a metric's time-series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendPoint {
    /// Sample time.
    pub at: EpochMs,
    /// Sample value.
    pub value: f64,
}

/// A document as stored in the estate (payload + tags + shape + stats).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredDoc {
    /// Document id.
    pub id: String,
    /// The text.
    pub text: String,
    /// Structured metadata.
    #[serde(default)]
    pub metadata: rro_core::Metadata,
    /// Tags attached to this document.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Shape fingerprint of `metadata`.
    #[serde(default)]
    pub shape: Shape,
    /// Content-token count (BM25 document length).
    pub token_len: u32,
    /// Connector this document came from, if ingested via one.
    #[serde(default)]
    pub connector_id: Option<String>,
    /// Sparse dimensions this document carries weights on (kept so an
    /// overwrite or removal can retract its sparse-postings rows exactly).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sparse_dims: Vec<u32>,
    /// Named vector spaces this document has vectors in (for exact
    /// retraction on overwrite/removal).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub named_spaces: Vec<String>,
    /// Number of late-interaction token vectors stored (0 = none).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub multi_len: u32,
    /// The named collection this document belongs to, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}
