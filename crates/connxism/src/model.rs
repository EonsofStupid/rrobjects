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
    pub fn of(metadata: &rrf_core::Metadata) -> Self {
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
    pub metadata: rrf_core::Metadata,
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
}
