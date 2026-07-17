//! # connXism — the kvs-connectome
//!
//! Real estate management for Reason Ready: one RocksDB per operator
//! **estate**, holding everything the engine knows —
//!
//! - **nodes** and their layer-2 a2a **warp points** (local / TCP / MCP mesh),
//! - **connectors** (third-party sources an operator shares — mail, drives,
//!   documents, databases) with resumable **sync state**,
//! - **documents**, **vectors**, and the persistent **BM25 inverted index**,
//! - **tags**, the **shape** census, and **trend** time-series.
//!
//! [`ConnXRecall`] implements [`rro_core::Recall`] over the estate: dense
//! cosine search, lexical BM25 search, and **hybrid** search fused by
//! reciprocal rank fusion. The flow plugs it in exactly like the in-memory
//! store — persistence is a component choice, not an architecture change.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod estate;
mod filter;
pub mod index;
pub mod keys;
pub mod model;
mod pending;
mod query;
mod rels;
mod store;
mod strategies;
mod txn;

pub use estate::{
    Estate, EstateConfig, FeedStats, GraphCompaction, GraphNodes, HealthReport, Issue, Quotas,
};
pub use recall::Quantizer;

/// How many column families one estate manages (ops surface sizing).
pub const COLUMN_FAMILY_COUNT: usize = keys::COLUMN_FAMILIES.len();
pub use index::{Bm25Params, Posting, Postings};
pub use model::{
    now_ms, now_ns, Change, ChangeOp, ConnectorInfo, ConnectorKind, EstateInfo, NodeInfo, Shape,
    StoredDoc, SyncState, SyncStatus, Transport, TrendPoint, WarpPoint,
};
pub use rels::{Relation, TraversalSpec};
/// Re-exported from the core contract so estate consumers keep one import.
pub use rro_core::{Condition, EstateQuery, Filter, FusionMode, HybridWeights};
pub use store::{ConnXRecall, WriteOp};
pub use strategies::Group;

/// Re-export so downstream crates can name the trait without a second dep.
pub use rro_core::Recall;
