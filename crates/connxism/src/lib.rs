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
//! [`ConnXRecall`] implements [`rrf_core::Recall`] over the estate: dense
//! cosine search, lexical BM25 search, and **hybrid** search fused by
//! reciprocal rank fusion. The flow plugs it in exactly like the in-memory
//! store — persistence is a component choice, not an architecture change.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod estate;
pub mod index;
pub mod keys;
pub mod model;
mod pending;
mod rels;
mod store;

pub use estate::Estate;
pub use index::{Bm25Params, Posting, Postings};
pub use model::{
    now_ms, now_ns, Change, ChangeOp, ConnectorInfo, ConnectorKind, EstateInfo, NodeInfo, Shape,
    StoredDoc, SyncState, SyncStatus, Transport, TrendPoint, WarpPoint,
};
pub use rels::{Relation, TraversalSpec};
pub use store::ConnXRecall;

/// Re-export so downstream crates can name the trait without a second dep.
pub use rrf_core::Recall;
