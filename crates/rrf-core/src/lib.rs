//! # rrf-core
//!
//! The shared contract of the Reason Ready engine: the domain vocabulary and
//! the four component traits ([`Embedder`], [`Recall`], [`Reranker`],
//! [`Classifier`]). Every other crate in the workspace depends on this one and
//! nothing else of ours — it is the single source of truth for the flow.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod error;
pub mod events;
pub mod text;
pub mod traits;
pub mod types;

pub use error::{Result, RrfError};
pub use traits::{Classifier, Embedder, Recall, Reranker, VectorRecord};
pub use types::{
    Candidate, Chunk, Document, Embedding, Id, Metadata, Query, Readiness, RecallResult,
};
