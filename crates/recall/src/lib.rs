//! # recall
//!
//! Dense vector memory for Reason Ready — the retrieval core, behind the
//! [`rro_core::Recall`] trait.
//!
//! [`FlatRecall`] is an exact in-memory store. It is the default engine; larger
//! deployments swap an ANN index in behind the same trait.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod ann;
mod flat;
pub mod quant;

pub use ann::{AnnConfig, AnnIndex, Quantizer};
pub use flat::FlatRecall;

/// Re-export so downstream crates can name the trait without a second dep.
pub use rro_core::Recall;
