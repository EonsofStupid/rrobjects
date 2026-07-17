//! # classifier
//!
//! The Reason Ready daemon: judges whether retrieved context is sufficient to
//! reason on, behind the [`rro_core::Classifier`] trait.
//!
//! - [`HeuristicClassifier`] — weightless coverage-based default. Runs today.
//! - [`ReasonReadyDaemon`] — runs any classifier as an embedded, message-driven
//!   service (the shape the tuned DevPULSE classifier will run in).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod constrained;
mod daemon;
mod heuristic;

pub use daemon::{DaemonHandle, ReasonReadyDaemon};
pub use heuristic::HeuristicClassifier;

pub use constrained::{ConstrainedClassifier, ConstrainedConfig, ReadyLabel};
/// Re-export so downstream crates can name the trait without a second dep.
pub use rro_core::Classifier;
