//! # embedder
//!
//! Perception for Reason Ready: text → dense vectors, behind the
//! [`rro_core::Embedder`] trait.
//!
//! - [`DeterministicEmbedder`] — weightless feature-hashing. The CI/no-weights
//!   choice: it is **not semantic**, and it is never a silent fallback.
//! - [`CandleQwenEmbedder`] (`candle` feature) — a real Qwen3 embedding forward
//!   pass, in-process, no server. Takes any Qwen3-family weights directory,
//!   including a fine-tuned checkpoint of your own, via `RRO_EMBEDDER_WEIGHTS`.
//! - [`OpenAiEmbedder`] — llama.cpp or vLLM over OpenAI-compatible HTTP, for
//!   when the model should live in a server rather than in this process.
//!
//! All three satisfy [`rro_core::Embedder`], so which one is in use is a
//! configuration decision (see the `model-registry` crate) rather than a code
//! decision.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "candle")]
mod candle_qwen;
mod deterministic;
mod openai;
mod tokenize;

#[cfg(feature = "candle")]
pub use candle_qwen::{CandleQwenEmbedder, Qwen3Encoder, QwenEmbedConfig};
pub use deterministic::DeterministicEmbedder;
pub use openai::{OpenAiEmbedConfig, OpenAiEmbedder, OpenAiKind};

/// Re-export so downstream crates can name the trait without a second dep.
/// The instruction Qwen3-Embedding prepends to a **query** — never a document.
///
/// Lives at crate level because it is the model's contract, not one backend's
/// detail: candle and the OpenAI-compatible backends must apply the identical
/// prefix or their vectors are not comparable.
pub const DEFAULT_QUERY_TASK: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

pub use rro_core::Embedder;
