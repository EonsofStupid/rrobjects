//! # reranker
//!
//! True-relevance ordering over recall candidates, behind the
//! [`rro_core::Reranker`] trait.
//!
//! - [`IdentityReranker`] — keeps recall's ordering, takes the top k. "No
//!   rerank" has to be a reranker, because the stage cannot be omitted; until
//!   this existed, a caller who trusted their recall order had no way to say so.
//! - [`LexicalReranker`] — weightless Okapi BM25. The default, and a sharp edge:
//!   it re-sorts candidates *lexically*, so handing it a strong dense ranking
//!   drags that ranking back toward BM25. Measured: it took the full pass from
//!   nDCG@10 0.3943 to 0.3199 on nfcorpus, below plain BM25 itself. Use it as a
//!   floor, not as a reranker you forgot to configure.
//! - [`CandleQwenReranker`] (`candle` feature) — a real cross-encoder forward
//!   pass, in-process.
//! - [`HttpReranker`] — llama.cpp or vLLM behind `/v1/rerank`.
//!
//! All three satisfy [`rro_core::Reranker`], so which one is in use is a
//! configuration decision (see the `model-registry` crate) rather than a code
//! decision.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod bm25;
#[cfg(feature = "candle")]
mod candle_qwen;
mod http;
mod identity;

pub use bm25::LexicalReranker;
#[cfg(feature = "candle")]
pub use candle_qwen::{CandleQwenReranker, CandleRerankConfig, DEFAULT_RERANK_TASK};
pub use http::{HttpRerankConfig, HttpRerankKind, HttpReranker};
pub use identity::IdentityReranker;

/// Re-export so downstream crates can name the trait without a second dep.
pub use rro_core::Reranker;
