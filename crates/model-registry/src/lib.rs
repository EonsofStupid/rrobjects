//! Selection is data, not code.
//!
//! This crate is the only place that turns *configuration* into a concrete
//! [`Embedder`] / [`Reranker`]. Everything downstream — the flow, the estate,
//! the query plane — depends on the trait and never on candle, ort, or any
//! model runtime. That is what makes the backends swappable: adding one is a
//! new enum arm plus a constructor, with zero flow changes.
//!
//! Three rules hold this shape (docs/MODELS.md §1):
//!
//! 1. **The trait is the only contract.** Real models drop in behind
//!    [`rro_core::Embedder`] / [`rro_core::Reranker`].
//! 2. **Selection is data.** [`EmbedderConfig`] / [`RerankerConfig`] are parsed
//!    from env; `RRO_EMBEDDER=candle-qwen` is the entire swap mechanism.
//! 3. **Performance lives inside the backend.** Batching, device placement,
//!    pooling, and quantization are the backend's business, behind the trait.
//!
//! Heavy backends sit behind the `candle` / `onnx` features so the default
//! workspace builds weightless. A kind whose feature is off is a **clear config
//! error**, never a silent fallback to the synthetic embedder — a silent
//! fallback is how you end up publishing synthetic numbers as if they were real.

#![deny(missing_docs)]

use std::sync::Arc;

use rro_core::{Embedder, Reranker, Result, RroError};

/// Where a model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Host CPU.
    #[default]
    Cpu,
    /// CUDA device by ordinal.
    Cuda(usize),
    /// Apple Metal.
    Metal,
}

impl std::str::FromStr for Device {
    type Err = RroError;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "cpu" => Ok(Device::Cpu),
            "metal" => Ok(Device::Metal),
            "cuda" => Ok(Device::Cuda(0)),
            other => match other.strip_prefix("cuda:") {
                Some(n) => n
                    .parse::<usize>()
                    .map(Device::Cuda)
                    .map_err(|_| config_err(format!("bad CUDA ordinal in device `{other}`"))),
                None => Err(config_err(format!(
                    "unknown device `{other}` (expected: cpu | cuda | cuda:<n> | metal)"
                ))),
            },
        }
    }
}

/// Which embedder backend to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EmbedderKind {
    /// Feature-hashing, weightless, non-semantic. The CI/default backend.
    #[default]
    Deterministic,
    /// Qwen-family embedding backbone via candle. Requires `candle` + weights.
    CandleQwen,
    /// llama.cpp `--embedding` server over OpenAI-compatible HTTP.
    LlamaCpp,
    /// vLLM OpenAI server over HTTP.
    Vllm,
    /// ONNX-exported embedder via `ort`. Requires `onnx` + weights.
    Onnx,
    /// Delegate to another RRO node's model over the a2a wire (MODELS.md §5).
    Remote,
}

impl EmbedderKind {
    /// The wire/env name of this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbedderKind::Deterministic => "deterministic",
            EmbedderKind::CandleQwen => "candle-qwen",
            EmbedderKind::LlamaCpp => "llamacpp",
            EmbedderKind::Vllm => "vllm",
            EmbedderKind::Onnx => "onnx",
            EmbedderKind::Remote => "remote",
        }
    }

    /// Every selectable kind, for error messages and `--help` output.
    pub const ALL: [EmbedderKind; 6] = [
        EmbedderKind::Deterministic,
        EmbedderKind::CandleQwen,
        EmbedderKind::LlamaCpp,
        EmbedderKind::Vllm,
        EmbedderKind::Onnx,
        EmbedderKind::Remote,
    ];
}

impl std::str::FromStr for EmbedderKind {
    type Err = RroError;

    fn from_str(s: &str) -> Result<Self> {
        match normalize(s).as_str() {
            "deterministic" => Ok(EmbedderKind::Deterministic),
            "candle-qwen" | "qwen" | "candle" => Ok(EmbedderKind::CandleQwen),
            "llamacpp" | "llama-cpp" | "llama" => Ok(EmbedderKind::LlamaCpp),
            "vllm" => Ok(EmbedderKind::Vllm),
            "onnx" => Ok(EmbedderKind::Onnx),
            "remote" => Ok(EmbedderKind::Remote),
            other => Err(unknown_kind(
                "RRO_EMBEDDER",
                other,
                &EmbedderKind::ALL.map(|k| k.as_str()),
            )),
        }
    }
}

/// Which reranker backend to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RerankerKind {
    /// BM25 lexical scoring. Weightless; the default.
    ///
    /// Sharp edge: it re-sorts *lexically*. Over a hybrid store — whose fusion
    /// already weighed BM25 once — it double-counts the lexical signal and
    /// re-sorts by the weaker retriever. Right for a dense-only store that wants
    /// lexical signal added; wrong as a reranker you forgot to configure.
    #[default]
    Lexical,
    /// Keep recall's ordering; take the top k. Weightless, free, and the only
    /// way to say "do not rerank" — the stage cannot be omitted, only filled.
    Identity,
    /// Cross-encoder (query,doc)->score via candle. Requires `candle` + weights.
    CandleCrossEncoder,
    /// llama.cpp `--reranking` server (`/v1/rerank`).
    LlamaCpp,
    /// vLLM `/rerank`.
    Vllm,
    /// ONNX-exported cross-encoder via `ort`. Requires `onnx` + weights.
    Onnx,
    /// Delegate to another RRO node over the a2a wire.
    Remote,
}

impl RerankerKind {
    /// The wire/env name of this kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            RerankerKind::Lexical => "lexical",
            RerankerKind::Identity => "identity",
            RerankerKind::CandleCrossEncoder => "candle-cross-encoder",
            RerankerKind::LlamaCpp => "llamacpp",
            RerankerKind::Vllm => "vllm",
            RerankerKind::Onnx => "onnx",
            RerankerKind::Remote => "remote",
        }
    }

    /// Every selectable kind.
    pub const ALL: [RerankerKind; 7] = [
        RerankerKind::Lexical,
        RerankerKind::Identity,
        RerankerKind::CandleCrossEncoder,
        RerankerKind::LlamaCpp,
        RerankerKind::Vllm,
        RerankerKind::Onnx,
        RerankerKind::Remote,
    ];
}

impl std::str::FromStr for RerankerKind {
    type Err = RroError;

    fn from_str(s: &str) -> Result<Self> {
        match normalize(s).as_str() {
            "lexical" | "bm25" => Ok(RerankerKind::Lexical),
            "identity" | "none" => Ok(RerankerKind::Identity),
            "candle-cross-encoder" | "candle-nemotron" | "candle-qwen" | "candle" => {
                Ok(RerankerKind::CandleCrossEncoder)
            }
            "llamacpp" | "llama-cpp" | "llama" => Ok(RerankerKind::LlamaCpp),
            "vllm" => Ok(RerankerKind::Vllm),
            "onnx" => Ok(RerankerKind::Onnx),
            "remote" => Ok(RerankerKind::Remote),
            other => Err(unknown_kind(
                "RRO_RERANKER",
                other,
                &RerankerKind::ALL.map(|k| k.as_str()),
            )),
        }
    }
}

/// How to build the embedder. Pure data — parsed from env or set by a caller.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Which backend.
    pub kind: EmbedderKind,
    /// Filesystem path to weights (a safetensors dir), for local backends.
    pub weights_path: Option<String>,
    /// Endpoint URL, for [`EmbedderKind::Remote`].
    pub endpoint: Option<String>,
    /// Output dimensionality. `None` = the backend's native dimension.
    ///
    /// Set it BELOW native only on a matryoshka-trained model (Qwen3-Embedding
    /// supports 32..=1024): the backend truncates and re-normalizes. On a model
    /// without MRL, truncating a vector is just corruption.
    pub dim: Option<usize>,
    /// Where the model runs.
    pub device: Device,
    /// Max texts per forward pass.
    pub batch: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        EmbedderConfig {
            kind: EmbedderKind::default(),
            weights_path: None,
            endpoint: None,
            dim: None,
            device: Device::default(),
            batch: DEFAULT_BATCH,
        }
    }
}

/// How to build the reranker. Pure data.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Which backend.
    pub kind: RerankerKind,
    /// Filesystem path to weights, for local backends.
    pub weights_path: Option<String>,
    /// Endpoint URL, for [`RerankerKind::Remote`].
    pub endpoint: Option<String>,
    /// Where the model runs.
    pub device: Device,
    /// Max (query,doc) pairs per forward pass.
    pub batch: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        RerankerConfig {
            kind: RerankerKind::default(),
            weights_path: None,
            endpoint: None,
            device: Device::default(),
            batch: DEFAULT_BATCH,
        }
    }
}

/// Default forward-pass batch size when `RRO_EMBED_BATCH` is unset.
pub const DEFAULT_BATCH: usize = 32;

impl EmbedderConfig {
    /// Read the embedder selection from the environment.
    ///
    /// | var | meaning |
    /// |---|---|
    /// | `RRO_EMBEDDER` | kind (default `deterministic`) |
    /// | `RRO_EMBEDDER_WEIGHTS` | weights dir |
    /// | `RRO_EMBEDDER_ENDPOINT` | URL, for `remote` |
    /// | `RRO_EMBED_DIM` | output dim (MRL truncation) |
    /// | `RRO_EMBED_BATCH` | batch size |
    /// | `RRO_DEVICE` | `cpu` \| `cuda` \| `cuda:<n>` \| `metal` |
    ///
    /// A malformed value is an error, not a shrug back to the default: a typo
    /// in `RRO_EMBEDDER` must not quietly hand back synthetic vectors.
    pub fn from_env() -> Result<Self> {
        Ok(EmbedderConfig {
            kind: parse_env("RRO_EMBEDDER")?.unwrap_or_default(),
            weights_path: var("RRO_EMBEDDER_WEIGHTS"),
            endpoint: var("RRO_EMBEDDER_ENDPOINT"),
            dim: parse_usize_env("RRO_EMBED_DIM")?,
            device: parse_env("RRO_DEVICE")?.unwrap_or_default(),
            batch: parse_usize_env("RRO_EMBED_BATCH")?.unwrap_or(DEFAULT_BATCH),
        })
    }
}

impl RerankerConfig {
    /// Read the reranker selection from the environment (`RRO_RERANKER`,
    /// `RRO_RERANKER_WEIGHTS`, `RRO_RERANKER_ENDPOINT`, `RRO_DEVICE`,
    /// `RRO_EMBED_BATCH`).
    pub fn from_env() -> Result<Self> {
        Ok(RerankerConfig {
            kind: parse_env("RRO_RERANKER")?.unwrap_or_default(),
            weights_path: var("RRO_RERANKER_WEIGHTS"),
            endpoint: var("RRO_RERANKER_ENDPOINT"),
            device: parse_env("RRO_DEVICE")?.unwrap_or_default(),
            batch: parse_usize_env("RRO_EMBED_BATCH")?.unwrap_or(DEFAULT_BATCH),
        })
    }
}

/// Build the configured embedder.
///
/// The weightless [`EmbedderKind::Deterministic`] arm always builds. Every
/// other arm needs its feature and its weights; when one is missing you get an
/// error naming exactly what to do about it.
pub async fn build_embedder(cfg: &EmbedderConfig) -> Result<Arc<dyn Embedder>> {
    match cfg.kind {
        EmbedderKind::Deterministic => Ok(Arc::new(match cfg.dim {
            Some(d) => embedder::DeterministicEmbedder::with_dim(d),
            None => embedder::DeterministicEmbedder::new(),
        })),

        EmbedderKind::CandleQwen => {
            #[cfg(feature = "candle")]
            {
                let dir = cfg.weights_path.as_deref().ok_or_else(|| {
                    config_err(
                        "RRO_EMBEDDER=candle-qwen needs RRO_EMBEDDER_WEIGHTS=<dir> containing \
                         model*.safetensors + config.json + tokenizer.json",
                    )
                })?;
                let mut qcfg = embedder::QwenEmbedConfig::new(dir);
                qcfg.device = to_candle_device(cfg.device)?;
                qcfg.batch = cfg.batch.max(1);
                qcfg.truncate_dim = cfg.dim;
                Ok(Arc::new(embedder::CandleQwenEmbedder::load(qcfg)?))
            }
            #[cfg(not(feature = "candle"))]
            {
                Err(feature_off("candle-qwen", "candle", "RRO_EMBEDDER"))
            }
        }

        EmbedderKind::Onnx => {
            #[cfg(feature = "onnx")]
            {
                Err(not_yet_wired("onnx", "docs/MODELS.md §5 (P-tail)"))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(feature_off("onnx", "onnx", "RRO_EMBEDDER"))
            }
        }

        EmbedderKind::LlamaCpp | EmbedderKind::Vllm => {
            let (kind, default_ep) = match cfg.kind {
                EmbedderKind::LlamaCpp => (
                    embedder::OpenAiKind::LlamaCpp,
                    "http://127.0.0.1:8090/v1/embeddings",
                ),
                _ => (
                    embedder::OpenAiKind::Vllm,
                    "http://127.0.0.1:8092/v1/embeddings",
                ),
            };
            let ep = cfg
                .endpoint
                .clone()
                .unwrap_or_else(|| default_ep.to_string());
            let mut ocfg = embedder::OpenAiEmbedConfig::new(ep, kind);
            ocfg.batch = cfg.batch.max(1);
            ocfg.truncate_dim = cfg.dim;
            Ok(Arc::new(embedder::OpenAiEmbedder::connect(ocfg).await?))
        }

        EmbedderKind::Remote => Err(not_yet_wired(
            "remote",
            "docs/MODELS.md §5 — delegates over the a2a client",
        )),
    }
}

/// Build the configured reranker. Same shape as [`build_embedder`].
pub async fn build_reranker(cfg: &RerankerConfig) -> Result<Arc<dyn Reranker>> {
    match cfg.kind {
        RerankerKind::Lexical => Ok(Arc::new(reranker::LexicalReranker::new())),
        RerankerKind::Identity => Ok(Arc::new(reranker::IdentityReranker::new())),

        RerankerKind::CandleCrossEncoder => {
            #[cfg(feature = "candle")]
            {
                let dir = cfg.weights_path.as_deref().ok_or_else(|| {
                    config_err(
                        "RRO_RERANKER=candle-cross-encoder needs RRO_RERANKER_WEIGHTS=<dir> \
                         containing a Qwen3-Reranker checkpoint (model*.safetensors + \
                         config.json + tokenizer.json)",
                    )
                })?;
                let mut rcfg = reranker::CandleRerankConfig::new(dir);
                rcfg.device = to_candle_device(cfg.device)?;
                rcfg.batch = cfg.batch.max(1);
                Ok(Arc::new(reranker::CandleQwenReranker::load(rcfg)?))
            }
            #[cfg(not(feature = "candle"))]
            {
                Err(feature_off(
                    "candle-cross-encoder",
                    "candle",
                    "RRO_RERANKER",
                ))
            }
        }

        RerankerKind::Onnx => {
            #[cfg(feature = "onnx")]
            {
                Err(not_yet_wired("onnx", "docs/MODELS.md §5 (P-tail)"))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(feature_off("onnx", "onnx", "RRO_RERANKER"))
            }
        }

        RerankerKind::LlamaCpp | RerankerKind::Vllm => {
            let (kind, default_ep) = match cfg.kind {
                RerankerKind::LlamaCpp => (
                    reranker::HttpRerankKind::LlamaCpp,
                    "http://127.0.0.1:8093/v1/rerank",
                ),
                _ => (
                    reranker::HttpRerankKind::Vllm,
                    "http://127.0.0.1:8092/rerank",
                ),
            };
            let ep = cfg
                .endpoint
                .clone()
                .unwrap_or_else(|| default_ep.to_string());
            let mut rcfg = reranker::HttpRerankConfig::new(ep, kind);
            rcfg.batch = cfg.batch.max(1);
            Ok(Arc::new(reranker::HttpReranker::connect(rcfg).await?))
        }

        RerankerKind::Remote => Err(not_yet_wired(
            "remote",
            "docs/MODELS.md §5 — delegates over the a2a client",
        )),
    }
}

/// Map the registry's device selection onto candle's.
///
/// Failing here (no CUDA driver, bad ordinal) is a startup error naming the
/// device, not a silent fall back to CPU: a "fast" run that quietly used the
/// CPU is a lie about what was measured.
#[cfg(feature = "candle")]
fn to_candle_device(d: Device) -> Result<candle_core::Device> {
    match d {
        Device::Cpu => Ok(candle_core::Device::Cpu),
        Device::Cuda(i) => candle_core::Device::new_cuda(i)
            .map_err(|e| config_err(format!("CUDA device {i} unavailable: {e}"))),
        Device::Metal => candle_core::Device::new_metal(0)
            .map_err(|e| config_err(format!("Metal device unavailable: {e}"))),
    }
}

// ---- error construction: every message says what to DO ----------------------

fn config_err(msg: impl Into<String>) -> RroError {
    RroError::Config(msg.into())
}

fn unknown_kind(var_name: &str, got: &str, known: &[&str]) -> RroError {
    config_err(format!(
        "unknown {var_name} `{got}` (expected one of: {})",
        known.join(" | ")
    ))
}

// Called only from the `#[cfg(not(feature = ...))]` arms of `build_*`. When
// every model feature is compiled in (e.g. CI's `--all-features`), those arms
// vanish and this is genuinely unused — allow it there rather than under a
// blanket `#[allow]` that would also hide a real dead-code regression.
#[cfg_attr(all(feature = "candle", feature = "onnx"), allow(dead_code))]
fn feature_off(kind: &str, feature: &str, var_name: &str) -> RroError {
    config_err(format!(
        "{var_name}=`{kind}` needs the `{feature}` feature, which this binary was not built with — \
         rebuild with `--features {feature}`, or select a weightless kind. \
         Refusing to fall back to the deterministic embedder: it is synthetic, and silently \
         substituting it would report fake retrieval quality as real."
    ))
}

fn not_yet_wired(kind: &str, spec: &str) -> RroError {
    config_err(format!(
        "backend `{kind}` is not implemented yet — see {spec}. \
         It is selectable so the seam is real, but it has no forward pass; \
         it errors here rather than pretending to embed."
    ))
}

// ---- env helpers ------------------------------------------------------------

fn normalize(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace('_', "-")
}

fn var(name: &str) -> Option<String> {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => Some(v),
        _ => None,
    }
}

fn parse_env<T: std::str::FromStr<Err = RroError>>(name: &str) -> Result<Option<T>> {
    match var(name) {
        Some(v) => v.parse::<T>().map(Some),
        None => Ok(None),
    }
}

fn parse_usize_env(name: &str) -> Result<Option<usize>> {
    match var(name) {
        Some(v) => v
            .trim()
            .parse::<usize>()
            .map(Some)
            .map_err(|_| config_err(format!("{name} must be a positive integer, got `{v}`"))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The P7.1 gate: the weightless default builds with no features, no
    // weights, and no network.
    #[tokio::test]
    async fn deterministic_builds_weightless_and_embeds() {
        let e = build_embedder(&EmbedderConfig::default())
            .await
            .expect("deterministic must build");
        let v = e.embed_one("the cat sat on the mat").await.unwrap();
        assert_eq!(v.dim(), 384);
        assert_eq!(e.model_name(), "deterministic-hash");
    }

    #[tokio::test]
    async fn deterministic_honors_requested_dim() {
        let cfg = EmbedderConfig {
            dim: Some(128),
            ..Default::default()
        };
        assert_eq!(build_embedder(&cfg).await.unwrap().dim(), 128);
    }

    #[tokio::test]
    async fn lexical_reranker_builds_weightless() {
        assert!(build_reranker(&RerankerConfig::default()).await.is_ok());
    }

    // The other half of the gate: an unknown kind is a clear error.
    #[test]
    fn unknown_kind_is_a_clear_error() {
        let err = "qwen4000".parse::<EmbedderKind>().unwrap_err().to_string();
        assert!(err.contains("qwen4000"), "names the bad value: {err}");
        assert!(err.contains("deterministic"), "lists valid kinds: {err}");
    }

    #[test]
    fn kind_parsing_is_forgiving_about_shape_not_meaning() {
        assert_eq!(
            "  Candle_Qwen ".parse::<EmbedderKind>().unwrap(),
            EmbedderKind::CandleQwen
        );
        assert_eq!(
            "BM25".parse::<RerankerKind>().unwrap(),
            RerankerKind::Lexical
        );
    }

    #[test]
    fn device_parses_including_ordinal() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("cuda".parse::<Device>().unwrap(), Device::Cuda(0));
        assert_eq!("cuda:3".parse::<Device>().unwrap(), Device::Cuda(3));
        assert!("cuda:x".parse::<Device>().is_err());
        assert!("tpu".parse::<Device>().is_err());
    }

    // A backend that cannot run must never resolve to the synthetic embedder:
    // that is the exact mechanism that produced fake accuracy numbers.
    #[tokio::test]
    async fn unavailable_backend_errors_instead_of_falling_back() {
        for kind in [
            EmbedderKind::CandleQwen,
            EmbedderKind::Onnx,
            EmbedderKind::Remote,
        ] {
            let cfg = EmbedderConfig {
                kind,
                ..Default::default()
            };
            let err = match build_embedder(&cfg).await {
                Ok(_) => panic!("{} must not build in a weightless workspace", kind.as_str()),
                Err(e) => e.to_string(),
            };
            assert!(
                !err.is_empty() && err.contains(kind.as_str()),
                "error names the kind: {err}"
            );
        }
    }
}
