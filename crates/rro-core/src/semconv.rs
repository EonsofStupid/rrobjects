//! Semantic conventions — the field and stage vocabulary every signal uses.
//!
//! Two rules decide everything in this file:
//!
//! 1. **RRO owns its own namespace.** Every RRO-specific key is under `rro.`.
//! 2. **Where a standard exists, use the standard.** Model/token fields follow
//!    OpenTelemetry's `gen_ai.*` semantic conventions, so an off-the-shelf
//!    collector understands RRO without being taught anything.
//!
//! ## Why not just emit clyffy's names
//!
//! The first cut of this file mirrored clyffy's `clyffy-telemetry` vocabulary
//! **verbatim** — `devpulse.stage`, `devpulse.latency_ms` — and pinned it with a
//! test. That is backwards. RRO is the engine that gets pulled into other
//! systems; clyffy is one consumer. Emitting a consumer's private, branded keys
//! from the engine means every *other* consumer inherits vocabulary that names
//! something they've never heard of. Qdrant does not emit your project's
//! internal attribute names.
//!
//! The right seam is the one this project already uses everywhere else: the
//! **adapter maps at the boundary**. `clyffy-storage/src/adapters/rro/` renames
//! `rro.*` → `devpulse.*` on the way in — about thirty lines, in the one place
//! that knows what DevPULSE is.
//!
//! What RRO *does* keep from clyffy is the **patterns**, which are good and are
//! the actual thing worth aligning on: typed enums rather than loose strings,
//! short wire tags, an explicit cascade tier, and stage names
//! (`shape`/`embed`/`recall`/`rerank`/`reason`) that describe the pipeline
//! rather than the implementation. Those names are right on their own merits.

/// The event name for a pipeline-stage signal.
pub const EVENT_STAGE: &str = "rro.stage";

/// Canonical attribute keys.
pub mod attr {
    /// Which pipeline stage (a [`super::stage`] constant).
    pub const STAGE: &str = "rro.stage";
    /// The logical slot the stage filled (e.g. `recall.embedder`).
    pub const SLOT: &str = "rro.slot";
    /// Which concrete engine served it (a [`super::backend`] constant).
    pub const BACKEND: &str = "rro.backend";
    /// The resolved model id. Also emitted as [`GEN_AI_MODEL`].
    pub const MODEL: &str = "rro.model";
    /// Scope — `global` or a project slug.
    pub const SCOPE: &str = "rro.scope";
    /// The node that ran it.
    pub const NODE: &str = "rro.node";
    /// Stage wall-time, milliseconds.
    pub const LATENCY_MS: &str = "rro.latency_ms";
    /// Correlates every signal emitted by ONE pass. Without it the stream has
    /// aggregates but no turns.
    pub const TURN: &str = "rro.turn";
    /// The gate ladder's verdict (`pass` / `flag` / `block`).
    pub const GATE: &str = "rro.gate";
    /// Which cascade tier settled it — see [`super::Cascade`].
    pub const CASCADE: &str = "rro.cascade";
    /// Model calls this turn spent. A gated turn must report 0.
    pub const MODEL_CALLS: &str = "rro.model_calls";

    // ---- OpenTelemetry `gen_ai.*` — the standard, not our invention ---------

    /// OTel: the model a request asked for.
    pub const GEN_AI_MODEL: &str = "gen_ai.request.model";
    /// OTel: prompt tokens.
    pub const GEN_AI_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    /// OTel: completion tokens.
    pub const GEN_AI_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
}

/// Canonical stage names — the pipeline, named for what each step *does*.
pub mod stage {
    /// Intent → shaping. RRD's gate ladder lands here.
    pub const SHAPE: &str = "shape";
    /// Text → vectors.
    pub const EMBED: &str = "embed";
    /// Vector memory: ANN + BM25, fused.
    pub const RECALL: &str = "recall";
    /// True relevance.
    pub const RERANK: &str = "rerank";
    /// The reason-ready verdict. The classifier lands here.
    pub const REASON: &str = "reason";
}

/// Canonical backend names.
pub mod backend {
    /// llama.cpp server.
    pub const LLAMACPP: &str = "llama.cpp";
    /// vLLM server.
    pub const VLLM: &str = "vllm";
    /// candle, in-process.
    pub const CANDLE: &str = "candle";
    /// RRO itself — the fused map+vector engine.
    pub const RRO: &str = "rro";
    /// The weightless deterministic embedder (CI / no-weights).
    pub const DETERMINISTIC: &str = "deterministic";
    /// BM25 — lexical, no model.
    pub const LEXICAL: &str = "lexical";
}

/// Which tier settled a decision.
///
/// This is the distinction RRD's whole cost claim rests on: `Deterministic`
/// means arithmetic and lookups — **no model call**. Reporting it per turn is
/// what makes "the gate saves you money" auditable instead of asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cascade {
    /// Settled by a cheap deterministic signal — no LLM.
    Deterministic,
    /// Settled by the constrained-decode LLM pass.
    Reasoned,
}

impl Cascade {
    /// Both tiers.
    pub const ALL: [Cascade; 2] = [Cascade::Deterministic, Cascade::Reasoned];

    /// The wire/column tag.
    pub fn tag(self) -> &'static str {
        match self {
            Cascade::Deterministic => "deterministic",
            Cascade::Reasoned => "reasoned",
        }
    }
}

/// Which store a signal came from.
///
/// RRO fuses the map and the vector into one binary, so a single pass
/// legitimately produces both kinds. That is not a taxonomy violation — it is
/// the unification showing up in the telemetry, and it is why one RRO can
/// replace a separate graph engine plus a separate vector engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    /// Knowledge-graph: concepts + relations.
    Ksig,
    /// Vector: semantic recall.
    Vsig,
}

impl SignalKind {
    /// The short wire tag.
    pub fn tag(self) -> &'static str {
        match self {
            SignalKind::Ksig => "ksig",
            SignalKind::Vsig => "vsig",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The namespace rule, enforced. An engine that leaks a consumer's private
    /// vocabulary into everyone else's telemetry is a bug — and it is a bug that
    /// no compiler catches, so it gets a test.
    #[test]
    fn every_key_is_ours_or_a_standard() {
        let keys = [
            EVENT_STAGE,
            attr::STAGE,
            attr::SLOT,
            attr::BACKEND,
            attr::MODEL,
            attr::SCOPE,
            attr::NODE,
            attr::LATENCY_MS,
            attr::TURN,
            attr::GATE,
            attr::CASCADE,
            attr::MODEL_CALLS,
            attr::GEN_AI_MODEL,
            attr::GEN_AI_INPUT_TOKENS,
            attr::GEN_AI_OUTPUT_TOKENS,
        ];
        for k in keys {
            assert!(
                k.starts_with("rro.") || k.starts_with("gen_ai."),
                "`{k}` is neither RRO's own namespace nor an OTel standard — a \
                 consumer's private vocabulary must be mapped in that consumer's \
                 adapter, not emitted from the engine"
            );
        }
    }

    /// The `gen_ai.*` keys are copied from the OTel spec; a typo here silently
    /// makes RRO invisible to a standard collector.
    #[test]
    fn otel_keys_match_the_spec() {
        assert_eq!(attr::GEN_AI_MODEL, "gen_ai.request.model");
        assert_eq!(attr::GEN_AI_INPUT_TOKENS, "gen_ai.usage.input_tokens");
        assert_eq!(attr::GEN_AI_OUTPUT_TOKENS, "gen_ai.usage.output_tokens");
    }

    /// Stage names describe the pipeline, not the implementation that happens to
    /// fill each slot. Renaming one is an API break for anything querying the
    /// event stream, so pin them.
    #[test]
    fn the_pipeline_is_named_for_what_it_does() {
        assert_eq!(stage::SHAPE, "shape");
        assert_eq!(stage::EMBED, "embed");
        assert_eq!(stage::RECALL, "recall");
        assert_eq!(stage::RERANK, "rerank");
        assert_eq!(stage::REASON, "reason");
    }

    #[test]
    fn wire_tags_are_stable() {
        assert_eq!(Cascade::Deterministic.tag(), "deterministic");
        assert_eq!(Cascade::Reasoned.tag(), "reasoned");
        assert_eq!(SignalKind::Ksig.tag(), "ksig");
        assert_eq!(SignalKind::Vsig.tag(), "vsig");
    }
}
