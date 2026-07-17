//! The Reason Ready flow: the one pass that ties the components together.

/// Intent routing — RRO's own stage, with no counterpart in clyffy's `stage::`
/// vocabulary. It is post-embed *routing* (`route_tags` over the query vector),
/// which is neither `shape` (pre-model) nor `recall`. Named as an extension
/// rather than folded into a clyffy stage that means something else.
const ST_INTENT: &str = "intent";

use std::sync::Arc;

use classifier::HeuristicClassifier;
use connectome::{Connectome, ConnectomeGraph};
use embedder::DeterministicEmbedder;
use recall::FlatRecall;
use reranker::LexicalReranker;
use rro_core::{
    Classifier, Document, Embedder, Recall, RecallResult, Reranker, Result, VectorRecord,
};

/// How wide each stage runs.
#[derive(Debug, Clone)]
pub struct ObjectConfig {
    /// Candidates pulled from recall (vector) before reranking.
    pub recall_k: usize,
    /// Candidates kept after reranking (and handed to the classifier).
    pub rerank_k: usize,
}

impl Default for ObjectConfig {
    fn default() -> Self {
        ObjectConfig {
            recall_k: 20,
            rerank_k: 5,
        }
    }
}

/// The assembled engine: **RRD first**, then embedder → recall → reranker →
/// classifier → connectome.
pub struct ReasonReadyObject {
    rrd: Option<Arc<rrd::Rrd>>,
    embedder: Arc<dyn Embedder>,
    recall: Arc<dyn Recall>,
    reranker: Arc<dyn Reranker>,
    classifier: Arc<dyn Classifier>,
    connectome: Connectome,
    config: ObjectConfig,
}

impl ReasonReadyObject {
    /// Start building a flow.
    pub fn builder() -> ObjectBuilder {
        ObjectBuilder::new()
    }

    /// The default, weightless engine: deterministic embedder, flat recall,
    /// BM25 reranker, heuristic reason-ready classifier. Runs today, no weights.
    pub fn default_engine() -> Self {
        ObjectBuilder::new().build()
    }

    /// Index documents: embed each, then upsert into recall. Returns the new
    /// total record count.
    pub async fn index(&self, docs: Vec<Document>) -> Result<usize> {
        if docs.is_empty() {
            return self.recall.len().await;
        }
        let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
        let embeddings = self.embedder.embed_documents(&texts).await?;
        let records: Vec<VectorRecord> = docs
            .into_iter()
            .zip(embeddings)
            .map(|(d, e)| {
                let mut r = VectorRecord::new(d.id, e, d.text);
                r.metadata = d.metadata;
                r
            })
            .collect();
        self.recall.upsert(records).await?;
        self.recall.len().await
    }

    /// Run one full pass for `query`: embed → recall → rerank → classify.
    ///
    /// Recall is *hybrid*: stores that maintain a lexical index fuse dense and
    /// lexical rankings (reciprocal rank fusion); pure vector stores fall back
    /// to dense search via the trait's default.
    pub async fn ask(&self, query: &str) -> Result<RecallResult> {
        use std::time::Instant;
        let pass = Instant::now();
        // One id, carried by every signal this pass emits, so the stream can be
        // replayed into exactly THIS turn. Without it, concurrent queries
        // interleave and only aggregates are readable.
        let turn = rro_core::TurnId::next();
        let stage = |name: &str, since: Instant, fields: serde_json::Value| {
            rro_core::emit_stage(turn, name, since, fields);
        };
        // Stage names come from `rro_core::semconv`, which mirrors clyffy's
        // vocabulary: RRD's gate IS clyffy's `shape` stage, and the readiness
        // verdict IS its `reason` stage. Emitting "rrd"/"classify" instead meant
        // the same pipeline had two names depending on who was reading.
        use rro_core::semconv::stage as st;
        rro_core::emit_turn(
            turn,
            "flow.open",
            serde_json::json!({ "query": query, "chars": query.chars().count() }),
        );

        // RRD is literally the instant first thing: stamp + gate ladder on
        // the query BEFORE any model cost. A blocked query never reaches the
        // embedder — it returns gated, with the verdict on the record.
        let mut query_rro = None;
        if let Some(rrd) = &self.rrd {
            let t = Instant::now();
            let stamp = rrd::SourceStamp {
                channel: Some("query".to_string()),
                ..rrd::SourceStamp::default()
            };
            let rro = rrd.distill_stamped("query", query, &rro_core::Metadata::new(), None, stamp);
            stage(
                st::SHAPE,
                t,
                serde_json::json!({
                    "gate": rro.gate,
                    "mode": rro.mode.name(),
                    "sliver": rro.sliver_id,
                    "signals": rro.signals,
                }),
            );
            if rro.gate == rrd::GateVerdict::Block {
                // A block is the most interesting turn in the stream: it is the
                // engine refusing to spend a model call. Close the turn so the
                // refusal is as visible as a success — a turn that just stops
                // reads as a crash.
                rro_core::emit_turn(
                    turn,
                    "flow.turn",
                    serde_json::json!({
                        "total_ms": pass.elapsed().as_micros() as f64 / 1000.0,
                        "gated": true,
                        "ready": false,
                        "candidates": 0,
                        "model_calls": 0,
                    }),
                );
                return Ok(RecallResult {
                    turn,
                    query: query.to_string(),
                    candidates: Vec::new(),
                    readiness: rro_core::Readiness::not_ready(
                        1.0,
                        "gated",
                        "blocked by the RRD deterministic gate before any model ran",
                    ),
                    intent: Vec::new(),
                });
            }
            query_rro = Some(rro);
        }

        let t = Instant::now();
        let q = self.embedder.embed_query_one(query).await?;
        stage(st::EMBED, t, serde_json::json!({ "dim": q.dim() }));

        // Intent: the L2 half of the query's distillation, on the embedding
        // we just paid for anyway.
        let t = Instant::now();
        let intent: Vec<String> = match (&self.rrd, &query_rro) {
            (Some(rrd), Some(_)) => rrd.route_tags(&q).into_iter().map(|t| t.tag).collect(),
            _ => Vec::new(),
        };
        // Intent was computed and never emitted — invisible in the stream, so a
        // routed turn looked identical to an unrouted one.
        stage(ST_INTENT, t, serde_json::json!({ "tags": intent }));

        let t = Instant::now();
        let recalled = self
            .recall
            .hybrid_search(query, &q, self.config.recall_k)
            .await?;
        stage(
            st::RECALL,
            t,
            serde_json::json!({
                "candidates": recalled.len(),
                "top": recalled.iter().take(3).map(|c| c.id.as_str()).collect::<Vec<_>>(),
            }),
        );

        let t = Instant::now();
        let ranked = self
            .reranker
            .rerank(query, recalled, self.config.rerank_k)
            .await?;
        // Which docs, not just how many: "rerank changed the answer" is the
        // claim, and only the ids can show it.
        stage(
            st::RERANK,
            t,
            serde_json::json!({
                "kept": ranked.len(),
                "top": ranked.iter().take(3).map(|c| c.id.as_str()).collect::<Vec<_>>(),
            }),
        );

        let t = Instant::now();
        let readiness = self.classifier.classify(query, &ranked).await?;
        stage(
            st::REASON,
            t,
            serde_json::json!({
                "ready": readiness.ready,
                "confidence": readiness.confidence,
                "label": readiness.label,
                "rationale": readiness.rationale,
            }),
        );

        rro_core::emit_turn(
            turn,
            "flow.turn",
            serde_json::json!({
                // Sub-ms passes exist (the gate, a warm local store), and
                // as_millis() rounded them to 0 — a stage that reports 0 is a
                // stage nobody profiles.
                "total_ms": pass.elapsed().as_micros() as f64 / 1000.0,
                "gated": false,
                "ready": readiness.ready,
                "confidence": readiness.confidence,
                "candidates": ranked.len(),
                "intent": intent,
            }),
        );

        Ok(RecallResult {
            turn,
            query: query.to_string(),
            candidates: ranked,
            readiness,
            intent,
        })
    }

    /// Build the visual map for a completed pass.
    pub fn connectome(&self, result: &RecallResult) -> ConnectomeGraph {
        self.connectome
            .map(&result.query, &result.candidates, &result.readiness)
    }

    /// Convenience: run a pass and build its map in one call.
    pub async fn ask_with_map(&self, query: &str) -> Result<(RecallResult, ConnectomeGraph)> {
        let result = self.ask(query).await?;
        let map = self.connectome(&result);
        Ok((result, map))
    }

    /// Embed one query text with the flow's embedder (what the a2a `query`
    /// verb uses when a typed query arrives with text but no vector).
    pub async fn embed_query(&self, text: &str) -> Result<rro_core::Embedding> {
        self.embedder.embed_query_one(text).await
    }

    /// The active configuration.
    pub fn config(&self) -> &ObjectConfig {
        &self.config
    }

    /// Names of the active component models, for telemetry / the connectome.
    pub fn model_names(&self) -> [(&'static str, &str); 4] {
        [
            ("embedder", self.embedder.model_name()),
            ("recall", "flat-cosine"),
            ("reranker", self.reranker.model_name()),
            ("classifier", self.classifier.model_name()),
        ]
    }
}

/// Fluent builder for [`ReasonReadyObject`]. Any component left unset falls back
/// to its weightless default.
pub struct ObjectBuilder {
    rrd: Option<Arc<rrd::Rrd>>,
    embedder: Option<Arc<dyn Embedder>>,
    recall: Option<Arc<dyn Recall>>,
    reranker: Option<Arc<dyn Reranker>>,
    classifier: Option<Arc<dyn Classifier>>,
    config: ObjectConfig,
}

impl Default for ObjectBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectBuilder {
    /// A builder with all-default components.
    pub fn new() -> Self {
        ObjectBuilder {
            rrd: None,
            embedder: None,
            recall: None,
            reranker: None,
            classifier: None,
            config: ObjectConfig::default(),
        }
    }

    /// Attach RRD as the flow's front door (query gating + intent routing).
    pub fn rrd(mut self, rrd: Arc<rrd::Rrd>) -> Self {
        self.rrd = Some(rrd);
        self
    }

    /// Override the embedder (e.g. the DevPULSE / Qwen model).
    pub fn embedder(mut self, e: Arc<dyn Embedder>) -> Self {
        self.embedder = Some(e);
        self
    }

    /// Override the recall store.
    pub fn recall(mut self, r: Arc<dyn Recall>) -> Self {
        self.recall = Some(r);
        self
    }

    /// Override the reranker (e.g. the DevPULSE / Nemotron model).
    pub fn reranker(mut self, r: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(r);
        self
    }

    /// Override the reason-ready classifier.
    pub fn classifier(mut self, c: Arc<dyn Classifier>) -> Self {
        self.classifier = Some(c);
        self
    }

    /// Set the flow widths.
    pub fn config(mut self, config: ObjectConfig) -> Self {
        self.config = config;
        self
    }

    /// Assemble the flow, filling any unset component with its default.
    pub fn build(self) -> ReasonReadyObject {
        ReasonReadyObject {
            rrd: self.rrd,
            embedder: self
                .embedder
                .unwrap_or_else(|| Arc::new(DeterministicEmbedder::new()) as Arc<dyn Embedder>),
            recall: self
                .recall
                .unwrap_or_else(|| Arc::new(FlatRecall::new()) as Arc<dyn Recall>),
            reranker: self
                .reranker
                .unwrap_or_else(|| Arc::new(LexicalReranker::new()) as Arc<dyn Reranker>),
            classifier: self
                .classifier
                .unwrap_or_else(|| Arc::new(HeuristicClassifier::new()) as Arc<dyn Classifier>),
            connectome: Connectome::new(),
            config: self.config,
        }
    }
}
