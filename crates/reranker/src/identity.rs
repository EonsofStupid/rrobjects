//! The reranker that doesn't: keep recall's own ordering, take the top `k`.

use async_trait::async_trait;
use rro_core::{Candidate, Reranker, Result};

/// Keeps recall's ranking exactly as it arrived and truncates to `top_k`.
///
/// ## Why "no reranker" has to be a reranker
///
/// Every stage of the flow is a trait object, so there is no way to *omit* the
/// rerank stage — only to fill it. Until this existed, the choices were a real
/// cross-encoder (~1 s for 100 pairs) or [`crate::LexicalReranker`], and a
/// caller who wanted neither had no way to say so. "I trust my recall ordering"
/// was not expressible.
///
/// That gap had teeth, because the fallback was BM25. Reranking a **hybrid**
/// ranking with BM25 double-counts the lexical signal — the fusion already
/// weighed BM25 once — and re-sorts by the weaker retriever. Measured on
/// nfcorpus it drags the full pass from nDCG@10 0.3943 to 0.3199, *below plain
/// BM25 alone*. Observed live on a real estate, it is starker: the semantically
/// correct document scores **0.0000** and sinks, because it shares no words with
/// the query — which is the entire reason dense retrieval was used to find it.
///
/// Use this when recall's own ordering is the answer: dense-only stores, fused
/// hybrid rankings you trust, or a latency budget that cannot afford a
/// cross-encoder (a per-prompt memory hook, say). Use a real cross-encoder when
/// true relevance is worth ~1 s. Use [`crate::LexicalReranker`] when the store
/// is dense-only and you want lexical signal *added* — that is the one case
/// where it earns its place.
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityReranker;

impl IdentityReranker {
    /// Construct it. There is nothing to configure — that is the point.
    pub fn new() -> Self {
        IdentityReranker
    }
}

#[async_trait]
impl Reranker for IdentityReranker {
    async fn rerank(
        &self,
        _query: &str,
        mut candidates: Vec<Candidate>,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        candidates.truncate(top_k);
        Ok(candidates)
    }

    fn model_name(&self) -> &str {
        "identity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rro_core::Candidate;

    fn candidates() -> Vec<Candidate> {
        vec![
            Candidate::new("a", "alpha", 0.9),
            Candidate::new("b", "beta", 0.8),
            Candidate::new("c", "gamma", 0.7),
        ]
    }

    /// The whole contract: order in == order out.
    #[tokio::test]
    async fn ordering_survives_untouched() {
        let out = IdentityReranker::new()
            .rerank("anything at all", candidates(), 3)
            .await
            .unwrap();
        let ids: Vec<&str> = out.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(
            ids,
            ["a", "b", "c"],
            "recall's ranking must survive verbatim"
        );
    }

    /// It must still honour `top_k`, or it is not a stage — it is a no-op that
    /// silently returns more than the caller asked for.
    #[tokio::test]
    async fn top_k_is_honoured() {
        let out = IdentityReranker::new()
            .rerank("q", candidates(), 2)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id.as_str(), "a");
    }

    /// Scores are recall's, not ours. A reranker that rewrote them while
    /// claiming to preserve the ranking would make the numbers unreadable.
    #[tokio::test]
    async fn scores_are_left_alone() {
        let out = IdentityReranker::new()
            .rerank("q", candidates(), 3)
            .await
            .unwrap();
        assert_eq!(out[0].score, 0.9);
        assert_eq!(out[2].score, 0.7);
    }

    /// The query is deliberately ignored — that is the definition. Pinned so a
    /// future "small improvement" that starts reading it has to argue with a
    /// test first.
    #[tokio::test]
    async fn the_query_cannot_change_the_answer() {
        let a = IdentityReranker::new()
            .rerank("alpha alpha alpha", candidates(), 3)
            .await
            .unwrap();
        let b = IdentityReranker::new()
            .rerank("gamma gamma gamma", candidates(), 3)
            .await
            .unwrap();
        let ids = |v: &[Candidate]| -> Vec<String> {
            v.iter().map(|c| c.id.as_str().to_string()).collect()
        };
        assert_eq!(ids(&a), ids(&b));
    }

    #[tokio::test]
    async fn empty_in_empty_out() {
        let out = IdentityReranker::new()
            .rerank("q", vec![], 10)
            .await
            .unwrap();
        assert!(out.is_empty());
    }
}
