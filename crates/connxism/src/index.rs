//! The persistent BM25 inverted index.
//!
//! Postings live in the `terms` column family as **one row per
//! (term, document)**: key `term \x00 doc_id`, value a single [`Posting`].
//! Writes are blind puts — no read-modify-write — and reads are sorted prefix
//! scans, so indexing cost stays flat as hot terms grow. Entries carry the
//! document token length so lexical scoring never fetches payloads.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// One document's entry in a term's postings list.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Posting {
    /// Term frequency in the document.
    pub tf: u32,
    /// The document's content-token length (BM25 `dl`).
    pub len: u32,
}

/// A term's fetched postings: `(doc id, posting)` rows, unique by doc id.
pub type Postings = Vec<(String, Posting)>;

/// Okapi BM25 parameters.
#[derive(Debug, Clone, Copy)]
pub struct Bm25Params {
    /// Term-frequency saturation.
    pub k1: f32,
    /// Length normalization.
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Bm25Params { k1: 1.2, b: 0.75 }
    }
}

/// Score every document matching any query term. `n_docs` and `avgdl` come
/// from the estate counters; `term_postings` is the fetched postings per term.
pub fn bm25_scores(
    params: Bm25Params,
    n_docs: u64,
    avgdl: f32,
    term_postings: &[(String, Postings)],
) -> HashMap<String, f32> {
    let n = n_docs.max(1) as f32;
    let avgdl = avgdl.max(1.0);
    let mut scores: HashMap<String, f32> = HashMap::new();

    for (_term, postings) in term_postings {
        let df = postings.len() as f32;
        if df == 0.0 {
            continue;
        }
        // BM25 idf with +0.5 smoothing, clamped non-negative.
        let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln().max(0.0);
        for (doc_id, p) in postings.iter() {
            let f = p.tf as f32;
            let dl = p.len as f32;
            let denom = f + params.k1 * (1.0 - params.b + params.b * dl / avgdl);
            let s = idf * (f * (params.k1 + 1.0)) / denom;
            *scores.entry(doc_id.clone()).or_insert(0.0) += s;
        }
    }
    scores
}

/// Reciprocal rank fusion: fuse ranked lists into one ranking.
///
/// `score(d) = Σ_lists 1 / (k + rank_of_d_in_list)` with ranks starting at 1;
/// documents absent from a list contribute nothing for it. The standard
/// constant is `k = 60`.
pub fn reciprocal_rank_fusion(lists: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    reciprocal_rank_fusion_weighted(lists, &[], k)
}

/// Reciprocal Rank Fusion, **weighted per list**.
///
/// `weights[i]` scales list `i`'s vote; a missing or short `weights` implies
/// 1.0 (so this is a strict generalization of [`reciprocal_rank_fusion`]).
///
/// ## Why the weight has to exist
///
/// Plain RRF gives every list an equal vote, which silently assumes every
/// retriever is equally good. When they are not, fusion drags the strong arm
/// toward the weak one: on nfcorpus, dense scores nDCG@10 **0.4119** and BM25
/// **0.3283**, and unweighted fusion of the two lands at **0.3943** — *below
/// dense alone*. The mechanism is visible in the arithmetic: a BM25-only hit at
/// rank 1 contributes `1/(60+1) = 0.01639`, beating a dense hit at rank 2 at
/// `1/(60+2) = 0.01613`. The weaker retriever outvotes the stronger one on its
/// own turf.
///
/// That is not evidence that hybrid retrieval hurts — it is a missing
/// parameter. Every production hybrid engine exposes this knob (Qdrant/Vespa
/// fusion weights, Elastic's `rank_constant`+boosts, Weaviate's `alpha`); RRO
/// shipped RRF without one, then measured the consequence and nearly recorded
/// it as a property of fusion.
///
/// Weights are **not** defaulted to a tuned value: the caller owns the corpus,
/// and a weight tuned on nfcorpus is not a weight for your data. The default
/// stays 1.0/1.0 (identical to plain RRF). The knob lives on
/// [`rro_core::EstateQuery::fusion`] — fusion is a **per-query** decision, not a
/// property of the estate.
pub fn reciprocal_rank_fusion_weighted(
    lists: &[Vec<String>],
    weights: &[f32],
    k: f32,
) -> Vec<(String, f32)> {
    let mut fused: HashMap<String, f32> = HashMap::new();
    for (li, list) in lists.iter().enumerate() {
        let w = weights.get(li).copied().unwrap_or(1.0);
        if w == 0.0 {
            continue; // A zero-weight list is an ablation, not a tie-breaker.
        }
        for (i, id) in list.iter().enumerate() {
            *fused.entry(id.clone()).or_insert(0.0) += w / (k + (i as f32 + 1.0));
        }
    }
    let mut out: Vec<(String, f32)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_prefers_agreement() {
        // "b" is ranked well by both lists; "a" and "c" only by one each.
        let lists = vec![vec!["a".into(), "b".into()], vec!["b".into(), "c".into()]];
        let fused = reciprocal_rank_fusion(&lists, 60.0);
        assert_eq!(fused[0].0, "b");
    }

    /// The regression, reproduced in six lines.
    ///
    /// This is the exact mechanism behind "hybrid scores 0.3943 vs dense
    /// 0.4119" on nfcorpus. `dense_only` is the better retriever's #2 pick;
    /// `lex_only` is the worse retriever's #1. Unweighted, the worse retriever
    /// wins — which is how a weak arm drags a strong one down, one rank at a
    /// time, across 323 queries.
    #[test]
    fn unweighted_fusion_lets_a_weak_list_outvote_a_strong_one() {
        let dense = vec!["shared".into(), "dense_only".into()];
        let lexical = vec!["lex_only".into(), "shared".into()];
        let lists = vec![dense, lexical];

        let fused = reciprocal_rank_fusion(&lists, 60.0);
        let rank = |id: &str| fused.iter().position(|(d, _)| d == id).unwrap();
        assert!(
            rank("lex_only") < rank("dense_only"),
            "unweighted RRF ranks the weak list's #1 above the strong list's #2 \
             — 1/61 > 1/62. THIS is the 'fusion regression'; it is a missing \
             weight, not a property of fusion."
        );
    }

    /// And the knob actually answers it: tell the fusion that dense is the
    /// better retriever, and the strong arm's pick comes back up.
    #[test]
    fn weighting_the_stronger_list_restores_its_ranking() {
        let dense = vec!["shared".into(), "dense_only".into()];
        let lexical = vec!["lex_only".into(), "shared".into()];
        let lists = vec![dense, lexical];

        let fused = reciprocal_rank_fusion_weighted(&lists, &[2.0, 1.0], 60.0);
        let rank = |id: &str| fused.iter().position(|(d, _)| d == id).unwrap();
        assert!(
            rank("dense_only") < rank("lex_only"),
            "with dense weighted 2:1 its #2 must outrank the weak list's #1"
        );
        assert_eq!(fused[0].0, "shared", "agreement must still win outright");
    }

    /// Weights must generalize plain RRF exactly, or the knob silently changes
    /// every existing caller's results the day it lands.
    #[test]
    fn unit_weights_are_identical_to_plain_rrf() {
        let lists = vec![
            vec!["a".into(), "b".into(), "c".into()],
            vec!["c".into(), "d".into()],
        ];
        let plain = reciprocal_rank_fusion(&lists, 60.0);
        let weighted = reciprocal_rank_fusion_weighted(&lists, &[1.0, 1.0], 60.0);
        assert_eq!(plain, weighted);

        // A short/absent weight vector implies 1.0 — the default path.
        assert_eq!(plain, reciprocal_rank_fusion_weighted(&lists, &[], 60.0));
    }

    /// Weight 0 is how an ablation asks for "dense only" without a second code
    /// path. It must drop the list entirely, not merely shrink its vote.
    #[test]
    fn zero_weight_ablates_a_list_completely() {
        let lists = vec![vec!["a".into()], vec!["b".into()]];
        let fused = reciprocal_rank_fusion_weighted(&lists, &[1.0, 0.0], 60.0);
        assert_eq!(fused.len(), 1, "a zero-weighted list must not contribute");
        assert_eq!(fused[0].0, "a");
    }

    #[test]
    fn bm25_scores_rarer_terms_higher() {
        let common: Postings = (0..50)
            .map(|i| (format!("d{i}"), Posting { tf: 1, len: 10 }))
            .collect();
        let rare: Postings = vec![("d0".into(), Posting { tf: 1, len: 10 })];

        let scores = bm25_scores(
            Bm25Params::default(),
            100,
            10.0,
            &[("common".into(), common), ("rare".into(), rare)],
        );
        // d0 matched both terms; its score must beat any common-only doc.
        let d0 = scores["d0"];
        let d1 = scores["d1"];
        assert!(d0 > d1);
    }
}
