//! Fusion is a property of the QUERY, not of the estate.
//!
//! This gate exists because of a measured, nearly-published mistake. RRO's RRF
//! had **no weight at all**: every ranked list got an equal vote, which silently
//! assumes every retriever is equally good. On nfcorpus dense scores nDCG@10
//! 0.4120 and BM25 0.3283, and equal-vote fusion lands at **0.3943 — below dense
//! alone**. That was almost written up as "hybrid fusion hurts", a finding about
//! the architecture. It was a missing parameter.
//!
//! The weight also has to live on the query rather than the collection, because
//! the right weight is a property of *what is being asked*: "find `E0521`" wants
//! the lexical arm, a paraphrase question does not. A single estate-wide
//! constant cannot serve both, and tuning one on a benchmark is just fitting the
//! test set.
//!
//! See `docs/BENCHMARKS_REAL.md` Finding 1 for the measurement.

use connxism::{Estate, EstateQuery, HybridWeights};
use rro_core::{Embedding, Recall, VectorRecord};

/// A corpus where the two retrievers **disagree on purpose**.
///
/// `lexical_star` is a perfect BM25 match for the query terms but points away
/// from the query vector. `dense_star` is the opposite. Fusion's weight is the
/// only thing that decides which one wins — which is exactly what we want to
/// observe.
///
/// The filler is not padding. RRF at `k=60` compresses ranks hard — 1/61 vs
/// 1/64 is a 5% difference — so on a tiny corpus *every* document lands in
/// *both* lists at nearly the same score, and fusion correctly rewards the
/// doc that appears in both. That drowns the signal under test. The corpus has
/// to be big enough that the dense list is genuinely selective and can exclude
/// `lexical_star`.
async fn disagreeing_corpus(estate: &Estate) -> connxism::ConnXRecall {
    let recall = estate.recall();
    let mut recs = vec![
        // Aligned with the query vector, but shares no query terms.
        VectorRecord::new(
            "dense_star",
            Embedding(vec![1.0, 0.0, 0.0]).normalized(),
            "unrelated wording entirely",
        ),
        // Stuffed with the query terms, but ORTHOGONAL to the query vector —
        // so it sorts last on the dense side and out of the dense top-k.
        VectorRecord::new(
            "lexical_star",
            Embedding(vec![0.0, 1.0, 0.0]).normalized(),
            "quantum entanglement quantum entanglement quantum",
        ),
    ];
    // Fillers span the cosine range between the two stars: closer to the query
    // than `lexical_star`, further than `dense_star`.
    for i in 0..40 {
        let t = 0.1 + (i as f32) * 0.02; // 0.10 ..= 0.88
        recs.push(VectorRecord::new(
            format!("filler_{i}"),
            Embedding(vec![t, 1.0 - t, 0.0]).normalized(),
            format!("document {i} about gardening bicycles and weather"),
        ));
    }
    recall.upsert(recs).await.unwrap();
    recall
}

fn query_with(fusion: HybridWeights) -> EstateQuery {
    EstateQuery {
        text: Some("quantum entanglement".to_string()),
        vector: Some(Embedding(vec![1.0, 0.0, 0.0]).normalized()),
        top_k: 10,
        fusion,
        ..Default::default()
    }
}

fn rank_of(hits: &[rro_core::Candidate], id: &str) -> Option<usize> {
    hits.iter().position(|c| c.id.as_str() == id)
}

/// THE gate: the weight on the query must actually reach the fusion.
///
/// A knob that compiles but is never read is worse than no knob — it reads as a
/// feature and behaves as a default.
#[tokio::test(flavor = "multi_thread")]
async fn the_query_weight_reaches_the_fusion() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fusion").unwrap();
    let recall = disagreeing_corpus(&estate).await;

    let dense_heavy = recall
        .query(query_with(HybridWeights {
            dense: 20.0,
            lexical: 1.0,
        }))
        .await
        .unwrap();
    let lexical_heavy = recall
        .query(query_with(HybridWeights {
            dense: 1.0,
            lexical: 20.0,
        }))
        .await
        .unwrap();

    assert_eq!(
        rank_of(&dense_heavy, "dense_star"),
        Some(0),
        "weighted 20:1 toward dense, the dense match must win — if it doesn't, \
         EstateQuery::fusion is not reaching the fusion"
    );
    assert_eq!(
        rank_of(&lexical_heavy, "lexical_star"),
        Some(0),
        "weighted 1:20 toward lexical, the BM25 match must win"
    );
}

/// The default must remain plain RRF, or this knob landing silently rewrote
/// every existing caller's rankings.
#[tokio::test(flavor = "multi_thread")]
async fn the_default_is_still_plain_rrf() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fusion").unwrap();
    let recall = disagreeing_corpus(&estate).await;

    let defaulted = recall
        .query(query_with(HybridWeights::default()))
        .await
        .unwrap();
    let explicit_1_1 = recall
        .query(query_with(HybridWeights {
            dense: 1.0,
            lexical: 1.0,
        }))
        .await
        .unwrap();

    let ids = |h: &[rro_core::Candidate]| -> Vec<String> {
        h.iter().map(|c| c.id.as_str().to_string()).collect()
    };
    assert_eq!(
        ids(&defaulted),
        ids(&explicit_1_1),
        "HybridWeights::default() must be exactly 1:1"
    );
}

/// A zero weight is how an ablation asks for "dense only" without a second code
/// path. It must drop the arm entirely, not merely shrink its vote — that is
/// what makes the `hybrid:w*` sweep in `rro-eval` mean what it says.
#[tokio::test(flavor = "multi_thread")]
async fn zero_weight_ablates_an_arm() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fusion").unwrap();
    let recall = disagreeing_corpus(&estate).await;

    let dense_only = recall
        .query(query_with(HybridWeights {
            dense: 1.0,
            lexical: 0.0,
        }))
        .await
        .unwrap();

    // With the lexical arm silenced, the pure-BM25 hit has no route into the
    // ranking above the dense winner.
    assert_eq!(
        rank_of(&dense_only, "dense_star"),
        Some(0),
        "lexical weight 0 must leave the dense ranking untouched at the top"
    );
}
