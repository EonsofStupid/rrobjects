//! The constrained classifier against a REAL brain. `#[ignore]` — needs :8091.
//!
//! ```sh
//! RRO_TEST_BRAIN=http://127.0.0.1:8091/v1/chat/completions \
//!   cargo test -p classifier --test live_brain -- --ignored --nocapture
//! ```
//!
//! The gate is discrimination, not agreement-with-me: the judge must call
//! on-topic context ready and irrelevant context NOT ready. A classifier that
//! says "ready" to everything is worse than the heuristic it replaces, because
//! it costs a model call to learn nothing.

use classifier::{Classifier, ConstrainedClassifier, ConstrainedConfig, HeuristicClassifier};
use rro_core::Candidate;

fn ctx(texts: &[&str]) -> Vec<Candidate> {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| Candidate::new(format!("d{i}"), *t, 1.0 - i as f32 * 0.1))
        .collect()
}

async fn brain() -> Option<ConstrainedClassifier> {
    let ep = std::env::var("RRO_TEST_BRAIN")
        .ok()
        .filter(|s| !s.trim().is_empty())?;
    match ConstrainedClassifier::connect(ConstrainedConfig::new(&ep)).await {
        Ok(c) => Some(c),
        // Set-but-broken must FAIL, never skip: a green skip is how a real
        // failure hides.
        Err(e) => panic!("RRO_TEST_BRAIN={ep} is set but connecting failed: {e}"),
    }
}

#[tokio::test]
#[ignore]
async fn judges_sufficient_context_ready() {
    let Some(c) = brain().await else {
        eprintln!("SKIP: set RRO_TEST_BRAIN");
        return;
    };
    println!("classifier: {}", c.model_name());
    let r = c
        .classify(
            "What is the capital of China?",
            &ctx(&[
                "The capital of China is Beijing.",
                "Beijing is China's seat of government.",
            ]),
        )
        .await
        .unwrap();
    println!(
        "  ready={} label={} conf={:.4}\n  {}",
        r.ready, r.label, r.confidence, r.rationale
    );
    assert!(
        r.ready,
        "context directly answers the query but was judged {}",
        r.label
    );
    assert!(!r.rationale.is_empty(), "a verdict must carry its reason");
    assert!(r.confidence > 0.0, "logprobs must yield a real confidence");
}

#[tokio::test]
#[ignore]
async fn judges_irrelevant_context_not_ready() {
    let Some(c) = brain().await else {
        eprintln!("SKIP: set RRO_TEST_BRAIN");
        return;
    };
    let r = c
        .classify(
            "What is the capital of China?",
            &ctx(&[
                "Bananas are rich in potassium.",
                "The mitochondrion is the powerhouse of the cell.",
            ]),
        )
        .await
        .unwrap();
    println!(
        "  ready={} label={} conf={:.4}\n  {}",
        r.ready, r.label, r.confidence, r.rationale
    );
    assert!(
        !r.ready,
        "irrelevant context was judged ready — the classifier is a rubber stamp"
    );
    assert_eq!(
        r.label, "insufficient",
        "expected `insufficient`, got `{}`",
        r.label
    );
}

#[tokio::test]
#[ignore]
async fn empty_context_is_never_ready() {
    let Some(c) = brain().await else {
        eprintln!("SKIP: set RRO_TEST_BRAIN");
        return;
    };
    let r = c
        .classify("What is the capital of China?", &[])
        .await
        .unwrap();
    println!("  ready={} label={}", r.ready, r.label);
    assert!(!r.ready, "nothing was retrieved, yet it judged ready");
}

/// The claim this backend exists for: it must beat the heuristic somewhere the
/// heuristic is fooled. The heuristic scores lexical coverage, so context that
/// SHARES the query's words without answering it is exactly its blind spot.
#[tokio::test]
#[ignore]
async fn beats_the_heuristic_on_lexical_overlap_that_does_not_answer() {
    let Some(c) = brain().await else {
        eprintln!("SKIP: set RRO_TEST_BRAIN");
        return;
    };
    let query = "What is the capital of China?";
    // Loaded with the query's words; answers nothing.
    let decoys = ctx(&[
        "China is a large country in Asia with a long history.",
        "A capital is the city where a country's government sits.",
        "What is the capital of France? Paris is the capital of France.",
    ]);

    let h = HeuristicClassifier::new()
        .classify(query, &decoys)
        .await
        .unwrap();
    let m = c.classify(query, &decoys).await.unwrap();
    println!("  heuristic: ready={} label={}", h.ready, h.label);
    println!(
        "  model:     ready={} label={} conf={:.4}\n    {}",
        m.ready, m.label, m.confidence, m.rationale
    );

    assert!(
        !m.ready,
        "the model judged word-overlap-without-an-answer as ready ({}) — no lift \
         over the heuristic",
        m.label
    );
    if h.ready {
        println!("  => LIFT: the heuristic was fooled by lexical overlap; the model was not.");
    } else {
        println!("  => no lift here: the heuristic also refused. Recorded, not hidden.");
    }
}
