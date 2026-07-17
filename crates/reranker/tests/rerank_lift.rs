//! The reranker gate (docs/MODELS.md §4): a reranker must **lift** top-k
//! relevance versus not reranking. If it doesn't, that is a real finding and
//! this test says so rather than being quietly dropped.
//!
//! `#[ignore]` — needs live servers:
//!
//! ```sh
//! RRO_TEST_RERANK_LLAMACPP=http://127.0.0.1:8093/v1/rerank \
//! RRO_TEST_RERANK_VLLM=http://127.0.0.1:8092/rerank \
//!   cargo test -p reranker --test rerank_lift -- --ignored --nocapture
//! ```

use reranker::{HttpRerankConfig, HttpRerankKind, HttpReranker, LexicalReranker};
use rro_core::{Candidate, Reranker};

/// A small golden set: each query has exactly one correct document among
/// distractors that share vocabulary with it. Lexical overlap alone should
/// struggle; a cross-encoder should not.
struct Case {
    query: &'static str,
    gold: &'static str,
    docs: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case {
        query: "What is the capital of China?",
        gold: "beijing",
        docs: &[
            // Distractors deliberately loaded with query words.
            "China is a large country. This document is about the capital markets of Asia.",
            "The capital of France is Paris, not China.",
            "What is a capital? A capital is the seat of government of a country.",
            "The capital of China is Beijing.", // gold, last on purpose
        ],
    },
    Case {
        query: "How do plants make food from sunlight?",
        gold: "photosynthesis",
        docs: &[
            "Plants need food and sunlight to grow well in a garden.",
            "Sunlight is made of photons. Food is made in kitchens.",
            "Photosynthesis is the process by which plants convert light energy into chemical \
             energy, producing glucose from carbon dioxide and water.",
            "How do you make plant food at home from kitchen scraps?",
        ],
    },
];

fn gold_index(c: &Case) -> usize {
    match c.gold {
        "beijing" => 3,
        "photosynthesis" => 2,
        _ => unreachable!(),
    }
}

fn candidates(c: &Case) -> Vec<Candidate> {
    c.docs
        .iter()
        .enumerate()
        .map(|(i, t)| Candidate::new(format!("d{i}"), *t, 0.0))
        .collect()
}

/// golden@1: did the correct document end up first?
async fn golden_at_1(r: &dyn Reranker) -> f32 {
    let mut hits = 0.0;
    for c in CASES {
        let out = r.rerank(c.query, candidates(c), 4).await.unwrap();
        let want = format!("d{}", gold_index(c));
        let got = out[0].id.as_str();
        println!("    q={:?}\n      top1={:?}", c.query, out[0].text);
        if got == want {
            hits += 1.0;
        }
    }
    hits / CASES.len() as f32
}

async fn http(var: &str, kind: HttpRerankKind) -> Option<HttpReranker> {
    let ep = std::env::var(var).ok().filter(|s| !s.trim().is_empty())?;
    match HttpReranker::connect(HttpRerankConfig::new(&ep, kind)).await {
        Ok(r) => Some(r),
        // Set-but-broken must fail, never skip.
        Err(e) => panic!("{var}={ep} is set but connecting failed: {e}"),
    }
}

/// The BM25 floor. Recorded, not asserted: it is the baseline the cross-encoders
/// must beat, and its value is a fact about the corpus, not a pass/fail.
#[tokio::test]
async fn lexical_baseline_golden_at_1() {
    let r = LexicalReranker::new();
    println!("  lexical (BM25):");
    let score = golden_at_1(&r).await;
    println!("  => BM25 golden@1 = {score:.2}");
}

#[tokio::test]
#[ignore]
async fn llamacpp_reranker_lifts_over_bm25() {
    let Some(r) = http("RRO_TEST_RERANK_LLAMACPP", HttpRerankKind::LlamaCpp).await else {
        eprintln!("SKIP: set RRO_TEST_RERANK_LLAMACPP");
        return;
    };
    let bm25 = golden_at_1(&LexicalReranker::new()).await;
    println!("  llamacpp ({}):", r.model_name());
    let ce = golden_at_1(&r).await;
    println!("  => BM25 {bm25:.2} -> llamacpp {ce:.2}");
    assert!(
        ce >= bm25,
        "the cross-encoder ({ce}) did WORSE than BM25 ({bm25}) — that is a real finding, \
         not a flaky test: report it rather than deleting this assertion"
    );
    assert!(
        ce >= 1.0,
        "cross-encoder should rank every gold first, got {ce}"
    );
}

#[tokio::test]
#[ignore]
async fn vllm_reranker_lifts_over_bm25() {
    let Some(r) = http("RRO_TEST_RERANK_VLLM", HttpRerankKind::Vllm).await else {
        eprintln!("SKIP: set RRO_TEST_RERANK_VLLM");
        return;
    };
    let bm25 = golden_at_1(&LexicalReranker::new()).await;
    println!("  vllm ({}):", r.model_name());
    let ce = golden_at_1(&r).await;
    println!("  => BM25 {bm25:.2} -> vllm {ce:.2}");
    assert!(
        ce >= bm25,
        "vLLM cross-encoder ({ce}) did worse than BM25 ({bm25})"
    );
    assert!(
        ce >= 1.0,
        "cross-encoder should rank every gold first, got {ce}"
    );
}

/// Both engines serve the same model (llama-nemotron-rerank-1b-v2) on this box,
/// so they must agree on the ORDER even though their score scales differ wildly
/// (vLLM normalizes to [0,1]; llama.cpp returns raw logits like 18.68 / -11.89).
#[tokio::test]
#[ignore]
async fn llamacpp_and_vllm_agree_on_order() {
    let (Some(l), Some(v)) = (
        http("RRO_TEST_RERANK_LLAMACPP", HttpRerankKind::LlamaCpp).await,
        http("RRO_TEST_RERANK_VLLM", HttpRerankKind::Vllm).await,
    ) else {
        eprintln!("SKIP: set both RRO_TEST_RERANK_LLAMACPP and RRO_TEST_RERANK_VLLM");
        return;
    };
    for c in CASES {
        let lo = l.rerank(c.query, candidates(c), 4).await.unwrap();
        let vo = v.rerank(c.query, candidates(c), 4).await.unwrap();
        let lord: Vec<&str> = lo.iter().map(|c| c.id.as_str()).collect();
        let vord: Vec<&str> = vo.iter().map(|c| c.id.as_str()).collect();
        println!(
            "  q={:?}\n    llamacpp={lord:?}\n    vllm    ={vord:?}",
            c.query
        );
        assert_eq!(
            lord[0], vord[0],
            "same model, same query, but the engines disagree on the top document"
        );
    }
}

/// The candle cross-encoder: Qwen3-Reranker via yes/no token logits.
///
/// This is the third engine for rerank, and the one MODELS.md §4.2 would have
/// led astray — see `reranker::candle_qwen` for why the "relevance logit"
/// recipe does not apply to a causal-LM reranker.
///
/// ```sh
/// RRO_TEST_QWEN_RERANK_WEIGHTS=/path/to/qwen3-reranker-0-6b \
///   cargo test -p reranker --features candle --test rerank_lift -- --ignored --nocapture
/// ```
#[cfg(feature = "candle")]
#[tokio::test]
#[ignore]
async fn candle_reranker_does_not_regress_vs_bm25() {
    let Ok(dir) = std::env::var("RRO_TEST_QWEN_RERANK_WEIGHTS") else {
        eprintln!("SKIP: set RRO_TEST_QWEN_RERANK_WEIGHTS");
        return;
    };
    let r = reranker::CandleQwenReranker::load(reranker::CandleRerankConfig::new(&dir))
        .expect("load Qwen3-Reranker weights");
    let bm25 = golden_at_1(&LexicalReranker::new()).await;
    println!("  candle ({}):", r.model_name());
    let ce = golden_at_1(&r).await;
    println!("  => BM25 {bm25:.2} -> candle {ce:.2}");

    // NAME NOTE: this asserts NO REGRESSION, not lift, and is named for what it
    // checks. It was `candle_reranker_lifts_over_bm25`, which claimed a gate the
    // assertion does not enforce — exactly the theatre the finding below rejects.
    //
    // RECORDED FINDING (2026-07-16, qwen3-reranker-0-6b): candle scores 0.50
    // here — no lift. This is the MODEL, not the backend. Proof:
    //   * case 1 separates cleanly: 0.9995 / 0.3200 / 0.0727 / 0.0005
    //   * calibration is near-perfect: relevant 0.9995 vs irrelevant 0.000036
    //   * on case 2 it SATURATES: gold 0.989082 loses to 0.989714 by 0.0006,
    //     and a nonsense distractor ("Sunlight is made of photons. Food is
    //     made in kitchens.") still scores 0.942. Everything topically
    //     adjacent pins near 0.99 — classic small-cross-encoder behaviour.
    // llama-nemotron-rerank-1b-v2 lifts the same set 0.50 -> 1.00.
    //
    // So this test asserts NO REGRESSION vs BM25, not a lift. A 2-case set
    // cannot gate lift at all (n=2; 0.50 vs 1.00 is one document moving), and
    // pretending it can would be theatre. The real lift gate is the BRIGHT/TREC
    // eval at scale, where the 0.6/4/8B tier ladder gets decided on evidence.
    assert!(
        ce >= bm25,
        "the candle cross-encoder ({ce}) ranked WORSE than BM25 ({bm25}) — that is a real \
         regression in the backend, not a model-capacity finding: investigate before relaxing"
    );
}

/// Scores must be probabilities: the yes/no softmax means P(yes) in [0,1], and
/// a relevant doc must score far above an irrelevant one. A raw-logit leak or a
/// stacked-backwards [yes,no] would still "rank" but produce nonsense values.
#[cfg(feature = "candle")]
#[tokio::test]
#[ignore]
async fn candle_reranker_scores_are_calibrated_probabilities() {
    let Ok(dir) = std::env::var("RRO_TEST_QWEN_RERANK_WEIGHTS") else {
        eprintln!("SKIP: set RRO_TEST_QWEN_RERANK_WEIGHTS");
        return;
    };
    let r = reranker::CandleQwenReranker::load(reranker::CandleRerankConfig::new(&dir)).unwrap();
    let out = r
        .rerank(
            "What is the capital of China?",
            vec![
                Candidate::new("good", "The capital of China is Beijing.", 0.0),
                Candidate::new(
                    "bad",
                    "Bananas are a tropical fruit rich in potassium.",
                    0.0,
                ),
            ],
            2,
        )
        .await
        .unwrap();
    for c in &out {
        println!("  {} -> P(yes)={:.6}", c.id.as_str(), c.score);
        assert!(
            (0.0..=1.0).contains(&c.score),
            "score {} is not a probability — is the softmax stacked [no,yes]?",
            c.score
        );
    }
    assert_eq!(out[0].id.as_str(), "good", "relevant doc must win");
    assert!(
        out[0].score > 0.5 && out[1].score < 0.5,
        "expected a confident split, got {:.4} vs {:.4}",
        out[0].score,
        out[1].score
    );
}

/// Diagnostic: print P(yes) for every candidate on the case candle gets wrong.
/// Is the model weak, or is the implementation broken?
#[cfg(feature = "candle")]
#[tokio::test]
#[ignore]
async fn candle_score_dump() {
    let Ok(dir) = std::env::var("RRO_TEST_QWEN_RERANK_WEIGHTS") else {
        return;
    };
    let r = reranker::CandleQwenReranker::load(reranker::CandleRerankConfig::new(&dir)).unwrap();
    for c in CASES {
        println!("\n  QUERY: {:?}   (gold = d{})", c.query, gold_index(c));
        let out = r.rerank(c.query, candidates(c), 4).await.unwrap();
        for cand in &out {
            let mark = if cand.id.as_str() == format!("d{}", gold_index(c)) {
                " <-- GOLD"
            } else {
                ""
            };
            println!(
                "    {:>10.6}  {}  {:?}{mark}",
                cand.score,
                cand.id.as_str(),
                &cand.text[..cand.text.len().min(60)]
            );
        }
    }
}
