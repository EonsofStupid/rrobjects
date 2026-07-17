//! Phase 6a — the ANN graph is persisted, so a clean reopen *loads* it instead
//! of rebuilding it from scratch.
//!
//! The graph is a cache derived from the durable `vecs` column family. A clean
//! shutdown captures it to `CF_GRAPH` tagged with the changefeed seq it was taken
//! at; the next open loads that blob iff the seq still matches the live
//! `feed_seq`. This is what turns an O(N log N) rebuild-on-open into an O(read)
//! load — the "restart in read-time, not rebuild-time" property.
//!
//! Two things must hold, and both are gated here:
//!  1. **Loads, and is identical.** After a clean reopen the graph is loaded
//!     (`graph_was_loaded()`), and it returns the *same* search results as before
//!     the restart — a persisted graph that answered differently would be a
//!     silent corruption, worse than rebuilding.
//!  2. **Falls back safely.** If the persisted blob is stale (a crash left it
//!     behind newer writes), the open rebuilds from the durable vectors and every
//!     document — including the ones written after the stale capture — is present.
//!     The cache is never trusted over the source of truth.

use connxism::{Estate, EstateQuery};
use rro_core::{Embedding, Recall, VectorRecord};

const DIM: usize = 32;

/// Deterministic pseudo-random unit vector, stable across process restarts (no
/// RNG state, no time) so "identical before and after" is a meaningful claim.
fn vec_for(seed: u64) -> Embedding {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let v: Vec<f32> = (0..DIM)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect();
    Embedding(v).normalized()
}

fn corpus(n: usize) -> Vec<VectorRecord> {
    (0..n)
        .map(|i| VectorRecord::new(format!("d{i}"), vec_for(i as u64), format!("document {i}")))
        .collect()
}

/// The ordered `(id, score)` list a dense query returns — the comparison key for
/// "identical results". Scores are quantized to avoid f32 jitter masking a real
/// match while still catching a genuinely different ranking.
async fn ranked(recall: &connxism::ConnXRecall, q: &Embedding, k: usize) -> Vec<(String, i64)> {
    recall
        .query(EstateQuery::hybrid("document", q.clone(), k))
        .await
        .unwrap()
        .into_iter()
        .map(|c| (c.id.as_str().to_string(), (c.score * 10_000.0) as i64))
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn reopen_loads_persisted_graph_and_results_are_identical() {
    let dir = tempfile::tempdir().unwrap();
    // Seed well past the ANN threshold so the dense path is the graph, not a
    // brute-force scan — the graph is the thing being persisted.
    let n = 2000;
    let query = vec_for(9_999_001);

    let before = {
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let recall = estate.recall();
        recall.upsert(corpus(n)).await.unwrap();
        recall.quiesce().await.unwrap();
        let before = ranked(&recall, &query, 10).await;
        assert_eq!(before.len(), 10, "seed query must return a full page");
        // recall + estate drop here → clean shutdown persists the graph.
        before
    };

    let estate = Estate::open(dir.path(), "gp").unwrap();
    assert!(
        estate.graph_was_loaded(),
        "a clean reopen must LOAD the persisted graph, not rebuild it"
    );
    let recall = estate.recall();
    assert_eq!(recall.len().await.unwrap() as usize, n);

    let after = ranked(&recall, &query, 10).await;
    assert_eq!(
        before, after,
        "loaded graph must return byte-identical results to the graph it was saved from"
    );
}

/// The measured payoff: loading the persisted graph is dramatically faster than
/// rebuilding it by re-inserting every vector. `#[ignore]` — it seeds 50k vectors
/// (~seconds) to make the gap unmistakable; run with
/// `cargo test -p connxism --release --test graph_persist -- --ignored`.
///
/// This is the 6a half of the scale story (startup in read-time). The full
/// "10M vectors, restart < 5s" gate lands with 6b (mmap vectors), where the
/// vectors no longer have to fit in RAM at all.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "50k-vector timing gate; run under --release --ignored"]
async fn load_is_faster_than_rebuild() {
    use std::time::Instant;

    let dir = tempfile::tempdir().unwrap();
    let n = 50_000;

    // Seed once, capturing the graph on clean drop.
    let rebuild_ms = {
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let recall = estate.recall();
        recall.upsert(corpus(n)).await.unwrap();
        recall.quiesce().await.unwrap();
        // First reopen after a *rebuild-only* baseline: measure a from-scratch
        // rebuild by disabling persistence for this shutdown.
        estate.set_persist_graph_on_drop(false);
        drop(recall);
        drop(estate);

        let t = Instant::now();
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        assert!(!estate.graph_was_loaded(), "baseline must have rebuilt");
        // Now let this open persist the graph on drop for the load measurement.
        ms
    };

    let load_ms = {
        let t = Instant::now();
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        assert!(estate.graph_was_loaded(), "second reopen must have loaded");
        ms
    };

    println!(
        "6a — {n} vectors: rebuild {rebuild_ms:.0} ms → load {load_ms:.0} ms ({:.1}× faster)",
        rebuild_ms / load_ms
    );
    assert!(
        load_ms * 3.0 < rebuild_ms,
        "loading the graph must be at least 3× faster than rebuilding it \
         (load {load_ms:.0} ms vs rebuild {rebuild_ms:.0} ms)"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_persisted_graph_is_rejected_and_rebuild_sees_all_writes() {
    let dir = tempfile::tempdir().unwrap();

    {
        let estate = Estate::open(dir.path(), "gp").unwrap();
        let recall = estate.recall();

        // Batch A, then capture the graph at seq_A.
        recall.upsert(corpus(1000)).await.unwrap();
        recall.quiesce().await.unwrap();
        estate.persist_graph().unwrap();

        // Batch B lands *after* the capture: the on-disk blob is now stale.
        let batch_b: Vec<_> = (1000..1500)
            .map(|i| VectorRecord::new(format!("d{i}"), vec_for(i as u64), format!("late {i}")))
            .collect();
        recall.upsert(batch_b).await.unwrap();
        recall.quiesce().await.unwrap();

        // Simulate a crash: do NOT re-persist on drop, so the stale blob survives
        // exactly as an unclean shutdown would leave it.
        estate.set_persist_graph_on_drop(false);
    } // dropped with the stale blob intact

    let estate = Estate::open(dir.path(), "gp").unwrap();
    assert!(
        !estate.graph_was_loaded(),
        "a blob tagged with an older feed_seq must be rejected → rebuild"
    );
    let recall = estate.recall();

    // The rebuild is from the durable vectors, so every document is present —
    // including batch B, which was never in the persisted graph.
    assert_eq!(recall.len().await.unwrap(), 1500);
    let late = vec_for(1_234);
    let hits = ranked(&recall, &late, 5).await;
    assert!(!hits.is_empty(), "rebuilt graph must be queryable");
    let hits_b = ranked(&recall, &vec_for(1_400), 5).await;
    assert!(
        hits_b.iter().any(|(id, _)| id == "d1400"),
        "a batch-B document must be findable after rebuild"
    );
}
