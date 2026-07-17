//! Phase 6 — immutable segments / background optimizer (vector side): graph
//! compaction reclaims tombstones.
//!
//! Deletes and overwrites leave dead nodes in the ANN graph (soft tombstones —
//! still traversed, still holding RAM/disk). Nothing removed them until now; over
//! a churny estate's life they dominate. `compact_graph` rebuilds the graph
//! tombstone-free from the durable vectors (which hold only live entries) and
//! swaps it in, without changing what the estate returns.

use connxism::{Estate, Recall};
use rro_core::{Embedding, VectorRecord};

fn vec_for(seed: u64, dim: usize) -> Embedding {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let v: Vec<f32> = (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect();
    Embedding(v).normalized()
}

#[tokio::test(flavor = "multi_thread")]
async fn compaction_reclaims_tombstones_and_preserves_results() {
    let dim = 32;
    let n = 2000; // above the ANN threshold, so the graph answers
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "compact").unwrap();
    let recall = estate.recall();

    // Seed.
    let seed: Vec<_> = (0..n)
        .map(|i| VectorRecord::new(format!("d{i}"), vec_for(i as u64, dim), format!("doc {i}")))
        .collect();
    recall.upsert(seed).await.unwrap();
    // Drain before churning, so overwrites tombstone *applied* nodes rather than
    // coalescing with the still-pending seed insert for the same id.
    recall.quiesce().await.unwrap();

    // Churn: overwrite 500 (each leaves a tombstone) and remove 300.
    let overwrites: Vec<_> = (0..500)
        .map(|i| {
            VectorRecord::new(
                format!("d{i}"),
                vec_for(1_000_000 + i, dim),
                format!("doc {i}"),
            )
        })
        .collect();
    recall.upsert(overwrites).await.unwrap();
    for i in 500..800 {
        recall.remove(&format!("d{i}").into()).await.unwrap();
    }
    recall.quiesce().await.unwrap();

    // Tombstones accumulated: 500 overwrites + 300 removes = 800 dead nodes.
    let before = estate.graph_nodes();
    assert_eq!(
        before.tombstones, 800,
        "expected 800 tombstones, got {before:?}"
    );
    assert_eq!(before.live(), n - 300); // 300 removed, overwrites keep the id live
    let live_count = recall.len().await.unwrap();

    // Capture results for a set of queries, to prove compaction is invisible.
    let queries: Vec<Embedding> = (0..30).map(|q| vec_for(9_000_000 + q, dim)).collect();
    async fn ids(recall: &connxism::ConnXRecall, q: &Embedding) -> Vec<String> {
        recall
            .search(q, 10)
            .await
            .unwrap()
            .into_iter()
            .map(|c| c.id.as_str().to_string())
            .collect()
    }
    let mut before_results = Vec::new();
    for q in &queries {
        before_results.push(ids(&recall, q).await);
    }

    // Compact.
    let report = estate.compact_graph().unwrap();
    assert_eq!(report.reclaimed(), 800, "must reclaim the 800 tombstones");
    let after = estate.graph_nodes();
    assert_eq!(after.tombstones, 0, "compacted graph has no tombstones");
    assert_eq!(after.total, before.live(), "only the live nodes remain");

    // The live set and search results are unchanged by compaction.
    assert_eq!(recall.len().await.unwrap(), live_count);
    for (q, want) in queries.iter().zip(&before_results) {
        assert_eq!(
            &ids(&recall, q).await,
            want,
            "compaction must not change results"
        );
    }

    // A removed doc stays gone; an overwritten doc still resolves.
    let hits = recall.search(&vec_for(600, dim), 10).await.unwrap();
    assert!(
        hits.iter().all(|c| c.id.as_str() != "d600"),
        "a removed doc must not reappear after compaction"
    );
    assert_eq!(recall.len().await.unwrap(), n - 300);
}
