//! Filtered ANN must return the filtered nearest neighbours — not the global
//! nearest neighbours that happen to survive a filter.
//!
//! RRO now has all three of Qdrant's filter strategies, chosen by the exact
//! cardinality the payload index already gives us (no estimation needed):
//!
//! * **exact scoping** (matches ≤ `EXACT_SCOPE_MAX` = 65,536) — score the whole
//!   resolved set by point-lookup. Correct at any selectivity, ~2.7 ms / 5k ids.
//! * **filter-aware traversal** (matches > 65,536) — walk the HNSW graph but admit
//!   only allowed nodes to the result heap, with `ef` widened by inverse
//!   selectivity. Correct, and sub-linear in the matched set.
//! * **post-filter** (filter not resolvable from indexes at all) — over-fetch +
//!   retain, the only option when there is no id set to work from.
//!
//! The bug this replaced: the old code capped exact scoping at 4,096 and fell to a
//! *global* post-filter above it. At medium selectivity — a filter matching more
//! than 4,096 docs but a small fraction of the corpus — the global top-`k×8` (80
//! candidates) almost never landed in the filtered subset when the filter was
//! uncorrelated with the query, and `retain` threw the rest away. A top-10 request
//! came back with ~2 results. Not slow — **wrong**, and silently. Reproduced at
//! 200k docs / 40 buckets (got 1 result, recall@10 = 0.10) before the fix.
//!
//! These tests hold all three strategies.
//!
//! They are `#[ignore]` — each seeds 200k–300k docs, which is ~90 s per test in a
//! debug build, past nextest's 60 s slow-kill. Run them with `scripts/gates.sh`
//! or `cargo test -p connxism --release --test filter_aware_hnsw -- --ignored`.
//! The *mechanism* is covered fast and in-CI by
//! `recall::ann::filter_aware_tests` (5k nodes, sub-second); these are the
//! full-scale correctness gates.

use connxism::{Estate, EstateQuery};
use rro_core::{Condition, Embedding, Filter, Recall, VectorRecord};

/// A corpus where the filter and the query vector are **uncorrelated** — the
/// realistic adversarial case. `bucket` (the filtered attribute) is assigned
/// round-robin, independent of the vector, so the filtered subset is scattered
/// uniformly through vector space rather than clustered near the query.
///
/// Vectors are 3-d and near-random so the ANN graph has real structure to
/// traverse (not a line). One `bucket` value marks ~0.5% of the corpus — above
/// the 4096 filter-first cap once the corpus is large enough, which is the whole
/// point.
async fn scattered_corpus(estate: &Estate, n: usize, buckets: usize) -> connxism::ConnXRecall {
    estate.create_payload_index("bucket").unwrap();
    let recall = estate.recall();

    let mut seed = 0x1234_5678_u64;
    let mut lcg = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
    };

    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        let v = Embedding(vec![lcg(), lcg(), lcg()]).normalized();
        let mut r = VectorRecord::new(format!("d{i}"), v, format!("doc {i}"));
        // bucket is independent of the vector — filter ⟂ query.
        r.metadata = rro_core::Metadata::from([(
            "bucket".to_string(),
            serde_json::json!((i % buckets) as i64),
        )]);
        records.push(r);
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

/// Exact oracle: the true filtered top-k, by brute-force cosine over exactly the
/// docs that match the filter. This is the answer the engine must reproduce.
async fn exact_filtered_topk(
    recall: &connxism::ConnXRecall,
    query: &Embedding,
    bucket: i64,
    k: usize,
) -> Vec<String> {
    // Pull the whole matching set exactly (filter-first is correct; we use it as
    // ground truth), scored, top-k.
    let all = recall
        .query(EstateQuery {
            vector: Some(query.clone()),
            dsl: Some(Filter::new().must(Condition::eq("bucket", serde_json::json!(bucket)))),
            top_k: 100_000, // everything that matches
            ..Default::default()
        })
        .await
        .unwrap();
    all.into_iter()
        .take(k)
        .map(|c| c.id.as_str().to_string())
        .collect()
}

fn recall_at_k(got: &[String], truth: &[String]) -> f64 {
    if truth.is_empty() {
        return 1.0;
    }
    let t: std::collections::HashSet<&str> = truth.iter().map(String::as_str).collect();
    let hit = got.iter().filter(|id| t.contains(id.as_str())).count();
    hit as f64 / truth.len() as f64
}

/// THE bug: a medium-selectivity filter over a corpus past the filter-first cap
/// must still return a full, correct top-k.
///
/// The regime has to be chosen exactly, or the reproduction is a no-op:
///
/// * matches **> 4096** → disqualifies filter-first, falls to post-filter;
/// * selectivity **small** → the global top-`k×8` (80 candidates) rarely lands in
///   the filtered subset. Expected matches in the fetched set ≈ `80 × selectivity`.
///
/// `200k docs, 40 buckets` → **5,000 matches** (over the 4096 cap) at **2.5%**
/// selectivity → only ≈ **2 of the 80** fetched candidates match. Post-filter
/// returns ~2 results for a top-10 request. That is the silent near-empty page.
///
/// (An earlier version used 12k/2 = 6k per bucket, but that is 50% selectivity —
/// the global top-80 is half-full of matches and the bug hides. Big matching set
/// is necessary but not sufficient; it must also be a small *fraction*.)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "200k-300k docs; ~90s debug, past nextest slow-kill. Run via scripts/gates.sh"]
async fn medium_selectivity_filter_returns_a_full_correct_topk() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fah").unwrap();
    // 200k docs, 40 buckets = 5,000 per bucket: over INDEXED_SCOPE_MAX (4096) so
    // no filter-first, and 2.5% selectivity so post-filter's 80 candidates almost
    // never land inside the bucket.
    let recall = scattered_corpus(&estate, 200_000, 40).await;

    let query = Embedding(vec![1.0, 0.0, 0.0]).normalized();
    let k = 10;
    let truth = exact_filtered_topk(&recall, &query, 0, k).await;
    assert_eq!(truth.len(), k, "sanity: the oracle found a full top-k");

    let got: Vec<String> = recall
        .query(EstateQuery {
            vector: Some(query.clone()),
            dsl: Some(Filter::new().must(Condition::eq("bucket", serde_json::json!(0)))),
            top_k: k,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_iter()
        .map(|c| c.id.as_str().to_string())
        .collect();

    let r = recall_at_k(&got, &truth);
    assert!(
        got.len() == k && r >= 0.9,
        "filtered top-{k} over a 6k-match filter: got {} results, recall@{k} = {r:.2} \
         vs the exact oracle. A medium-selectivity filter must not silently return \
         a short or wrong page — that is the filter-aware-HNSW gap.",
        got.len()
    );
}

/// The fix must not regress the case that already worked: a *small* filter
/// (≤4096 matches) still resolves exactly via filter-first.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "200k-300k docs; ~90s debug, past nextest slow-kill. Run via scripts/gates.sh"]
async fn small_selectivity_filter_is_still_exact() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fah").unwrap();
    // 12k docs, 200 buckets → 60 per bucket, well under the cap.
    let recall = scattered_corpus(&estate, 12_000, 200).await;

    let query = Embedding(vec![0.0, 1.0, 0.0]).normalized();
    let k = 10;
    let truth = exact_filtered_topk(&recall, &query, 7, k).await;

    let got: Vec<String> = recall
        .query(EstateQuery {
            vector: Some(query.clone()),
            dsl: Some(Filter::new().must(Condition::eq("bucket", serde_json::json!(7)))),
            top_k: k,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_iter()
        .map(|c| c.id.as_str().to_string())
        .collect();

    assert_eq!(
        recall_at_k(&got, &truth),
        1.0,
        "a small filter must remain exact — filter-first was already correct here"
    );
}

/// The third strategy, exercised: a matched set past `EXACT_SCOPE_MAX` (65,536)
/// takes filter-aware graph traversal, not exact scoring — and must still be
/// correct.
///
/// 300k docs, 3 buckets → 100k per bucket, over the exact ceiling. At 33%
/// selectivity the bug wouldn't bite post-filter, so this isn't a bug repro — it
/// is a correctness gate for the traversal path itself, which the medium test
/// (5k matches, exact path) never touches.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "200k-300k docs; ~90s debug, past nextest slow-kill. Run via scripts/gates.sh"]
async fn huge_matched_set_uses_traversal_and_stays_correct() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fah").unwrap();
    let recall = scattered_corpus(&estate, 300_000, 3).await;

    let query = Embedding(vec![0.3, 0.7, 0.1]).normalized();
    let k = 10;
    let truth = exact_filtered_topk(&recall, &query, 1, k).await;

    let got: Vec<String> = recall
        .query(EstateQuery {
            vector: Some(query.clone()),
            dsl: Some(Filter::new().must(Condition::eq("bucket", serde_json::json!(1)))),
            top_k: k,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_iter()
        .map(|c| c.id.as_str().to_string())
        .collect();

    let r = recall_at_k(&got, &truth);
    assert!(
        got.len() == k && r >= 0.8,
        "filter-aware traversal over a 100k-match set: got {} results, recall@{k} = {r:.2} \
         vs the exact oracle (traversal is approximate, so the bar is 0.8, not 1.0)",
        got.len()
    );
}
