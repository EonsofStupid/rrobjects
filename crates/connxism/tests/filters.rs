//! Sprint 9 gates: the filter DSL, payload secondary indexes (filter-first
//! vs post-filter), score thresholds, lean payloads, and index maintenance.

use connxism::{Condition, Estate, EstateQuery, Filter};
use embedder::DeterministicEmbedder;
use rro_core::{Embedder, Metadata, Recall, VectorRecord};

fn meta(team: &str, priority: i64, status: &str) -> Metadata {
    let mut m = Metadata::new();
    m.insert("team".into(), serde_json::json!(team));
    m.insert("priority".into(), serde_json::json!(priority));
    m.insert("status".into(), serde_json::json!(status));
    m
}

/// 300 docs; `team`/`priority` payload-indexed, `status` deliberately not.
async fn seed(estate: &Estate, n: usize) -> connxism::ConnXRecall {
    estate.create_payload_index("team").unwrap();
    estate.create_payload_index("priority").unwrap();

    let recall = estate.recall();
    let embed = DeterministicEmbedder::with_dim(32);
    let mut records = Vec::new();
    for i in 0..n {
        let team = ["ops", "eng", "sec"][i % 3];
        let status = if i % 2 == 0 { "open" } else { "done" };
        let text = format!("rollout checklist entry {i} for the estate");
        let mut r = VectorRecord::new(
            format!("doc{i}"),
            embed.embed_one(&text).await.unwrap(),
            text,
        );
        r.metadata = meta(team, (i % 10) as i64, status);
        records.push(r);
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn dsl_matches_brute_force_on_both_strategies() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "dsl").unwrap();
    let recall = seed(&estate, 300).await;
    let embed = DeterministicEmbedder::with_dim(32);
    let qv = embed.embed_one("rollout checklist estate").await.unwrap();

    // Fully indexed filter → filter-first: must + range + should + must_not.
    let indexed = Filter::new()
        .must(Condition::eq("team", serde_json::json!("ops")))
        .must(Condition::range("priority", Some(2.0), Some(7.0)))
        .must_not(Condition::eq("priority", serde_json::json!(5)));
    // Same filter, plus a clause on the UNindexed field → post-filter path.
    let mixed = Filter::new()
        .must(Condition::eq("team", serde_json::json!("ops")))
        .must(Condition::range("priority", Some(2.0), Some(7.0)))
        .must_not(Condition::eq("priority", serde_json::json!(5)))
        .must(Condition::eq("status", serde_json::json!("open")));

    // The indexed strategy really is chosen (and refuses the mixed filter).
    assert!(estate.ids_where(&indexed).unwrap().is_some());
    assert!(estate.ids_where(&mixed).unwrap().is_none());

    for (filter, label) in [(indexed, "filter-first"), (mixed, "post-filter")] {
        let hits = recall
            .query(
                EstateQuery::hybrid("rollout checklist estate", qv.clone(), 25)
                    .filtered(filter.clone()),
            )
            .await
            .unwrap();
        assert!(!hits.is_empty(), "{label}: no hits");
        for c in &hits {
            assert!(
                filter.matches(&c.metadata),
                "{label}: {} violates the filter: {:?}",
                c.id,
                c.metadata
            );
        }
    }

    // Exact set arithmetic: count_where equals the brute-force count.
    let f = Filter::new()
        .must(Condition::eq("team", serde_json::json!("eng")))
        .must(Condition::range("priority", Some(3.0), None));
    // eng = i%3==1; priority = i%10 in [3,9]. Brute-force over 300:
    let expected = (0..300).filter(|i| i % 3 == 1 && (i % 10) >= 3).count() as u64;
    assert_eq!(estate.count_where(&f).unwrap(), expected);

    // `should` union: ops OR sec.
    let f = Filter::new()
        .should(Condition::eq("team", serde_json::json!("ops")))
        .should(Condition::eq("team", serde_json::json!("sec")));
    assert_eq!(estate.count_where(&f).unwrap(), 200);

    // Match-any equals the equivalent should-union.
    let f = Filter::new().must(Condition::any(
        "team",
        vec![serde_json::json!("ops"), serde_json::json!("sec")],
    ));
    assert_eq!(estate.count_where(&f).unwrap(), 200);
}

#[tokio::test(flavor = "multi_thread")]
async fn index_maintenance_tracks_overwrites_and_removals() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "maint").unwrap();
    let recall = seed(&estate, 30).await;
    let embed = DeterministicEmbedder::with_dim(32);

    let ops = Filter::new().must(Condition::eq("team", serde_json::json!("ops")));
    let before = estate.count_where(&ops).unwrap();

    // Overwrite doc0 (ops → eng): the old index row must retract.
    let mut r = VectorRecord::new(
        "doc0".to_string(),
        embed.embed_one("rewritten entry").await.unwrap(),
        "rewritten entry".to_string(),
    );
    r.metadata = meta("eng", 1, "open");
    recall.upsert(vec![r]).await.unwrap();
    assert_eq!(estate.count_where(&ops).unwrap(), before - 1);
    let ids = estate.ids_where(&ops).unwrap().unwrap();
    assert!(!ids.contains(&"doc0".to_string()));

    // Remove doc3 (ops): its rows must retract too.
    recall.remove(&"doc3".into()).await.unwrap();
    assert_eq!(estate.count_where(&ops).unwrap(), before - 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn threshold_and_lean_payload() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "thr").unwrap();
    let recall = seed(&estate, 60).await;
    let embed = DeterministicEmbedder::with_dim(32);
    let qv = embed.embed_one("rollout checklist estate").await.unwrap();

    let all = recall
        .query(EstateQuery::hybrid(
            "rollout checklist estate",
            qv.clone(),
            10,
        ))
        .await
        .unwrap();
    assert_eq!(all.len(), 10);
    let cutoff = all[4].score;

    let thresholded = recall
        .query(EstateQuery::hybrid("rollout checklist estate", qv.clone(), 10).threshold(cutoff))
        .await
        .unwrap();
    assert!(thresholded.len() < all.len());
    assert!(thresholded.iter().all(|c| c.score >= cutoff));

    let lean = recall
        .query(EstateQuery::hybrid("rollout checklist estate", qv, 10).ids_only())
        .await
        .unwrap();
    assert_eq!(lean.len(), 10);
    assert!(lean
        .iter()
        .all(|c| c.text.is_empty() && c.metadata.is_empty()));
}

/// Measured (printed, recorded in BENCHMARKS.md): indexed count vs full scan
/// at 10k docs. Correctness is asserted; the timing is reported honestly.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "10k docs; ~25s debug, past nextest slow-kill on 2-core CI. Run via scripts/gates.sh"]
async fn p9_gate_indexed_filter_vs_scan_at_10k() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "p9").unwrap();
    let _recall = seed(&estate, 10_000).await;

    let f = Filter::new()
        .must(Condition::eq("team", serde_json::json!("sec")))
        .must(Condition::range("priority", Some(8.0), None));
    let expected = (0..10_000usize)
        .filter(|i| i % 3 == 2 && (i % 10) >= 8)
        .count() as u64;

    let t = std::time::Instant::now();
    let indexed = estate.count_where(&f).unwrap();
    let indexed_us = t.elapsed().as_micros();

    // Force the scan strategy by adding an unindexed clause that's always true.
    let scan_filter = f.clone().must(Condition::exists("status"));
    assert!(estate.ids_where(&scan_filter).unwrap().is_none());
    let t = std::time::Instant::now();
    let scanned = estate.count_where(&scan_filter).unwrap();
    let scan_us = t.elapsed().as_micros();

    assert_eq!(indexed, expected);
    assert_eq!(scanned, expected);
    println!(
        "P9 GATE — 10k docs: indexed count {indexed_us}µs vs full scan {scan_us}µs ({:.1}x)",
        scan_us as f64 / indexed_us.max(1) as f64
    );
    assert!(
        indexed_us < scan_us,
        "indexed path must beat the scan: {indexed_us}µs vs {scan_us}µs"
    );
}
