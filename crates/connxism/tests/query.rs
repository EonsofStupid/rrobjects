//! Sprint 8 gates: typed queries with filters, facets, count, scroll.

use connxism::{Estate, EstateQuery};
use embedder::DeterministicEmbedder;
use rrf_core::{Embedder, Metadata, Recall, VectorRecord};

async fn seed(estate: &Estate) -> connxism::ConnXRecall {
    let recall = estate.recall();
    let embed = DeterministicEmbedder::new();
    let mut records = Vec::new();
    for i in 0..60 {
        let team = if i % 3 == 0 { "ops" } else { "eng" };
        let text = format!("deployment checklist item {i} for rollout");
        let mut meta = Metadata::new();
        meta.insert("team".into(), serde_json::json!(team));
        meta.insert("priority".into(), serde_json::json!(i % 5));
        let mut r = VectorRecord::new(
            format!("doc{i}"),
            embed.embed_one(&text).await.unwrap(),
            text,
        );
        r.metadata = meta;
        records.push(r);
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn filtered_query_returns_only_matching_docs() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "q").unwrap();
    let recall = seed(&estate).await;
    let embed = DeterministicEmbedder::new();

    let qv = embed
        .embed_one("deployment checklist rollout")
        .await
        .unwrap();
    let hits = recall
        .query(
            EstateQuery::hybrid("deployment checklist rollout", qv, 10)
                .must("team", serde_json::json!("ops")),
        )
        .await
        .unwrap();
    assert!(!hits.is_empty());
    assert!(
        hits.iter()
            .all(|c| c.metadata.get("team") == Some(&serde_json::json!("ops"))),
        "every hit must satisfy the filter"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn facet_count_scroll() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "f").unwrap();
    let _recall = seed(&estate).await;

    // Facet: 60 docs, team=ops on every i%3==0 → 20 ops / 40 eng.
    let facet = estate.facet("team").unwrap();
    assert_eq!(facet.get("ops"), Some(&20));
    assert_eq!(facet.get("eng"), Some(&40));

    // Count: filtered and total.
    let mut filter = Metadata::new();
    filter.insert("team".into(), serde_json::json!("eng"));
    assert_eq!(estate.count(&filter).unwrap(), 40);
    assert_eq!(estate.count(&Metadata::new()).unwrap(), 60);

    // Scroll: pages cover everything exactly once.
    let mut seen = std::collections::HashSet::new();
    let mut cursor: Option<String> = None;
    loop {
        let page = estate.scroll(cursor.as_deref(), 17).unwrap();
        if page.is_empty() {
            break;
        }
        for d in &page {
            assert!(seen.insert(d.id.clone()), "no overlap between pages");
        }
        cursor = page.last().map(|d| d.id.clone());
    }
    assert_eq!(seen.len(), 60, "scroll covers the whole estate");
}
