//! Sprint 14 gates: the analyzer is part of the lexical index's identity —
//! stemming collapses inflections at index AND query time, stopwords never
//! reach the postings, prefix analysis serves autocomplete, and the
//! persisted analyzer survives reopen (creation config wins once, forever).

use connxism::{Estate, EstateConfig};
use rrf_core::text::Analyzer;
use rrf_core::{Embedding, Recall, VectorRecord};

fn rec(id: &str, text: &str) -> VectorRecord {
    VectorRecord::new(id, Embedding(vec![0.1, 0.2, 0.3, 0.4]), text)
}

fn stemming_config() -> EstateConfig {
    EstateConfig {
        analyzer: Analyzer::stemming(),
        ..EstateConfig::default()
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn stemming_matches_inflections_legacy_does_not() {
    let dir = tempfile::tempdir().unwrap();

    // Legacy estate: "run" does not match "running" (exact-token BM25).
    let legacy = Estate::open(dir.path().join("legacy"), "l").unwrap();
    let lr = legacy.recall();
    lr.upsert(vec![rec("d1", "the runner was running quickly")])
        .await
        .unwrap();
    assert!(
        lr.lexical_search("run", 5).await.unwrap().is_empty(),
        "legacy analyzer must not stem"
    );
    // …but the exact inflection matches.
    assert_eq!(lr.lexical_search("running", 5).await.unwrap().len(), 1);

    // Stemming estate: "run", "runs", "running" all hit the same postings.
    let stemmed = Estate::open_with(dir.path().join("stem"), "s", stemming_config()).unwrap();
    let sr = stemmed.recall();
    sr.upsert(vec![
        rec("d1", "the runner was running quickly"),
        rec("d2", "estates connect to the connectome"),
    ])
    .await
    .unwrap();
    for q in ["run", "runs", "running"] {
        let hits = sr.lexical_search(q, 5).await.unwrap();
        assert_eq!(hits.len(), 1, "query {q:?} matches the running doc");
        assert_eq!(hits[0].id.as_str(), "d1");
    }
    // Inflections agree across index and query on the other doc too.
    let hits = sr.lexical_search("connected estate", 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "d2");

    // Stopwords never reach the postings: a pure-stopword query is empty.
    assert!(sr.lexical_search("the was to", 5).await.unwrap().is_empty());

    // Overwrite retracts through the SAME analyzer: rewrite d1 without the
    // running vocabulary and the stemmed postings must be gone.
    sr.upsert(vec![rec("d1", "entirely different words now")])
        .await
        .unwrap();
    assert!(sr.lexical_search("running", 5).await.unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn autocomplete_prefix_analyzer() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open_with(
        dir.path(),
        "auto",
        EstateConfig {
            analyzer: Analyzer::autocomplete(2, 6),
            ..EstateConfig::default()
        },
    )
    .unwrap();
    let recall = estate.recall();
    recall
        .upsert(vec![
            rec("c", "connectome estates"),
            rec("r", "reason ready flow"),
        ])
        .await
        .unwrap();

    // Partial prefixes hit the right doc — autocomplete over BM25.
    let hits = recall.lexical_search("con", 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "c");
    let hits = recall.lexical_search("rea", 5).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "r");
}

#[tokio::test(flavor = "multi_thread")]
async fn analyzer_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let estate = Estate::open_with(dir.path(), "p", stemming_config()).unwrap();
        estate
            .recall()
            .upsert(vec![rec("d", "connections running deep")])
            .await
            .unwrap();
        assert_eq!(estate.info().analyzer, Analyzer::stemming());
    } // dropped: RocksDB lock released

    // Reopen with the DEFAULT config: the persisted analyzer must win —
    // the postings were built with it.
    let reopened = Estate::open(dir.path(), "p").unwrap();
    assert_eq!(
        reopened.info().analyzer,
        Analyzer::stemming(),
        "creation-time analyzer survives reopen"
    );
    let hits = reopened
        .recall()
        .lexical_search("connect", 5)
        .await
        .unwrap();
    assert_eq!(hits.len(), 1, "stemmed match still works after reopen");
}
