//! Sprint 18 gates: aliases redirect queries atomically without touching
//! data; each payload op leaves the payload indexes exactly consistent
//! (asserted via index-resolved id-sets) and lands one changefeed row.

use connxism::{Estate, EstateQuery};
use rrf_core::{Condition, Embedding, Filter, Recall, VectorRecord};

fn rec(id: &str, coll: Option<&str>, team: &str) -> VectorRecord {
    let mut r = VectorRecord::new(
        id,
        Embedding(vec![0.4, 0.3, 0.2, 0.1]),
        format!("payload corpus {id}"),
    );
    r.metadata.insert("team".into(), serde_json::json!(team));
    if let Some(c) = coll {
        r = r.in_collection(c);
    }
    r
}

fn team_filter(team: &str) -> Filter {
    Filter::default().must(Condition::eq("team", serde_json::json!(team)))
}

#[tokio::test(flavor = "multi_thread")]
async fn alias_switch_redirects_queries_atomically() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "al").unwrap();
    let recall = estate.recall();
    recall
        .upsert(vec![
            rec("a1", Some("alpha"), "x"),
            rec("a2", Some("alpha"), "x"),
            rec("b1", Some("beta"), "x"),
        ])
        .await
        .unwrap();

    // "prod" aliases alpha: queries through the alias see alpha's docs.
    estate.create_alias("prod", "alpha").unwrap();
    let q = || {
        EstateQuery::hybrid("payload corpus", Embedding(vec![0.4, 0.3, 0.2, 0.1]), 10)
            .in_collection("prod")
    };
    let hits = recall.query(q()).await.unwrap();
    let mut ids: Vec<&str> = hits.iter().map(|c| c.id.as_str()).collect();
    ids.sort();
    assert_eq!(ids, vec!["a1", "a2"]);

    // Atomic switch: the SAME query now sees beta — no data touched.
    estate.create_alias("prod", "beta").unwrap();
    let hits = recall.query(q()).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "b1");
    assert_eq!(estate.collections().unwrap().len(), 2, "data untouched");

    // List + delete: literal behavior returns (unknown name → empty).
    assert_eq!(estate.aliases().unwrap().get("prod").unwrap(), "beta");
    estate.delete_alias("prod").unwrap();
    assert!(recall.query(q()).await.unwrap().is_empty());

    // A real collection name still resolves as itself when no alias hides it.
    let hits = recall
        .query(
            EstateQuery::hybrid("payload corpus", Embedding(vec![0.4, 0.3, 0.2, 0.1]), 10)
                .in_collection("alpha"),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn payload_ops_keep_indexes_exactly_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "po").unwrap();
    estate.create_payload_index("team").unwrap();
    estate.create_payload_index("priority").unwrap();
    let recall = estate.recall();
    recall
        .upsert(vec![rec("d1", None, "red"), rec("d2", None, "blue")])
        .await
        .unwrap();

    let ids_for = |team: &str| estate.ids_where(&team_filter(team)).unwrap().unwrap();
    assert_eq!(ids_for("red"), vec!["d1".to_string()]);
    let feed0 = estate.changes(0, 10_000).unwrap().len();

    // set_payload merges: d1 moves team red→green, gains priority.
    let mut patch = rrf_core::Metadata::new();
    patch.insert("team".into(), serde_json::json!("green"));
    patch.insert("priority".into(), serde_json::json!(7));
    recall.set_payload("d1", patch).await.unwrap();
    assert!(ids_for("red").is_empty(), "old index row retracted");
    assert_eq!(ids_for("green"), vec!["d1".to_string()]);
    let pri = Filter::default().must(Condition::range("priority", Some(5.0), None));
    assert_eq!(
        estate.ids_where(&pri).unwrap().unwrap(),
        vec!["d1".to_string()]
    );
    let doc = recall.doc("d1").await.unwrap().unwrap();
    assert_eq!(doc.metadata["team"], serde_json::json!("green"));
    assert_eq!(estate.changes(0, 10_000).unwrap().len(), feed0 + 1);

    // delete_payload_keys: priority gone from doc AND index.
    recall
        .delete_payload_keys("d1", vec!["priority".into()])
        .await
        .unwrap();
    assert!(estate.ids_where(&pri).unwrap().unwrap().is_empty());
    assert!(!recall
        .doc("d1")
        .await
        .unwrap()
        .unwrap()
        .metadata
        .contains_key("priority"));
    assert_eq!(estate.changes(0, 10_000).unwrap().len(), feed0 + 2);

    // overwrite_payload replaces wholesale.
    let mut fresh = rrf_core::Metadata::new();
    fresh.insert("team".into(), serde_json::json!("gold"));
    recall.overwrite_payload("d1", fresh).await.unwrap();
    assert!(ids_for("green").is_empty());
    assert_eq!(ids_for("gold"), vec!["d1".to_string()]);
    assert_eq!(estate.changes(0, 10_000).unwrap().len(), feed0 + 3);

    // clear_payload: every index row for d1 retracts; d2 untouched.
    recall.clear_payload("d1").await.unwrap();
    assert!(ids_for("gold").is_empty());
    assert!(recall.doc("d1").await.unwrap().unwrap().metadata.is_empty());
    assert_eq!(ids_for("blue"), vec!["d2".to_string()], "sibling untouched");
    assert_eq!(estate.changes(0, 10_000).unwrap().len(), feed0 + 4);

    // Mutating a missing doc errors.
    assert!(recall.clear_payload("ghost").await.is_err());

    // Filtered count agrees post-ops (the census path still works).
    assert_eq!(estate.count_where(&team_filter("blue")).unwrap(), 1);
}
