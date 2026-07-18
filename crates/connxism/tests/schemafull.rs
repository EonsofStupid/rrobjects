//! Phase 10: schemafull `DEFINE FIELD ... TYPE ...` is enforced on write. A
//! record in a collection whose declared field is present with the wrong type is
//! rejected (the whole upsert rolls back); a conforming record is accepted; and
//! `REMOVE FIELD` lifts the constraint.

use connxism::Estate;
use rro_core::{Embedding, Metadata, Recall, VectorRecord};

fn rec(id: &str, collection: &str, field: &str, value: serde_json::Value) -> VectorRecord {
    let mut r = VectorRecord::new(id, Embedding(vec![0.1, 0.2, 0.3, 0.4]), format!("doc {id}"));
    r.collection = Some(collection.to_string());
    r.metadata = Metadata::from([(field.to_string(), value)]);
    r
}

#[tokio::test(flavor = "multi_thread")]
async fn schemafull_field_type_is_enforced() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "schema").unwrap();
    estate.define_field("products", "price", "float").unwrap();
    estate.define_field("products", "sku", "string").unwrap();
    let recall = estate.recall();

    // Wrong type: price as a string → the upsert is rejected.
    let err = recall
        .upsert(vec![rec(
            "p1",
            "products",
            "price",
            serde_json::json!("cheap"),
        )])
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("schemafull violation"),
        "wrong-typed field must be rejected: {err}"
    );
    // And nothing landed (the upsert rolled back).
    assert_eq!(recall.len().await.unwrap(), 0);

    // Right type: price as a number → accepted.
    recall
        .upsert(vec![rec(
            "p2",
            "products",
            "price",
            serde_json::json!(9.99),
        )])
        .await
        .unwrap();
    assert_eq!(recall.len().await.unwrap(), 1);

    // A field not present is allowed (schemas can be added to a live collection).
    recall
        .upsert(vec![rec("p3", "products", "sku", serde_json::json!("ABC"))])
        .await
        .unwrap();
    assert_eq!(recall.len().await.unwrap(), 2);

    // A record in a DIFFERENT collection is not constrained by `products`.
    recall
        .upsert(vec![rec(
            "o1",
            "orders",
            "price",
            serde_json::json!("free"),
        )])
        .await
        .unwrap();
    assert_eq!(recall.len().await.unwrap(), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn int_type_rejects_floats_and_remove_field_lifts_the_constraint() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "schema2").unwrap();
    estate.define_field("inv", "qty", "int").unwrap();
    let recall = estate.recall();

    // 3.5 is a number but not an integer → rejected under TYPE int.
    let err = recall
        .upsert(vec![rec("i1", "inv", "qty", serde_json::json!(3.5))])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("schemafull violation"), "{err}");

    // 3 is an integer → accepted.
    recall
        .upsert(vec![rec("i2", "inv", "qty", serde_json::json!(3))])
        .await
        .unwrap();
    assert_eq!(recall.len().await.unwrap(), 1);

    // REMOVE FIELD lifts the constraint — the previously-rejected value now lands.
    estate.remove_field("inv", "qty").unwrap();
    recall
        .upsert(vec![rec("i1", "inv", "qty", serde_json::json!(3.5))])
        .await
        .unwrap();
    assert_eq!(recall.len().await.unwrap(), 2);
    assert!(estate.schema().unwrap().is_empty());
}
