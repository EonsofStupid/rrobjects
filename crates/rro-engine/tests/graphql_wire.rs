//! GraphQL over the a2a transport — no HTTP server.
//!
//! GraphQL is a query language, not a transport, so the `graphql` verb rides the
//! same NDJSON/TCP connection as every other verb (tokio-signal spine, token
//! auth, all of it). This proves the client-chooses-the-shape property that
//! distinguishes GraphQL from REST, end to end over TCP.

use std::sync::Arc;

use rro_client::Client;
use rro_core::Document;
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;

async fn node(estate: Arc<connxism::Estate>) -> Client {
    let flow = Arc::new(
        ReasonReadyObject::builder()
            .recall(Arc::new(estate.recall()))
            .build(),
    );
    let node = FlowNode::new(flow, "gql-node").with_estate(estate);
    let (addr, task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    std::mem::forget(task);
    Client::new(addr.to_string())
}

async fn gql(client: &Client, query: &str) -> serde_json::Value {
    client.graphql(query).await.unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn graphql_search_over_the_wire_returns_only_requested_fields() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "gql").unwrap());
    let client = node(estate.clone()).await;

    client
        .index(vec![
            Document::new("alpha vector recall engine").with_id("a"),
            Document::new("beta storage layer").with_id("b"),
        ])
        .await
        .unwrap();

    // The client asks for id + score ONLY — not text. GraphQL must return that
    // shape and no more.
    let resp = gql(
        &client,
        r#"{ search(query: "vector recall", topK: 2) { id score } }"#,
    )
    .await;

    let hits = resp["data"]["search"].as_array().expect("search array");
    assert!(!hits.is_empty(), "search returned hits");
    for h in hits {
        assert!(h.get("id").is_some(), "id was requested");
        assert!(h.get("score").is_some(), "score was requested");
        assert!(
            h.get("text").is_none(),
            "text was NOT requested — GraphQL must not include it"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn graphql_document_and_nested_projection() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "gql").unwrap());
    let client = node(estate.clone()).await;

    let mut d = Document::new("hello graphql").with_id("doc1");
    d.metadata.insert("kind".into(), serde_json::json!("note"));
    d.metadata.insert("team".into(), serde_json::json!("eng"));
    client.index(vec![d]).await.unwrap();

    let resp = gql(
        &client,
        r#"{ document(id: "doc1") { id metadata { kind } } }"#,
    )
    .await;

    let doc = &resp["data"]["document"];
    assert_eq!(doc["id"], "doc1");
    assert_eq!(doc["metadata"]["kind"], "note");
    assert!(
        doc["metadata"].get("team").is_none(),
        "nested projection drops unrequested subfields"
    );
    assert!(doc.get("text").is_none(), "text not requested");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_graphql_parse_error_returns_the_errors_envelope_not_a_drop() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "gql").unwrap());
    let client = node(estate).await;

    let resp = gql(&client, "{ search(query: }").await;
    assert!(
        resp.get("errors").is_some(),
        "a malformed query must come back as a GraphQL errors envelope: {resp}"
    );
}
