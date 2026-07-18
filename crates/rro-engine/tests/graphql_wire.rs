//! GraphQL over the a2a transport — no HTTP server.
//!
//! GraphQL is a query language, not a transport, so the `graphql` verb rides the
//! same NDJSON/TCP connection as every other verb (tokio-signal spine, token
//! auth, all of it). This proves the client-chooses-the-shape property that
//! distinguishes GraphQL from REST, end to end over TCP.

use std::sync::Arc;

use rro_client::Client;
use rro_core::Document;
use rro_engine::{AuthPolicy, FlowNode, ReasonReadyObject, Role};
use rro_net::{tcp, Message};

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
async fn mutations_upsert_then_search_then_delete_over_the_wire() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "gql").unwrap());
    let client = node(estate).await;

    // upsert via a mutation — the write half of GraphQL, over the same verb.
    let resp = gql(
        &client,
        r#"mutation { upsert(id: "m1", text: "vector recall engine") { id indexed } }"#,
    )
    .await;
    assert_eq!(resp["data"]["upsert"]["id"], "m1", "{resp}");
    assert_eq!(resp["data"]["upsert"]["indexed"], 1);

    // The upserted document is immediately searchable in the same estate.
    let resp = gql(
        &client,
        r#"{ search(query: "vector recall", topK: 5) { id } }"#,
    )
    .await;
    let ids: Vec<String> = resp["data"]["search"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|h| h["id"].as_str().map(str::to_string))
        .collect();
    assert!(
        ids.contains(&"m1".to_string()),
        "upsert is searchable: {resp}"
    );

    // delete via a mutation removes it.
    let resp = gql(&client, r#"mutation { delete(id: "m1") { id deleted } }"#).await;
    assert_eq!(resp["data"]["delete"]["deleted"], true, "{resp}");

    let resp = gql(&client, r#"{ document(id: "m1") { id } }"#).await;
    assert!(
        resp["data"]["document"].is_null(),
        "deleted document is gone: {resp}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_readers_graphql_mutation_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "gql").unwrap());
    let policy = AuthPolicy::new(b"k".to_vec());
    let flow = Arc::new(
        ReasonReadyObject::builder()
            .recall(Arc::new(estate.recall()))
            .build(),
    );
    let node = FlowNode::new(flow, "gql-guard")
        .with_estate(estate)
        .with_auth(policy.clone());
    let (addr, task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    std::mem::forget(task);
    let reader = policy.issue_for("r", Role::Reader, None, 3600);

    // A reader's GraphQL mutation is refused — the write surface is gated
    // whichever language carries it.
    let m = Message::request(
        "c",
        "gql-guard",
        "graphql",
        serde_json::json!({ "query": r#"mutation { delete(id: "x") { id } }"# }),
    )
    .with_token(reader.clone());
    let reply = tcp::request(addr, &m).await.unwrap().body;
    assert_eq!(
        reply["error"], "unauthorized",
        "reader mutation refused: {reply}"
    );

    // But a reader's GraphQL query still resolves.
    let q = Message::request(
        "c",
        "gql-guard",
        "graphql",
        serde_json::json!({ "query": "{ health { docs } }" }),
    )
    .with_token(reader);
    let reply = tcp::request(addr, &q).await.unwrap().body;
    assert_ne!(
        reply["error"], "unauthorized",
        "reader query allowed: {reply}"
    );
    assert!(
        reply.get("data").is_some(),
        "reader gets a data envelope: {reply}"
    );
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
