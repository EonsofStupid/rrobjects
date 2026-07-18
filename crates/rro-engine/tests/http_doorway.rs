//! Phase 11 gate: the HTTP doorway is byte-identical to a2a.
//!
//! The doorway does not re-implement `query`/`ask`/`health` — it builds the same
//! a2a `Message` and calls the same `FlowNode::handle`. These tests prove that
//! equivalence at the wire: `POST /query` over HTTP returns the *same*
//! candidates as the a2a `query` verb and as a direct `estate.recall().query()`.

use std::sync::Arc;

use rro_client::Client;
use rro_core::{Embedding, EstateQuery, Recall, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn rec(id: &str, seed: f32) -> VectorRecord {
    VectorRecord::new(
        id,
        Embedding(vec![seed, 0.5, 0.25, 0.125]),
        format!("http corpus {id}"),
    )
    .in_collection("http")
}

/// Send one HTTP request, return (status line, body).
async fn http(
    addr: std::net::SocketAddr,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<&str>,
) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: x\r\n");
    if let Some(t) = token {
        req.push_str(&format!("Authorization: Bearer {t}\r\n"));
    }
    if let Some(b) = body {
        req.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        ));
        req.push_str("\r\n");
        req.push_str(b);
    } else {
        req.push_str("\r\n");
    }
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = String::new();
    stream.read_to_string(&mut buf).await.unwrap();
    let (head, body) = buf.split_once("\r\n\r\n").unwrap();
    let status = head.lines().next().unwrap_or("").to_string();
    (status, body.to_string())
}

async fn seeded_node() -> (Arc<FlowNode>, Arc<connxism::Estate>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "h").unwrap());
    let recall = estate.recall();
    recall
        .upsert(vec![rec("a", 0.1), rec("b", 0.2), rec("c", 0.3)])
        .await
        .unwrap();
    recall.quiesce().await.unwrap();
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = Arc::new(FlowNode::new(flow, "http-node").with_estate(estate.clone()));
    (node, estate, dir)
}

#[tokio::test(flavor = "multi_thread")]
async fn post_query_equals_a2a_query_and_direct_recall() {
    let (node, estate, _dir) = seeded_node().await;

    // The same typed query, three ways.
    let q = EstateQuery {
        vector: Some(Embedding(vec![0.1, 0.5, 0.25, 0.125])),
        top_k: 3,
        ..Default::default()
    };
    let q_json = serde_json::to_string(&q).unwrap();

    // 1) Direct in-process recall.
    let direct = estate.recall().query(q.clone()).await.unwrap();
    let direct_json = serde_json::to_value(&direct).unwrap();

    // 2) a2a `query` verb (typed client).
    let (a2a_addr, _a2a) = tcp::serve("127.0.0.1:0", node.clone()).await.unwrap();
    let a2a = Client::new(a2a_addr.to_string()).query(&q).await.unwrap();
    let a2a_json = serde_json::to_value(&a2a).unwrap();

    // 3) HTTP `POST /query`.
    let (http_addr, _http) = rro_engine::serve_http("127.0.0.1:0", node.clone())
        .await
        .unwrap();
    let (status, http_body) = http(http_addr, "POST", "/query", None, Some(&q_json)).await;
    assert!(status.contains("200"), "status: {status}");
    let http_json: serde_json::Value = serde_json::from_str(&http_body).unwrap();

    // All three carry the identical candidate list.
    assert_eq!(
        http_json["candidates"], direct_json,
        "HTTP candidates must equal direct recall"
    );
    assert_eq!(
        http_json["candidates"], a2a_json,
        "HTTP candidates must equal the a2a query verb byte-for-byte"
    );
    assert_eq!(
        http_json["candidates"].as_array().map(Vec::len),
        Some(3),
        "all three docs recalled: {http_body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn get_health_and_probes() {
    let (node, _estate, _dir) = seeded_node().await;
    let (addr, _http) = rro_engine::serve_http("127.0.0.1:0", node).await.unwrap();

    for probe in ["/healthz", "/livez", "/readyz"] {
        let (status, body) = http(addr, "GET", probe, None, None).await;
        assert!(status.contains("200"), "{probe}: {status}");
        assert_eq!(body, "ok\n");
    }

    let (status, body) = http(addr, "GET", "/health", None, None).await;
    assert!(status.contains("200"), "{status}");
    let h: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(h["node"], "http-node");
    assert_eq!(h["estate"]["docs"], 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn post_ask_returns_the_full_pass() {
    let (node, _estate, _dir) = seeded_node().await;
    let (addr, _http) = rro_engine::serve_http("127.0.0.1:0", node).await.unwrap();

    let (status, body) = http(
        addr,
        "POST",
        "/ask",
        None,
        Some(r#"{"query": "http corpus a"}"#),
    )
    .await;
    assert!(status.contains("200"), "{status}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        v.get("candidates").is_some() && v.get("context").is_some(),
        "ask returns candidates + LLM-ready context: {body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn doorway_carries_the_bearer_token_into_the_same_gate() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "g").unwrap());
    estate.recall().upsert(vec![rec("a", 0.1)]).await.unwrap();
    estate.recall().quiesce().await.unwrap();
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = Arc::new(
        FlowNode::new(flow, "guarded")
            .with_estate(estate)
            .with_token("s3cret"),
    );
    let (addr, _http) = rro_engine::serve_http("127.0.0.1:0", node).await.unwrap();

    let q = r#"{"vector":[0.1,0.5,0.25,0.125],"top_k":1}"#;

    // No token → 401 unauthorized, the same refusal a2a gives.
    let (status, body) = http(addr, "POST", "/query", None, Some(q)).await;
    assert!(status.contains("401"), "no token must be 401: {status}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"], "unauthorized");

    // Wrong token → 401.
    let (status, _) = http(addr, "POST", "/query", Some("nope"), Some(q)).await;
    assert!(status.contains("401"), "wrong token must be 401: {status}");

    // Bearer → 200 with candidates.
    let (status, body) = http(addr, "POST", "/query", Some("s3cret"), Some(q)).await;
    assert!(status.contains("200"), "bearer must be 200: {status}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["candidates"].as_array().map(Vec::len), Some(1));
}

#[tokio::test(flavor = "multi_thread")]
async fn unknown_paths_and_methods_are_refused() {
    let (node, _estate, _dir) = seeded_node().await;
    let (addr, _http) = rro_engine::serve_http("127.0.0.1:0", node).await.unwrap();

    let (status, _) = http(addr, "GET", "/nope", None, None).await;
    assert!(status.contains("404"), "unknown path: {status}");

    // A known verb path with the wrong method is a 405.
    let (status, _) = http(addr, "GET", "/query", None, None).await;
    assert!(status.contains("405"), "GET /query: {status}");
}
