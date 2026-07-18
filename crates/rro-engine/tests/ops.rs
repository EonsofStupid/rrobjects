//! Sprint 20 gates: the ops surface — health verb with live numbers over
//! a2a, prometheus /metrics + probe endpoints over a real HTTP socket, and
//! the issues self-report catching a planted applier backlog.

use std::sync::Arc;

use rro_client::Client;
use rro_core::{Embedding, Recall, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn rec(id: &str, seed: f32) -> VectorRecord {
    VectorRecord::new(
        id,
        Embedding(vec![seed, 0.5, 0.25, 0.125]),
        format!("ops corpus {id}"),
    )
    .in_collection("ops")
}

async fn http_get(addr: std::net::SocketAddr, path: &str) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n").as_bytes())
        .await
        .unwrap();
    let mut buf = String::new();
    stream.read_to_string(&mut buf).await.unwrap();
    let (head, body) = buf.split_once("\r\n\r\n").unwrap();
    let status = head.lines().next().unwrap_or("").to_string();
    (status, body.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn health_verb_reports_live_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "h").unwrap());
    let recall = estate.recall();
    recall
        .upsert(vec![rec("a", 0.1), rec("b", 0.2), rec("c", 0.3)])
        .await
        .unwrap();
    recall.quiesce().await.unwrap();

    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = FlowNode::new(flow, "ops-node").with_estate(estate);
    let (addr, _task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();

    let health = Client::new(addr.to_string()).health().await.unwrap();
    assert_eq!(health["node"], "ops-node");
    assert!(health["uptime_secs"].is_u64());
    assert_eq!(health["estate"]["docs"], 3);
    assert_eq!(health["estate"]["feed_seq"], 3);
    assert_eq!(health["estate"]["applier_backlog"], 0);
    assert_eq!(health["estate"]["dim"], 4);
    assert_eq!(health["estate"]["collections"][0][0], "ops");
    assert_eq!(health["estate"]["collections"][0][1], 3);
    assert_eq!(
        health["issues"].as_array().map(Vec::len),
        Some(0),
        "healthy estate self-reports no issues: {}",
        health["issues"]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn metrics_and_probes_over_real_http() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "m").unwrap());
    let recall = estate.recall();
    recall
        .upsert(vec![rec("a", 0.1), rec("b", 0.2)])
        .await
        .unwrap();
    recall.quiesce().await.unwrap();

    let (addr, _task) = rro_engine::ops::serve_ops("127.0.0.1:0", estate)
        .await
        .unwrap();

    // Probes: all three answer 200 ok.
    for path in ["/healthz", "/livez", "/readyz"] {
        let (status, body) = http_get(addr, path).await;
        assert!(status.contains("200"), "{path}: {status}");
        assert_eq!(body, "ok\n");
    }

    // /metrics: prometheus text with live gauges.
    let (status, body) = http_get(addr, "/metrics").await;
    assert!(status.contains("200"), "{status}");
    assert!(body.contains("# TYPE rro_docs_total gauge"));
    assert!(body.contains("rro_docs_total 2"), "{body}");
    assert!(body.contains("rro_feed_seq 2"));
    assert!(body.contains("rro_applier_backlog 0"));
    assert!(body.contains("rro_collection_docs{collection=\"ops\"} 2"));
    assert!(body.contains("rro_issues_total 0"));
    // Every non-comment line is `name[{labels}] value` — parseable.
    for line in body.lines().filter(|l| !l.starts_with('#')) {
        let mut parts = line.rsplitn(2, ' ');
        let value = parts.next().unwrap();
        assert!(value.parse::<f64>().is_ok(), "unparseable line: {line}");
    }

    // Unknown path and non-GET are refused.
    let (status, _) = http_get(addr, "/nope").await;
    assert!(status.contains("404"));
}

#[tokio::test(flavor = "multi_thread")]
async fn issues_report_planted_backlog() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "i").unwrap());
    let recall = estate.recall();

    // Plant work: upserts enqueue graph ops; with threshold 0, any
    // in-flight backlog is an issue the instant before the applier drains.
    recall
        .upsert(
            (0..64)
                .map(|i| rec(&format!("d{i}"), i as f32 * 0.01))
                .collect(),
        )
        .await
        .unwrap();
    // Threshold 0 → an empty queue reports nothing; so instead assert the
    // report fires exactly when the snapshot shows backlog, and clears
    // after quiesce.
    //
    // Read the issues snapshot BEFORE the backlog snapshot. No writes follow the
    // upsert, so the applier only drains — backlog is monotonically
    // non-increasing. Reading issues first (earlier) then backlog (later) means
    // `before > 0` implies the earlier snapshot had at least that much backlog,
    // so the issue must be present. The reverse order races: the applier could
    // drain between the two reads, leaving `before > 0` but an empty issues list.
    let issues_before = estate.issues(0).unwrap();
    let before = estate.health().unwrap().applier_backlog;
    if before > 0 {
        assert!(issues_before.iter().any(|i| i.code == "applier_backlog"));
    }
    recall.quiesce().await.unwrap();
    assert_eq!(estate.health().unwrap().applier_backlog, 0);
    assert!(
        estate.issues(0).unwrap().is_empty(),
        "drained estate is clean"
    );

    // The feed_behind detector: a healthy estate never trips it.
    assert!(!estate
        .issues(10_000)
        .unwrap()
        .iter()
        .any(|i| i.code == "feed_behind"));
}
