//! Phase 13 (stage 2) gate: synchronous quorum-ack.
//!
//! A `durability: "quorum"` write must not be acknowledged until a quorum of
//! members hold it — that is what makes "no acked write lost" true across a
//! leader failure. And a write that CANNOT reach a quorum must refuse to ack as
//! durable rather than lie.

use std::sync::Arc;
use std::time::Duration;

use rro_engine::{Cluster, FlowNode, ReasonReadyObject, Replica};
use rro_net::{tcp, Message};

async fn leader(cluster: Arc<Cluster>) -> (std::net::SocketAddr, Arc<connxism::Estate>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.keep();
    let estate = Arc::new(connxism::Estate::open(&path, "leader").unwrap());
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = Arc::new(
        FlowNode::new(flow, "leader")
            .with_estate(estate.clone())
            .with_cluster(cluster),
    );
    let (addr, task) = tcp::serve("127.0.0.1:0", node).await.unwrap();
    std::mem::forget(task);
    (addr, estate)
}

fn quorum_write(id: &str, timeout_ms: u64) -> Message {
    Message::request(
        "client",
        "leader",
        "tx",
        serde_json::json!({
            "ops": [{ "upsert": [{ "id": id, "text": format!("durable {id}") }] }],
            "durability": "quorum",
            "quorum_timeout_ms": timeout_ms,
        }),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn a_quorum_durable_write_is_on_a_quorum_before_it_acks() {
    // 3-member cluster, quorum 2 → the leader plus one follower.
    let cluster = Arc::new(Cluster::new(2));
    let (laddr, _lestate) = leader(cluster).await;

    let fdir = tempfile::tempdir().unwrap();
    let festate = Arc::new(connxism::Estate::open(fdir.path(), "f1").unwrap());
    let mut replica = Replica::new("f1", laddr, festate.clone());

    // Fire the durable write; server-side it blocks in await_quorum.
    let write = tokio::spawn(async move { tcp::request(laddr, &quorum_write("w1", 5_000)).await });

    // Drive the follower until the write completes. Each sync's final poll reports
    // the follower's cursor to the leader, advancing the quorum.
    loop {
        replica.sync_to_head(256).await.unwrap();
        if write.is_finished() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let reply = write.await.unwrap().unwrap().body;
    assert_eq!(
        reply["durable"], "quorum",
        "the write must ack as quorum-durable: {reply}"
    );

    // The follower holds the acked write — the property "no acked write lost"
    // rests on: if the leader dies now, the survivor already has w1.
    assert!(
        festate.recall().vector_of("w1").await.unwrap().is_some(),
        "the follower must hold the acked write"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_write_that_cannot_reach_a_quorum_refuses_to_ack_durable() {
    // 5-member cluster, quorum 3 → needs TWO followers, but only one exists.
    let cluster = Arc::new(Cluster::new(3));
    let (laddr, _lestate) = leader(cluster).await;

    let fdir = tempfile::tempdir().unwrap();
    let festate = Arc::new(connxism::Estate::open(fdir.path(), "f1").unwrap());
    let mut replica = Replica::new("f1", laddr, festate.clone());

    // Short timeout: the one follower can never satisfy a 2-follower quorum.
    let write = tokio::spawn(async move { tcp::request(laddr, &quorum_write("w1", 300)).await });

    // Drive the single follower; it replicates the write but cannot form quorum.
    for _ in 0..8 {
        replica.sync_to_head(256).await.unwrap();
        if write.is_finished() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let reply = write.await.unwrap().unwrap().body;

    // The leader must NOT claim durability it did not achieve.
    assert!(
        reply.get("durable").is_none(),
        "an under-replicated write must not ack as durable: {reply}"
    );
    assert!(
        reply.get("error").is_some(),
        "it must report the quorum failure: {reply}"
    );
    // It is still durable LOCALLY (and on the one follower) — just not quorum-acked.
    assert_eq!(reply["committed_local"], 1);
    assert!(
        festate.recall().vector_of("w1").await.unwrap().is_some(),
        "the write did land locally + on the reachable follower"
    );
}
