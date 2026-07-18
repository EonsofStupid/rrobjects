//! Phase 13 (stage 1) gate: a follower rebuilds a leader's estate from the
//! `replicate` stream and stays caught up.
//!
//! This proves the replication primitive the cluster stands on:
//!   - a fresh follower reaches the leader's EXACT state (same docs, same
//!     vectors) purely from the stream — no back-channel fetch,
//!   - new writes on the leader are picked up on the next sync,
//!   - removes replicate,
//!   - replay is idempotent and resumes from a cursor without gaps.

use std::sync::Arc;

use rro_core::{Embedding, Recall, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject, Replica};
use rro_net::tcp;

fn rec(id: &str, seed: f32) -> VectorRecord {
    VectorRecord::new(
        id,
        Embedding(vec![seed, 0.5, 0.25, 0.125]),
        format!("doc {id}"),
    )
    .in_collection("repl")
}

/// Serve a leader estate's a2a surface; return its address (task leaked to live
/// for the test).
async fn leader(estate: Arc<connxism::Estate>) -> std::net::SocketAddr {
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = Arc::new(FlowNode::new(flow, "leader").with_estate(estate));
    let (addr, task) = tcp::serve("127.0.0.1:0", node).await.unwrap();
    std::mem::forget(task);
    addr
}

/// The set of (id, vector) pairs an estate holds — the equality oracle.
async fn snapshot(estate: &connxism::Estate) -> Vec<(String, Vec<f32>)> {
    let recall = estate.recall();
    let ids = ["a", "b", "c", "d", "e"];
    let mut out = Vec::new();
    for id in ids {
        if let Some(v) = recall.vector_of(id).await.unwrap() {
            out.push((id.to_string(), v.0));
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn a_follower_converges_to_the_leader_then_stays_caught_up() {
    // Leader with three documents.
    let ldir = tempfile::tempdir().unwrap();
    let lestate = Arc::new(connxism::Estate::open(ldir.path(), "leader").unwrap());
    lestate
        .recall()
        .upsert(vec![rec("a", 0.1), rec("b", 0.2), rec("c", 0.3)])
        .await
        .unwrap();
    lestate.recall().quiesce().await.unwrap();
    let laddr = leader(lestate.clone()).await;

    // Fresh follower — empty estate, replicates from seq 0.
    let fdir = tempfile::tempdir().unwrap();
    let festate = Arc::new(connxism::Estate::open(fdir.path(), "follower").unwrap());
    let mut replica = Replica::new("f1", laddr, festate.clone());

    let applied = replica.sync_to_head(256).await.unwrap();
    assert_eq!(applied, 3, "three upserts replicated");
    assert_eq!(festate.recall().len().await.unwrap(), 3);

    // EXACT state: same ids, same vectors, byte for byte.
    assert_eq!(
        snapshot(&lestate).await,
        snapshot(&festate).await,
        "follower must mirror the leader's vectors exactly"
    );

    // New writes on the leader are picked up on the next sync.
    lestate
        .recall()
        .upsert(vec![rec("d", 0.4), rec("e", 0.5)])
        .await
        .unwrap();
    lestate.recall().quiesce().await.unwrap();
    let applied = replica.sync_to_head(256).await.unwrap();
    assert_eq!(applied, 2, "two new upserts replicated");
    assert_eq!(festate.recall().len().await.unwrap(), 5);
    assert_eq!(snapshot(&lestate).await, snapshot(&festate).await);

    // A remove replicates too.
    lestate
        .recall()
        .remove(&rro_core::Id::from("b"))
        .await
        .unwrap();
    lestate.recall().quiesce().await.unwrap();
    replica.sync_to_head(256).await.unwrap();
    assert_eq!(
        festate.recall().len().await.unwrap(),
        4,
        "remove replicated"
    );
    assert!(
        festate.recall().vector_of("b").await.unwrap().is_none(),
        "removed doc is gone on the follower"
    );
    assert_eq!(snapshot(&lestate).await, snapshot(&festate).await);

    // The cursor is at the leader's head — a redundant sync applies nothing.
    let head = replica.cursor();
    assert_eq!(
        replica.sync_to_head(256).await.unwrap(),
        0,
        "at head, no-op"
    );
    assert_eq!(replica.cursor(), head, "cursor does not move past head");
}

#[tokio::test(flavor = "multi_thread")]
async fn replay_is_idempotent_and_resumes_from_a_cursor() {
    let ldir = tempfile::tempdir().unwrap();
    let lestate = Arc::new(connxism::Estate::open(ldir.path(), "leader").unwrap());
    lestate
        .recall()
        .upsert(vec![rec("a", 0.1), rec("b", 0.2), rec("c", 0.3)])
        .await
        .unwrap();
    lestate.recall().quiesce().await.unwrap();
    let laddr = leader(lestate.clone()).await;

    // A follower that already holds the data (e.g. after a restart) resumes from
    // cursor 0 and re-applies every entry — upserts overwrite, so the end state
    // is identical, not doubled.
    let fdir = tempfile::tempdir().unwrap();
    let festate = Arc::new(connxism::Estate::open(fdir.path(), "follower").unwrap());
    festate
        .recall()
        .upsert(vec![rec("a", 0.1), rec("b", 0.2), rec("c", 0.3)])
        .await
        .unwrap();
    festate.recall().quiesce().await.unwrap();

    let mut replica = Replica::new("f1", laddr, festate.clone());
    replica.sync_to_head(256).await.unwrap();
    assert_eq!(
        festate.recall().len().await.unwrap(),
        3,
        "idempotent replay does not duplicate records"
    );
    assert_eq!(snapshot(&lestate).await, snapshot(&festate).await);

    // Small batches force multiple round-trips; the cursor must not skip an entry.
    let fdir2 = tempfile::tempdir().unwrap();
    let festate2 = Arc::new(connxism::Estate::open(fdir2.path(), "follower2").unwrap());
    let mut replica2 = Replica::new("f2", laddr, festate2.clone());
    let applied = replica2.sync_to_head(1).await.unwrap(); // one entry per pull
    assert_eq!(applied, 3, "batched-by-one still applies every entry");
    assert_eq!(snapshot(&lestate).await, snapshot(&festate2).await);
}
