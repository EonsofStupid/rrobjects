//! Phase 13 (stage 3) gate: kill the leader, lose no acked write.
//!
//! This is the headline cluster property. It composes the earlier stages:
//! quorum-ack (stage 2) put every acked write on a majority; the lease detects
//! the leader's death; the election promotes the highest-cursor survivor. Because
//! a quorum-acked write reached a majority and the survivors are a majority, the
//! two majorities intersect — so the highest-cursor survivor holds *every* acked
//! write. This test forces the interesting case: an acked write that lives on the
//! leader and only ONE follower, then kills the leader and proves the election
//! promotes the follower that has it.

use std::sync::Arc;
use std::time::Duration;

use rro_core::Recall;
use rro_engine::{elect, Cluster, FlowNode, Lease, ReasonReadyObject, Replica};
use rro_net::{tcp, Message};

fn durable_write(id: &str) -> Message {
    Message::request(
        "client",
        "leader",
        "tx",
        serde_json::json!({
            "ops": [{ "upsert": [{ "id": id, "text": format!("acked {id}") }] }],
            "durability": "quorum",
            "quorum_timeout_ms": 5_000,
        }),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn killing_the_leader_loses_no_acked_write() {
    // 3-member cluster, quorum 2: a write acks with the leader + ONE follower.
    let cluster = Arc::new(Cluster::new(Cluster::majority(3)));
    let ldir = tempfile::tempdir().unwrap();
    let lestate = Arc::new(connxism::Estate::open(ldir.path(), "leader").unwrap());
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = Arc::new(
        FlowNode::new(flow, "leader")
            .with_estate(lestate.clone())
            .with_cluster(cluster.clone()),
    );
    let (laddr, leader_task) = tcp::serve("127.0.0.1:0", node).await.unwrap();

    // Two followers. f1 will stay caught up; f2 will lag — so acked writes land on
    // the leader + f1 but NOT f2.
    let f1dir = tempfile::tempdir().unwrap();
    let f1estate = Arc::new(connxism::Estate::open(f1dir.path(), "f1").unwrap());
    let mut f1 = Replica::new("f1", laddr, f1estate.clone());

    let f2dir = tempfile::tempdir().unwrap();
    let f2estate = Arc::new(connxism::Estate::open(f2dir.path(), "f2").unwrap());
    let mut f2 = Replica::new("f2", laddr, f2estate.clone());

    // The leader holds a lease; followers see it as alive while it heartbeats.
    let ttl = 500u64;
    let mut lease = Lease::granted("leader", 0, ttl);
    assert!(lease.is_live(100), "leader alive under write load");

    // Write five records, each quorum-durable. Only f1 syncs, so every write acks
    // via leader+f1 and f2 falls behind.
    let ids: Vec<String> = (0..5).map(|i| format!("w{i}")).collect();
    for id in &ids {
        let for_write = id.clone();
        let write =
            tokio::spawn(async move { tcp::request(laddr, &durable_write(&for_write)).await });
        loop {
            f1.sync_to_head(256).await.unwrap();
            if write.is_finished() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let reply = write.await.unwrap().unwrap().body;
        assert_eq!(
            reply["durable"], "quorum",
            "write {id} acked durable: {reply}"
        );
        // heartbeat while healthy
        lease.renew(100, ttl);
    }

    // f2 lags: it holds strictly fewer acked writes than f1. (Proves the election
    // choice is load-bearing — promoting f2 WOULD lose data.)
    f2.sync_once(2).await.unwrap(); // a partial catch-up, on purpose
    assert!(
        f2.cursor() < f1.cursor(),
        "f2 ({}) must lag f1 ({}) for this test to be meaningful",
        f2.cursor(),
        f1.cursor()
    );

    // ---- kill the leader ---------------------------------------------------
    // Stop serving and stop heartbeating. (Killing = ceasing leader service; the
    // same lease/elect logic drives real processes.)
    leader_task.abort();
    let now_after_death = 100 + ttl + 1; // one tick past the last heartbeat + TTL
    assert!(
        !lease.is_live(now_after_death),
        "the lease must expire once the leader stops heartbeating"
    );

    // ---- elect among the survivors ----------------------------------------
    let survivors = [
        ("f1".to_string(), f1.cursor()),
        ("f2".to_string(), f2.cursor()),
    ];
    let new_leader = elect(&survivors).expect("a majority survived, so a leader is electable");
    assert_eq!(
        new_leader, "f1",
        "the highest-cursor survivor must win (f2 lags and would lose writes)"
    );

    // ---- no acked write lost ----------------------------------------------
    // The promoted leader holds EVERY write the old leader acked.
    let winner = &f1estate;
    for id in &ids {
        assert!(
            winner.recall().vector_of(id).await.unwrap().is_some(),
            "the new leader must hold acked write {id} — none may be lost"
        );
    }

    // And reads work on the new leader immediately (availability survives).
    assert_eq!(
        winner.recall().len().await.unwrap(),
        5,
        "the new leader serves the full acked dataset"
    );

    // Concretely why the election mattered: f2, had it been promoted, was missing
    // at least one acked write.
    let mut f2_missing = 0;
    for id in &ids {
        if f2estate.recall().vector_of(id).await.unwrap().is_none() {
            f2_missing += 1;
        }
    }
    assert!(
        f2_missing > 0,
        "the laggard f2 was missing acked writes — electing it WOULD have lost data"
    );
}
