//! Leader-side cluster coordination: synchronous quorum-ack.
//!
//! Stage 1 gave a follower that mirrors a leader. Stage 2 makes a write *durable
//! across the cluster before it is acknowledged*: the leader does not tell the
//! client "committed" until a quorum of members hold the write. That is what lets
//! a leader die with no acked write lost — the survivors already have it.
//!
//! ## The poll cursor IS the ack
//!
//! A follower pulls the replication stream by polling `replicate` with
//! `since_seq = C`, which means "I have applied everything below C." So the leader
//! learns each follower's durable position from its polls — no separate ack
//! channel. The position is a **lower bound** (the follower may have applied more
//! since its last poll), which is exactly the safe direction: quorum-ack may wait
//! a little longer than strictly necessary, but never acks a write a quorum does
//! not yet hold.
//!
//! ## Quorum
//!
//! The leader is always one member and always holds every write. A write whose
//! feed cursor is `target` (one past its seq) is quorum-durable when the leader
//! plus at least `quorum - 1` followers have reached `target`. `quorum` is the
//! usual majority `floor(N/2) + 1` for an `N`-member cluster, so any two quorums
//! intersect.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

#[cfg(test)]
use std::sync::Arc;

use rro_core::{Result, RroError};
use tokio::sync::Notify;

/// The leader's view of follower progress, and the quorum gate over it.
pub struct Cluster {
    /// Total members (leader + followers) whose agreement commits a write.
    quorum: usize,
    /// follower id → its reported cursor (next-wanted seq = applied-through + 1).
    followers: Mutex<HashMap<String, u64>>,
    /// Woken whenever a follower's cursor advances, so quorum waiters re-check.
    progress: Notify,
}

impl Cluster {
    /// A coordinator that commits when `quorum` members (including the leader)
    /// hold a write. `quorum` is clamped to at least 1 (a lone leader).
    pub fn new(quorum: usize) -> Self {
        Cluster {
            quorum: quorum.max(1),
            followers: Mutex::new(HashMap::new()),
            progress: Notify::new(),
        }
    }

    /// The majority quorum for an `n`-member cluster: `floor(n/2) + 1`.
    pub fn majority(n: usize) -> usize {
        n / 2 + 1
    }

    /// Record a follower's reported cursor (from its `replicate` poll). Monotonic:
    /// a stale lower cursor never rewinds a follower's known position.
    pub fn observe(&self, follower: &str, cursor: u64) {
        {
            let mut f = self.followers.lock().expect("cluster lock");
            let e = f.entry(follower.to_string()).or_insert(0);
            if cursor > *e {
                *e = cursor;
            } else {
                return; // no advance, no waiters to wake
            }
        }
        self.progress.notify_waiters();
    }

    /// How many members (leader + followers) have reached `target` cursor.
    fn holders(&self, target: u64) -> usize {
        let f = self.followers.lock().expect("cluster lock");
        // The leader always holds every write — it is member 1.
        1 + f.values().filter(|&&c| c >= target).count()
    }

    /// Whether a write at feed cursor `target` is held by a quorum right now.
    pub fn is_committed(&self, target: u64) -> bool {
        self.holders(target) >= self.quorum
    }

    /// The follower positions the leader currently knows (id → cursor), for
    /// `/cluster` health and tests.
    pub fn follower_cursors(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self
            .followers
            .lock()
            .expect("cluster lock")
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        v.sort();
        v
    }

    /// Block until a write at feed cursor `target` is quorum-durable, or `timeout`
    /// elapses. On timeout the write is NOT acked — the caller reports it did not
    /// reach a quorum rather than lie that it did.
    pub async fn await_quorum(&self, target: u64, timeout: Duration) -> Result<()> {
        let wait = async {
            loop {
                // Arm the notify BEFORE the check so an advance between them still
                // wakes us (no lost wakeup).
                let armed = self.progress.notified();
                if self.is_committed(target) {
                    return;
                }
                armed.await;
            }
        };
        tokio::time::timeout(timeout, wait).await.map_err(|_| {
            RroError::Net(format!(
                "quorum not reached for cursor {target} within {timeout:?} \
                 (need {}, have {})",
                self.quorum,
                self.holders(target)
            ))
        })
    }
}

/// A leader lease — the liveness signal that triggers failover.
///
/// The leader renews the lease on a heartbeat; a follower treats the leader as
/// dead once `now > expires_at`. Time is passed in (monotonic millis) rather than
/// read from a global clock, so the lease is deterministic and testable, and the
/// same logic drives a real heartbeat loop.
#[derive(Debug, Clone)]
pub struct Lease {
    leader: String,
    expires_at_ms: u64,
}

impl Lease {
    /// Grant a lease to `leader`, valid for `ttl_ms` from `now_ms`.
    pub fn granted(leader: impl Into<String>, now_ms: u64, ttl_ms: u64) -> Self {
        Lease {
            leader: leader.into(),
            expires_at_ms: now_ms.saturating_add(ttl_ms),
        }
    }

    /// Renew (heartbeat): push expiry out to `now_ms + ttl_ms`.
    pub fn renew(&mut self, now_ms: u64, ttl_ms: u64) {
        self.expires_at_ms = now_ms.saturating_add(ttl_ms);
    }

    /// Whether the lease is still valid at `now_ms` (leader considered alive).
    pub fn is_live(&self, now_ms: u64) -> bool {
        now_ms < self.expires_at_ms
    }

    /// The current lease-holder.
    pub fn leader(&self) -> &str {
        &self.leader
    }
}

/// Elect the next leader from the surviving members and their replication
/// cursors: the **highest-cursor** survivor wins, ties broken by id (a total,
/// deterministic order so every survivor computes the same winner).
///
/// This is the choice that makes "no acked write lost" hold. A quorum-acked write
/// reached a majority; if a majority survives, the two majorities intersect, so at
/// least one survivor holds every acked write — and the highest-cursor survivor
/// holds *all* of them. Returns `None` if there are no survivors.
pub fn elect(survivors: &[(String, u64)]) -> Option<String> {
    survivors
        .iter()
        .max_by(|(a_id, a_cur), (b_id, b_cur)| a_cur.cmp(b_cur).then_with(|| a_id.cmp(b_id)))
        .map(|(id, _)| id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn majority_is_a_strict_majority() {
        assert_eq!(Cluster::majority(1), 1);
        assert_eq!(Cluster::majority(2), 2);
        assert_eq!(Cluster::majority(3), 2);
        assert_eq!(Cluster::majority(5), 3);
    }

    #[test]
    fn a_lone_leader_commits_immediately() {
        let c = Cluster::new(1);
        assert!(c.is_committed(0));
        assert!(c.is_committed(999), "quorum=1 is the leader alone");
    }

    #[test]
    fn a_write_commits_once_enough_followers_hold_it() {
        // 3-member cluster: leader + 2 followers, quorum 2 → need 1 follower.
        let c = Cluster::new(2);
        assert!(
            !c.is_committed(5),
            "no follower yet — leader alone is not a quorum"
        );

        c.observe("a", 5); // follower a has applied through 4 (cursor 5)
        assert!(c.is_committed(5), "leader + a = quorum for cursor 5");
        assert!(!c.is_committed(6), "a has not reached cursor 6");

        c.observe("b", 7);
        assert!(c.is_committed(6), "b now covers cursor 6");
    }

    #[test]
    fn a_five_member_quorum_needs_two_followers() {
        let c = Cluster::new(Cluster::majority(5)); // quorum 3 → 2 followers
        c.observe("a", 10);
        assert!(
            !c.is_committed(10),
            "leader + 1 follower is not a majority of 5"
        );
        c.observe("b", 10);
        assert!(
            c.is_committed(10),
            "leader + 2 followers is a majority of 5"
        );
    }

    #[test]
    fn observed_cursor_is_monotonic() {
        let c = Cluster::new(2);
        c.observe("a", 9);
        c.observe("a", 4); // a stale/late poll must not rewind a's position
        assert!(c.is_committed(9), "a stays at 9, not rewound to 4");
    }

    #[test]
    fn a_lease_expires_without_a_heartbeat() {
        let mut lease = Lease::granted("leader", 1_000, 500); // valid until 1500
        assert!(lease.is_live(1_200), "still within TTL");
        assert!(
            !lease.is_live(1_500),
            "expiry is exclusive — dead at the boundary"
        );
        assert!(!lease.is_live(2_000), "long past → dead");
        // A heartbeat pushes expiry out.
        lease.renew(1_400, 500); // now valid until 1900
        assert!(lease.is_live(1_800), "renewed lease is live again");
        assert_eq!(lease.leader(), "leader");
    }

    #[test]
    fn election_picks_the_highest_cursor_survivor() {
        // f2 has replicated furthest → it becomes leader (least data loss).
        let winner = elect(&[("f1".into(), 5), ("f2".into(), 9), ("f3".into(), 7)]);
        assert_eq!(winner.as_deref(), Some("f2"));

        // Ties break by id, deterministically (every survivor agrees).
        let tie = elect(&[("f3".into(), 9), ("f1".into(), 9), ("f2".into(), 9)]);
        assert_eq!(tie.as_deref(), Some("f3"), "highest id wins a cursor tie");

        // No survivors → no leader (a minority cannot elect).
        assert_eq!(elect(&[]), None);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn await_quorum_wakes_on_progress_and_times_out_without_it() {
        let c = Arc::new(Cluster::new(2));

        // No follower → times out, does not falsely succeed.
        let r = c.await_quorum(3, Duration::from_millis(50)).await;
        assert!(r.is_err(), "no quorum must time out");

        // A follower advancing past the target wakes the waiter.
        let c2 = c.clone();
        let waiter =
            tokio::spawn(async move { c2.await_quorum(3, Duration::from_millis(2000)).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        c.observe("a", 4); // covers cursor 3
        assert!(
            waiter.await.unwrap().is_ok(),
            "progress must complete the wait"
        );
    }
}
