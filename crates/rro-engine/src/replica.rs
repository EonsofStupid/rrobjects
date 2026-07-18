//! Follower-side replication — rebuild a leader's estate from its `replicate`
//! stream, and stay caught up.
//!
//! This is the primitive the cluster stands on. A [`Replica`] holds a local
//! estate and a cursor, polls a leader's `replicate` verb from that cursor, and
//! applies each [`connxism::ReplEntry`] to its estate. Because the leader resolves
//! every upsert to its *current* record, replaying the stream in order is
//! **convergent**: a follower reaches the leader's current state regardless of
//! where it started, and re-applying an already-seen entry is idempotent (upsert
//! overwrites, remove of an absent id is a no-op).
//!
//! Stage 1 is the apply mechanism and catch-up. Synchronous quorum-ack on the
//! leader's write path and leader-lease failover build on top of this.

use std::sync::Arc;

use rro_core::{Id, Recall, Result, RroError};
use rro_net::{tcp, Message};

/// A follower that mirrors a leader estate over a2a.
pub struct Replica {
    leader: std::net::SocketAddr,
    estate: Arc<connxism::Estate>,
    id: String,
    token: Option<String>,
    cursor: u64,
}

impl Replica {
    /// A replica of `leader` writing into the local `estate`, starting from the
    /// beginning of the leader's changefeed. The `id` is how the leader tracks
    /// this follower's replication progress for quorum-ack — it must be unique
    /// per follower.
    pub fn new(
        id: impl Into<String>,
        leader: std::net::SocketAddr,
        estate: Arc<connxism::Estate>,
    ) -> Self {
        Replica {
            leader,
            estate,
            id: id.into(),
            token: None,
            cursor: 0,
        }
    }

    /// Start from a known cursor (e.g. persisted across a restart) instead of 0.
    pub fn from_cursor(mut self, cursor: u64) -> Self {
        self.cursor = cursor;
        self
    }

    /// Present a capability token on every `replicate` request (a guarded leader
    /// requires at least reader access to its log).
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// The next changefeed seq this replica will request — its resume point.
    pub fn cursor(&self) -> u64 {
        self.cursor
    }

    /// Pull one batch (up to `limit`) from the leader and apply it, advancing the
    /// cursor. Returns how many entries were applied (0 = already at head).
    pub async fn sync_once(&mut self, limit: usize) -> Result<usize> {
        let mut msg = Message::request(
            self.id.as_str(),
            "leader",
            "replicate",
            serde_json::json!({ "since_seq": self.cursor, "limit": limit }),
        );
        if let Some(t) = &self.token {
            msg = msg.with_token(t.clone());
        }
        let reply = tcp::request(self.leader, &msg).await?;
        if let Some(err) = reply.body.get("error").and_then(|v| v.as_str()) {
            return Err(RroError::Net(format!("replicate refused: {err}")));
        }
        let entries: Vec<connxism::ReplEntry> = reply
            .body
            .get("entries")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| RroError::Net(format!("bad replicate reply: {e}")))?
            .unwrap_or_default();

        let n = entries.len();
        let recall = self.estate.recall();
        for e in entries {
            match e.op {
                connxism::ChangeOp::Upsert => {
                    // A superseded upsert (record None) is skipped — a later,
                    // higher-seq entry reconciles it.
                    if let Some(record) = e.record {
                        recall.upsert(vec![record.into_record()]).await?;
                    }
                }
                // Remove of an absent id is a no-op, so this stays idempotent.
                connxism::ChangeOp::Remove => {
                    recall.remove(&Id::from(e.doc_id)).await?;
                }
            }
            self.cursor = e.seq + 1;
        }
        Ok(n)
    }

    /// Sync repeatedly until caught up to the leader's head. A batch shorter than
    /// `limit` means the leader had nothing more past this cursor.
    pub async fn sync_to_head(&mut self, limit: usize) -> Result<usize> {
        let mut total = 0;
        loop {
            let n = self.sync_once(limit).await?;
            total += n;
            if n < limit {
                break;
            }
        }
        // The graph applier is out-of-band; make the mirrored vectors queryable
        // before returning so a caller can assert convergence immediately.
        self.estate.recall().quiesce().await?;
        Ok(total)
    }
}
