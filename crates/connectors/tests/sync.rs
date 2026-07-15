//! P4 gates: full sync through the whole engine, and resume-after-interrupt.

use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use connectors::{sync, Batch, Driver, FsDriver, JsonlDriver};
use connxism::{ChangeOp, ConnectorInfo, ConnectorKind, Estate, SyncState, SyncStatus};
use embedder::DeterministicEmbedder;
use rrd::Rrd;
use rrf_core::{Recall, Result, RrfError};

fn register(estate: &Estate, id: &str, provider: &str, uri: &str) {
    estate
        .register_connector(ConnectorInfo {
            id: id.into(),
            name: format!("{id} source"),
            kind: ConnectorKind::Docs,
            provider: provider.into(),
            uri: uri.into(),
            sync: SyncState::default(),
            registered_at: 0,
        })
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn fs_sync_flows_through_the_whole_engine() {
    let src = tempfile::tempdir().unwrap();
    for i in 0..7 {
        std::fs::write(
            src.path().join(format!("note{i}.txt")),
            format!("meeting notes {i}: rollout of the estate connector pipeline"),
        )
        .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fs-sync").unwrap();
    let recall = estate.recall();
    let rrd = Rrd::new();
    let embed = DeterministicEmbedder::new();
    register(&estate, "c-fs", "fs", &src.path().to_string_lossy());

    let driver = FsDriver::new(src.path(), 3);
    let report = sync(&estate, &recall, &rrd, &embed, &driver, "c-fs")
        .await
        .unwrap();

    assert_eq!(report.ingested, 7);
    assert_eq!(recall.len().await.unwrap(), 7);

    // Provenance edges: connector -contains-> every doc.
    let contained = estate.relations_out("c-fs", Some("contains")).unwrap();
    assert_eq!(contained.len(), 7);

    // RRD ran: every doc carries its mode tag in the estate.
    let mode_tagged = estate.docs_by_tag("mode:document").unwrap();
    assert_eq!(
        mode_tagged.len(),
        7,
        "fs docs have path/modified metadata → document-mode shapes"
    );

    // Sync state settled.
    let conn = estate.connector("c-fs").unwrap().unwrap();
    assert_eq!(conn.sync.status, SyncStatus::Idle);
    assert_eq!(conn.sync.docs_synced, 7);

    // Changefeed recorded every ingest, in order.
    let changes = estate.changes(0, 100).unwrap();
    assert_eq!(changes.len(), 7);
    assert!(changes.windows(2).all(|w| w[0].seq < w[1].seq));
    assert!(changes.iter().all(|c| c.op == ChangeOp::Upsert));
}

/// Wraps a driver and fails hard on the Nth pull — the interruption.
struct FailAt<D> {
    inner: D,
    calls: AtomicUsize,
    fail_on: usize,
}

#[async_trait]
impl<D: Driver> Driver for FailAt<D> {
    fn provider(&self) -> &str {
        self.inner.provider()
    }
    async fn pull(&self, cursor: Option<&str>) -> Result<Batch> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        if n == self.fail_on {
            return Err(RrfError::msg("simulated connector outage"));
        }
        self.inner.pull(cursor).await
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn interrupted_sync_resumes_from_cursor_without_duplicates() {
    let src = tempfile::tempdir().unwrap();
    let feed = src.path().join("feed.jsonl");
    let lines: Vec<String> = (0..10)
        .map(|i| {
            format!(r#"{{"id":"r{i}","text":"record {i} payload","kind":"row","amount":{i}}}"#)
        })
        .collect();
    std::fs::write(&feed, lines.join("\n")).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "resume").unwrap();
    let recall = estate.recall();
    let rrd = Rrd::new();
    let embed = DeterministicEmbedder::new();
    register(&estate, "c-feed", "jsonl", "feed.jsonl");

    // First run dies on the third pull: batches 1–2 (6 docs) land durably.
    let flaky = FailAt {
        inner: JsonlDriver::new(&feed, 3),
        calls: AtomicUsize::new(0),
        fail_on: 3,
    };
    let err = sync(&estate, &recall, &rrd, &embed, &flaky, "c-feed").await;
    assert!(err.is_err(), "the outage must surface");
    assert_eq!(
        recall.len().await.unwrap(),
        6,
        "two batches landed before the outage"
    );
    let cursor_after_crash = estate.connector("c-feed").unwrap().unwrap().sync.cursor;
    assert_eq!(
        cursor_after_crash.as_deref(),
        Some("6"),
        "cursor holds at the last durable batch"
    );

    // Second run resumes: only the remaining 4, no duplicates, cursor drained.
    let report = sync(
        &estate,
        &recall,
        &rrd,
        &embed,
        &JsonlDriver::new(&feed, 3),
        "c-feed",
    )
    .await
    .unwrap();
    assert_eq!(report.ingested, 4, "resume ingests exactly the remainder");
    assert_eq!(recall.len().await.unwrap(), 10);

    let conn = estate.connector("c-feed").unwrap().unwrap();
    assert_eq!(conn.sync.status, SyncStatus::Idle);
    assert_eq!(conn.sync.docs_synced, 10);
    assert_eq!(conn.sync.cursor, None);

    // Changefeed: exactly 10 upserts — replay-free.
    let changes = estate.changes(0, 100).unwrap();
    assert_eq!(changes.len(), 10);

    // RRD shaped the rows: record-mode tags present.
    assert_eq!(estate.docs_by_tag("mode:record").unwrap().len(), 10);
}
