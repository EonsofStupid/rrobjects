//! # connectors — drivers and the sync engine
//!
//! An operator shares a **connector** (a third-party source: files, feeds,
//! mailboxes, databases). A [`Driver`] pulls its content in **resumable
//! batches** behind a cursor; [`sync`] runs the full estate-side pipeline for
//! each batch:
//!
//! ```text
//! driver.pull(cursor) ─▶ RRD distill (mode + tags, gate ladder)
//!                      ─▶ recall.upsert (durable + indexed)
//!                      ─▶ RELATE connector ─contains→ doc
//!                      ─▶ estate.tag(...) from the RRO
//!                      ─▶ cursor advance (durable in SyncState)
//! ```
//!
//! The cursor advances only after the batch is durably ingested, so an
//! interrupted sync resumes exactly where it stopped — no duplicates, no
//! gaps. Every batch is evented (`connector.batch`, `connector.synced`).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod fs;
mod jsonl;

pub use fs::FsDriver;
pub use jsonl::JsonlDriver;

use async_trait::async_trait;
use connxism::{ConnXRecall, Estate, SyncState, SyncStatus};
use rrd::Rrd;
use rrf_core::{Document, Embedder, Recall, Result, RrfError};

/// One resumable pull from a source.
pub struct Batch {
    /// Documents in this batch (empty = source drained at this cursor).
    pub docs: Vec<Document>,
    /// Cursor to persist once this batch is durably ingested; `None` when the
    /// source is fully drained.
    pub next_cursor: Option<String>,
}

/// A connector driver: pull content in resumable batches.
#[async_trait]
pub trait Driver: Send + Sync {
    /// Provider slug for the connector registry (e.g. `fs`, `jsonl`).
    fn provider(&self) -> &str;

    /// Pull the next batch after `cursor` (`None` = from the beginning).
    async fn pull(&self, cursor: Option<&str>) -> Result<Batch>;
}

/// Outcome of one [`sync`] run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncReport {
    /// Documents ingested by this run.
    pub ingested: u64,
    /// Batches pulled.
    pub batches: u64,
    /// Final cursor (persisted in the connector's [`SyncState`]).
    pub cursor: Option<String>,
}

/// Run one full sync pass for `connector_id`, resuming from its stored
/// cursor. Each document flows through the whole engine: RRD distills it
/// (mode + tags + gate verdict), recall ingests it, the estate RELATEs it to
/// its connector and tags it from the RRO. The cursor advances **after** the
/// batch is durable.
pub async fn sync(
    estate: &Estate,
    recall: &ConnXRecall,
    rrd: &Rrd,
    embedder: &dyn Embedder,
    driver: &dyn Driver,
    connector_id: &str,
) -> Result<SyncReport> {
    let conn = estate
        .connector(connector_id)?
        .ok_or_else(|| RrfError::msg(format!("no such connector: {connector_id}")))?;

    let mut cursor = conn.sync.cursor.clone();
    let mut docs_synced = conn.sync.docs_synced;
    let mut ingested = 0u64;
    let mut batches = 0u64;

    estate.update_sync(
        connector_id,
        SyncState {
            cursor: cursor.clone(),
            docs_synced,
            last_sync: conn.sync.last_sync,
            status: SyncStatus::Syncing,
        },
    )?;

    loop {
        let batch = driver.pull(cursor.as_deref()).await?;
        if batch.docs.is_empty() {
            break;
        }
        batches += 1;

        // Embed once; the same vectors serve recall AND RRD tag routing.
        let texts: Vec<String> = batch.docs.iter().map(|d| d.text.clone()).collect();
        let embeddings = embedder.embed(&texts).await?;

        let mut records = Vec::with_capacity(batch.docs.len());
        let mut post = Vec::with_capacity(batch.docs.len()); // (doc_id, tags, mode)
        for (doc, emb) in batch.docs.iter().zip(&embeddings) {
            let rro = rrd.distill(doc.id.as_str(), &doc.text, &doc.metadata, Some(emb));
            let mut tags: Vec<String> = rro.tags.iter().map(|t| t.tag.clone()).collect();
            tags.push(format!("mode:{}", rro.mode.name()));
            post.push((doc.id.as_str().to_string(), tags));

            let mut r = rrf_core::VectorRecord::new(doc.id.clone(), emb.clone(), doc.text.clone());
            r.metadata = doc.metadata.clone();
            records.push(r);
        }

        // Durable ingest first…
        recall.upsert(records).await?;
        ingested += post.len() as u64;
        docs_synced += post.len() as u64;

        // …then the map: provenance edges + tags from the RROs.
        for (doc_id, tags) in &post {
            estate.relate(connector_id, "contains", doc_id)?;
            estate.tag(doc_id, tags)?;
        }

        // …and only now the cursor. A crash before this line replays the
        // batch (idempotent upserts), never skips it.
        cursor = batch.next_cursor.clone();
        estate.update_sync(
            connector_id,
            SyncState {
                cursor: cursor.clone(),
                docs_synced,
                last_sync: Some(connxism::now_ms()),
                status: SyncStatus::Syncing,
            },
        )?;
        rrf_core::events::emit(
            "connector.batch",
            serde_json::json!({
                "connector": connector_id,
                "docs": post.len(),
                "cursor": cursor,
            }),
        );

        if cursor.is_none() {
            break;
        }
    }

    estate.update_sync(
        connector_id,
        SyncState {
            cursor: cursor.clone(),
            docs_synced,
            last_sync: Some(connxism::now_ms()),
            status: SyncStatus::Idle,
        },
    )?;
    estate.record_trend(
        &format!("connector.{connector_id}.docs_synced"),
        docs_synced as f64,
    )?;
    rrf_core::events::emit(
        "connector.synced",
        serde_json::json!({
            "connector": connector_id,
            "ingested": ingested,
            "batches": batches,
        }),
    );

    Ok(SyncReport {
        ingested,
        batches,
        cursor,
    })
}
