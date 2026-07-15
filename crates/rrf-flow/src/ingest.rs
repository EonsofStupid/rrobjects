//! The ingestion machine: tokio-native, backpressured, signal-drain-safe.
//!
//! Documents stream in through a **bounded** channel (backpressure is the
//! contract: a flooding producer awaits, memory stays flat), are gathered into
//! batches (size- or linger-bounded), embedded, and upserted into recall.
//! Batches run concurrently under a semaphore. Every transition is published
//! on a watch channel, so the engine's ingestion **state** is observable in
//! real time — and renderable by the connectome.
//!
//! Shutdown is graceful by construction: close the intake (drop the handle or
//! call [`IngestHandle::finish`]) and the worker drains in-flight batches to
//! completion before reporting [`IngestPhase::Indexed`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use rrf_core::{Document, Embedder, Recall, Result, RrfError, VectorRecord};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, watch, Semaphore};
use tokio::task::JoinSet;

/// Tuning for the ingestion machine.
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// Bounded intake depth — the backpressure point.
    pub queue_depth: usize,
    /// Maximum documents per batch.
    pub batch_size: usize,
    /// Flush a partial batch after this long without new documents.
    pub batch_linger: Duration,
    /// Concurrent batches in flight (embed + upsert).
    pub concurrency: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        IngestConfig {
            queue_depth: 4096,
            batch_size: 256,
            batch_linger: Duration::from_millis(50),
            concurrency: 4,
        }
    }
}

/// Where the machine currently stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IngestPhase {
    /// Accepting documents; nothing in flight.
    Idle,
    /// Documents are flowing.
    Ingesting,
    /// Intake closed; draining in-flight batches.
    Draining,
    /// All accepted documents are indexed.
    Indexed,
}

/// Live counters, published on every batch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestStats {
    /// Documents accepted into the machine.
    pub received: u64,
    /// Documents durably indexed.
    pub indexed: u64,
    /// Batches completed.
    pub batches: u64,
    /// Documents that failed (embed or upsert error).
    pub errors: u64,
    /// Wall-clock of the last completed batch, milliseconds.
    pub last_batch_ms: u64,
    /// Overall observed throughput, documents/second.
    pub docs_per_sec: f64,
}

/// The observable state: phase + counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestStatus {
    /// Current phase.
    pub phase: IngestPhase,
    /// Current counters.
    pub stats: IngestStats,
}

/// Producer-side handle: submit documents, then finish.
pub struct IngestHandle {
    tx: mpsc::Sender<Document>,
    status: watch::Receiver<IngestStatus>,
    worker: tokio::task::JoinHandle<IngestStats>,
}

impl IngestHandle {
    /// Submit one document. Awaits when the intake is full — this is the
    /// backpressure point.
    pub async fn submit(&self, doc: Document) -> Result<()> {
        self.tx
            .send(doc)
            .await
            .map_err(|_| RrfError::msg("ingest machine stopped"))
    }

    /// A watch receiver for live status (cheap to clone).
    pub fn status(&self) -> watch::Receiver<IngestStatus> {
        self.status.clone()
    }

    /// Close the intake and wait for a full drain. Returns final counters.
    pub async fn finish(self) -> Result<IngestStats> {
        drop(self.tx); // close intake; worker drains and exits
        self.worker
            .await
            .map_err(|e| RrfError::msg(format!("ingest worker panicked: {e}")))
    }
}

/// Spawn the ingestion machine over any embedder + recall store.
pub fn spawn_ingest(
    embedder: Arc<dyn Embedder>,
    recall: Arc<dyn Recall>,
    config: IngestConfig,
) -> IngestHandle {
    let (tx, rx) = mpsc::channel::<Document>(config.queue_depth.max(1));
    let (status_tx, status_rx) = watch::channel(IngestStatus {
        phase: IngestPhase::Idle,
        stats: IngestStats::default(),
    });

    let worker = tokio::spawn(run_machine(embedder, recall, config, rx, status_tx));

    IngestHandle {
        tx,
        status: status_rx,
        worker,
    }
}

async fn run_machine(
    embedder: Arc<dyn Embedder>,
    recall: Arc<dyn Recall>,
    config: IngestConfig,
    mut rx: mpsc::Receiver<Document>,
    status: watch::Sender<IngestStatus>,
) -> IngestStats {
    let started = Instant::now();
    let mut stats = IngestStats::default();
    let mut phase = IngestPhase::Idle;
    let semaphore = Arc::new(Semaphore::new(config.concurrency.max(1)));
    let mut inflight: JoinSet<(u64, u64, u64)> = JoinSet::new(); // (indexed, errors, batch_ms)

    let publish =
        |phase: IngestPhase, stats: &IngestStats, status: &watch::Sender<IngestStatus>| {
            let _ = status.send(IngestStatus {
                phase,
                stats: stats.clone(),
            });
        };

    let mut batch: Vec<Document> = Vec::with_capacity(config.batch_size);
    let mut intake_open = true;

    while intake_open || !batch.is_empty() || !inflight.is_empty() {
        // Harvest any finished batches without blocking.
        while let Some(done) = inflight.try_join_next() {
            if let Ok((indexed, errors, ms)) = done {
                harvest(&mut stats, indexed, errors, ms, &started);
                publish(phase, &stats, &status);
            }
        }

        // Fill the current batch: take what's queued, or linger briefly.
        if intake_open {
            let linger = tokio::time::sleep(config.batch_linger);
            tokio::pin!(linger);
            loop {
                if batch.len() >= config.batch_size {
                    break;
                }
                tokio::select! {
                    maybe = rx.recv() => match maybe {
                        Some(doc) => {
                            stats.received += 1;
                            if phase == IngestPhase::Idle {
                                phase = IngestPhase::Ingesting;
                                publish(phase, &stats, &status);
                            }
                            batch.push(doc);
                        }
                        None => { intake_open = false; break; }
                    },
                    _ = &mut linger => break,
                }
            }
        }

        if !intake_open && phase != IngestPhase::Draining && phase != IngestPhase::Indexed {
            phase = IngestPhase::Draining;
            publish(phase, &stats, &status);
        }

        // Dispatch the batch under the concurrency budget.
        if !batch.is_empty() {
            let docs = std::mem::take(&mut batch);
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("semaphore never closed");
            let e = embedder.clone();
            let r = recall.clone();
            inflight.spawn(async move {
                let _permit = permit;
                let t0 = Instant::now();
                let n = docs.len() as u64;
                match index_batch(e.as_ref(), r.as_ref(), docs).await {
                    Ok(()) => (n, 0, t0.elapsed().as_millis() as u64),
                    Err(err) => {
                        tracing::warn!("ingest batch failed: {err}");
                        (0, n, t0.elapsed().as_millis() as u64)
                    }
                }
            });
        } else if !intake_open {
            // Nothing new to dispatch; wait for stragglers.
            if let Some(Ok((indexed, errors, ms))) = inflight.join_next().await {
                harvest(&mut stats, indexed, errors, ms, &started);
                publish(phase, &stats, &status);
            }
        }
    }

    phase = IngestPhase::Indexed;
    let secs = started.elapsed().as_secs_f64().max(1e-9);
    stats.docs_per_sec = stats.indexed as f64 / secs;
    publish(phase, &stats, &status);
    rrf_core::events::emit(
        "ingest.finished",
        serde_json::json!({
            "received": stats.received,
            "indexed": stats.indexed,
            "errors": stats.errors,
            "batches": stats.batches,
            "docs_per_sec": stats.docs_per_sec,
        }),
    );
    stats
}

/// Fold one finished batch into the counters and emit the batch event —
/// the single place batch completion is accounted, so the stream is
/// consistent no matter which harvest site observed it.
fn harvest(stats: &mut IngestStats, indexed: u64, errors: u64, ms: u64, started: &Instant) {
    stats.indexed += indexed;
    stats.errors += errors;
    stats.batches += 1;
    stats.last_batch_ms = ms;
    let secs = started.elapsed().as_secs_f64().max(1e-9);
    stats.docs_per_sec = stats.indexed as f64 / secs;
    rrf_core::events::emit(
        "ingest.batch",
        serde_json::json!({
            "indexed": indexed,
            "errors": errors,
            "batch_ms": ms,
            "total_indexed": stats.indexed,
            "docs_per_sec": stats.docs_per_sec,
        }),
    );
}

async fn index_batch(
    embedder: &dyn Embedder,
    recall: &dyn Recall,
    docs: Vec<Document>,
) -> Result<()> {
    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let embeddings = embedder.embed(&texts).await?;
    let records: Vec<VectorRecord> = docs
        .into_iter()
        .zip(embeddings)
        .map(|(d, e)| {
            let mut r = VectorRecord::new(d.id, e, d.text);
            r.metadata = d.metadata;
            r
        })
        .collect();
    recall.upsert(records).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use embedder::DeterministicEmbedder;
    use recall::FlatRecall;

    #[tokio::test(flavor = "multi_thread")]
    async fn ingests_everything_and_reports_indexed() {
        let handle = spawn_ingest(
            Arc::new(DeterministicEmbedder::new()),
            Arc::new(FlatRecall::new()),
            IngestConfig {
                batch_size: 16,
                ..IngestConfig::default()
            },
        );
        let mut status = handle.status();

        for i in 0..500 {
            handle
                .submit(Document::new(format!(
                    "document number {i} about topic {}",
                    i % 7
                )))
                .await
                .unwrap();
        }
        let stats = handle.finish().await.unwrap();

        assert_eq!(stats.received, 500);
        assert_eq!(stats.indexed, 500);
        assert_eq!(stats.errors, 0);
        assert!(stats.docs_per_sec > 0.0);

        let last = status.borrow_and_update();
        assert_eq!(last.phase, IngestPhase::Indexed);
        assert_eq!(last.stats.indexed, 500);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn backpressure_bounds_the_queue() {
        // Tiny queue + tiny batches: submits must await rather than balloon.
        let handle = spawn_ingest(
            Arc::new(DeterministicEmbedder::new()),
            Arc::new(FlatRecall::new()),
            IngestConfig {
                queue_depth: 8,
                batch_size: 4,
                concurrency: 1,
                ..IngestConfig::default()
            },
        );
        for i in 0..200 {
            handle
                .submit(Document::new(format!("doc {i}")))
                .await
                .unwrap();
        }
        let stats = handle.finish().await.unwrap();
        assert_eq!(stats.indexed, 200);
    }
}
