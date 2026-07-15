//! The engine's event stream: every meaningful transition, consistently
//! emitted, analytics-ready.
//!
//! Events are structured records (`at_ms`, `kind`, free fields) written as
//! **JSONL** — one JSON object per line — which DuckDB ingests directly:
//!
//! ```sql
//! SELECT kind, count(*), avg(CAST(fields.docs_per_sec AS DOUBLE))
//! FROM read_json_auto('rrf-events.jsonl')
//! GROUP BY kind;
//! ```
//!
//! A process installs one global sink ([`set_sink`]); every component emits
//! through [`emit`]. No sink installed means events are dropped for free —
//! emission is always safe to call.

use std::io::Write;
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

use crate::types::Metadata;

/// One structured event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Epoch milliseconds.
    pub at_ms: u64,
    /// Dotted event kind, e.g. `signal.received`, `ingest.batch`.
    pub kind: String,
    /// Free-form structured payload.
    #[serde(default)]
    pub fields: Metadata,
}

/// Where events go.
pub trait EventSink: Send + Sync {
    /// Record one event.
    fn record(&self, event: Event);
}

static SINK: OnceLock<Box<dyn EventSink>> = OnceLock::new();

/// Install the process-wide sink. First caller wins; later calls are ignored
/// (returns whether this call installed it).
pub fn set_sink(sink: Box<dyn EventSink>) -> bool {
    SINK.set(sink).is_ok()
}

/// Emit an event through the installed sink (no-op without one).
///
/// `fields` is any `serde_json::Value::Object`-able map; non-object values are
/// wrapped under `"value"`.
pub fn emit(kind: &str, fields: serde_json::Value) {
    let Some(sink) = SINK.get() else { return };
    let fields = match fields {
        serde_json::Value::Object(m) => m.into_iter().collect(),
        serde_json::Value::Null => Metadata::new(),
        other => {
            let mut m = Metadata::new();
            m.insert("value".to_string(), other);
            m
        }
    };
    sink.record(Event {
        at_ms: now_ms(),
        kind: kind.to_string(),
        fields,
    });
}

/// Epoch milliseconds now.
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A sink that appends JSONL to a file — the DuckDB-ready default.
pub struct JsonlSink {
    file: Mutex<std::fs::File>,
}

impl JsonlSink {
    /// Open (append-create) the JSONL file at `path`.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(JsonlSink {
            file: Mutex::new(file),
        })
    }
}

impl EventSink for JsonlSink {
    fn record(&self, event: Event) {
        if let (Ok(mut f), Ok(mut line)) = (self.file.lock(), serde_json::to_string(&event)) {
            line.push('\n');
            let _ = f.write_all(line.as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Counter(Arc<AtomicUsize>);
    impl EventSink for Counter {
        fn record(&self, event: Event) {
            assert!(event.at_ms > 0);
            assert!(!event.kind.is_empty());
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn emit_without_sink_is_safe_and_sink_receives() {
        emit("noop.before_install", serde_json::json!({}));

        let count = Arc::new(AtomicUsize::new(0));
        // May race with other tests in this process; only assert when we won.
        if set_sink(Box::new(Counter(count.clone()))) {
            emit("test.event", serde_json::json!({"n": 1}));
            emit("test.scalar", serde_json::json!(42));
            assert_eq!(count.load(Ordering::SeqCst), 2);
        }
    }
}
