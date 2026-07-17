//! Can the event stream be replayed into ONE turn?
//!
//! The engine emitted `flow.stage` for every stage, but nothing carried a
//! correlation id — so two concurrent queries interleaved and only aggregates
//! were readable. A benchmark number produced by a system you cannot open is a
//! number you cannot argue with: when an arm scores badly, "which stage did it"
//! had no answer.
//!
//! These tests hold the line that the full turn is legible.

use std::sync::{Arc, Mutex, OnceLock};

use rro_core::events::{Event, EventSink};
use rro_core::semconv::attr;
use rro_core::{Recall, VectorRecord};
use rro_engine::ReasonReadyObject;

/// Captures the stream so a test can replay it.
#[derive(Default)]
struct Capture {
    events: Mutex<Vec<Event>>,
}

impl EventSink for Capture {
    fn record(&self, event: Event) {
        self.events.lock().unwrap().push(event);
    }
}

struct Handle(Arc<Capture>);
impl EventSink for Handle {
    fn record(&self, event: Event) {
        self.0.record(event)
    }
}

/// `set_sink` is a process-global OnceLock, so every test in this binary shares
/// one capture. Turn ids are what separate them — which is precisely the
/// property under test.
static CAP: OnceLock<Arc<Capture>> = OnceLock::new();

fn capture() -> Arc<Capture> {
    CAP.get_or_init(|| {
        let c = Arc::new(Capture::default());
        rro_core::events::set_sink(Box::new(Handle(c.clone())));
        c
    })
    .clone()
}

/// Every event carrying `turn == id`, in emission order.
fn turn_of(cap: &Capture, id: u64) -> Vec<Event> {
    cap.events
        .lock()
        .unwrap()
        .iter()
        .filter(|e| e.fields.get(attr::TURN).and_then(|t| t.as_u64()) == Some(id))
        .cloned()
        .collect()
}

fn stages(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter(|e| e.kind == rro_core::semconv::EVENT_STAGE)
        .filter_map(|e| {
            e.fields
                .get(attr::STAGE)
                .and_then(|s| s.as_str())
                .map(str::to_string)
        })
        .collect()
}

/// Every turn id the stream has closed, oldest first.
///
/// There is deliberately no `last_turn_id()` here any more. It read "the newest
/// `flow.turn` in the stream", which is a *guess* — and under concurrency it
/// guesses another test's turn, which is exactly what made this file flaky
/// (7 failures per 100 runs). `ask()` now returns the turn it ran, so a test
/// never has to guess.
fn closed_turns(cap: &Capture) -> Vec<u64> {
    cap.events
        .lock()
        .unwrap()
        .iter()
        .filter(|e| e.kind == "flow.turn")
        .filter_map(|e| e.fields.get(attr::TURN).and_then(|t| t.as_u64()))
        .collect()
}

async fn flow_with_docs() -> (tempfile::TempDir, Arc<ReasonReadyObject>) {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "turn").unwrap());
    let recs: Vec<VectorRecord> = (0..5)
        .map(|i| {
            VectorRecord::new(
                format!("d{i}"),
                rro_core::Embedding(vec![i as f32, 1.0, 0.5]).normalized(),
                format!("document number {i} about retrieval and embeddings"),
            )
        })
        .collect();
    estate.recall().upsert(recs).await.unwrap();
    let flow = Arc::new(
        ReasonReadyObject::builder()
            .rrd(Arc::new(rrd::Rrd::new()))
            .recall(Arc::new(estate.recall()))
            .build(),
    );
    (dir, flow)
}

/// THE gate: one query, and the whole journey is reconstructable — in order,
/// under one id.
#[tokio::test(flavor = "multi_thread")]
async fn one_query_emits_one_readable_turn() {
    let cap = capture();
    let (_d, flow) = flow_with_docs().await;

    let r = flow.ask("what is retrieval").await.unwrap();
    let events = turn_of(&cap, r.turn.get());

    // Opens and closes. A turn that just stops reads as a crash.
    assert_eq!(
        events.first().map(|e| e.kind.as_str()),
        Some("flow.open"),
        "a turn must open with the query it was asked"
    );
    assert_eq!(
        events.last().map(|e| e.kind.as_str()),
        Some("flow.turn"),
        "a turn must close"
    );

    // The pipeline, in pipeline order.
    assert_eq!(
        stages(&events),
        vec!["shape", "embed", "intent", "recall", "rerank", "reason"],
        "every stage must appear exactly once, in order — a missing stage means \
         the turn is unreadable at that point"
    );

    // Every event is timed and attributable.
    for e in events
        .iter()
        .filter(|e| e.kind == rro_core::semconv::EVENT_STAGE)
    {
        assert!(
            e.fields.contains_key(attr::LATENCY_MS),
            "every stage carries its own latency"
        );
        assert!(
            e.fields.contains_key(attr::TURN),
            "every stage carries its turn"
        );
    }

    let close = events.last().unwrap();
    assert!(close.fields.contains_key("total_ms"));
    assert_eq!(
        close.fields.get("gated").unwrap(),
        &serde_json::json!(false)
    );
}

/// The point of the id. Concurrent queries interleave in the stream; each must
/// still be separable into a complete, correct turn.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_turns_do_not_bleed_into_each_other() {
    let cap = capture();
    let (_d, flow) = flow_with_docs().await;

    let qs = ["alpha retrieval", "beta embeddings", "gamma documents"];
    let mut handles = Vec::new();
    for q in qs {
        let f = flow.clone();
        handles.push(tokio::spawn(async move { f.ask(q).await.unwrap() }));
    }
    // The ids of OUR three passes, straight from the results — no guessing.
    let mut ids = Vec::new();
    for h in handles {
        ids.push(h.await.unwrap().turn.get());
    }
    assert_eq!(ids.len(), 3);
    let closed = closed_turns(&cap);
    for id in &ids {
        assert!(closed.contains(id), "turn {id} never closed");
    }

    // Each must be individually complete — only possible if the ids actually
    // separate the interleaved streams.
    for id in &ids {
        let events = turn_of(&cap, *id);
        assert_eq!(
            stages(&events),
            vec!["shape", "embed", "intent", "recall", "rerank", "reason"],
            "turn {id} is incomplete — concurrent turns bled together"
        );
        let queries: Vec<&str> = events
            .iter()
            .filter(|e| e.kind == "flow.open")
            .filter_map(|e| e.fields.get("query").and_then(|q| q.as_str()))
            .collect();
        assert_eq!(queries.len(), 1, "turn {id} opened with exactly one query");
    }
}

/// A refusal is the most interesting turn in the stream: it is the engine
/// declining to spend a model call. It must be as legible as a success.
#[tokio::test(flavor = "multi_thread")]
async fn a_gated_turn_closes_and_shows_zero_model_calls() {
    let cap = capture();
    let (_d, flow) = flow_with_docs().await;

    // Past any sane L0 byte cap -> RRD blocks before the embedder.
    let huge = "x".repeat(8 * 1024 * 1024);
    let r = flow.ask(&huge).await.unwrap();
    assert!(!r.readiness.ready);
    assert_eq!(r.readiness.label, "gated");

    // r.turn is THIS pass's id — not "whatever closed last", which is how this
    // test used to read another test's turn and fail 7% of the time.
    let events = turn_of(&cap, r.turn.get());

    assert_eq!(
        stages(&events),
        vec!["shape"],
        "a blocked query must NOT reach embed/recall/rerank — if any later stage \
         appears, the gate is not saving the model call it claims to"
    );
    let close = events.last().unwrap();
    assert_eq!(close.kind, "flow.turn", "a refusal still closes its turn");
    assert_eq!(close.fields.get("gated").unwrap(), &serde_json::json!(true));
    assert_eq!(
        close.fields.get("model_calls").unwrap(),
        &serde_json::json!(0),
        "the gate's whole claim is that a refusal costs zero model calls"
    );
}

/// Sub-millisecond passes are real (a gate, a warm local store). The close used
/// `as_millis()`, which rounded them to 0 — a stage that reports 0 is a stage
/// nobody profiles.
#[tokio::test(flavor = "multi_thread")]
async fn a_fast_turn_still_reports_a_nonzero_total() {
    let cap = capture();
    let (_d, flow) = flow_with_docs().await;
    let r = flow.ask("fast").await.unwrap();
    let events = turn_of(&cap, r.turn.get());
    let total = events
        .last()
        .unwrap()
        .fields
        .get("total_ms")
        .and_then(|t| t.as_f64())
        .expect("total_ms");
    assert!(
        total > 0.0,
        "total_ms rounded to {total} — sub-ms turns must still be measurable"
    );
}
