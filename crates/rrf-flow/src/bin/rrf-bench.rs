//! `rrf-bench` — the measurement harness. Real runs, real numbers.
//!
//! Measures, end to end through the ingestion machine (embed → index →
//! persist) and the query path (hybrid search):
//!
//! - ingest throughput (docs/sec) and wall time,
//! - query latency p50 / p95 / p99 over a query mix.
//!
//! Stores: `mem` (in-memory) and `estate` (persistent kvs). External baselines
//! are run *outside* this tree with the same corpus/queries and compared on
//! the emitted numbers — this repo carries only its own engine.
//!
//! ```sh
//! cargo run --release --bin rrf-bench -- --docs 50000 --queries 500 --store estate
//! ```
//!
//! **Baseline tracking:** `--write-baseline <path>` records this run's
//! configuration + numbers; `--baseline <path>` compares a later run against
//! it and exits non-zero on regression beyond `--tolerance <percent>`
//! (default 25 — shared containers are noisy; tighten on dedicated hardware).
//! `--events <path>` streams the run as JSONL for DuckDB.

use std::sync::Arc;
use std::time::Instant;

use embedder::DeterministicEmbedder;
use recall::FlatRecall;
use rrf_core::{Document, Embedder, Recall};
use rrf_flow::{spawn_ingest, IngestConfig};
use serde::{Deserialize, Serialize};

/// The recorded shape of a run: configuration + headline numbers.
#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    config: BaselineConfig,
    recorded_at_ms: u64,
    ingest_docs_per_sec: f64,
    query_p50_ms: f64,
    query_p95_ms: f64,
}

/// The configuration a baseline is only comparable under.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct BaselineConfig {
    store: String,
    docs: usize,
    queries: usize,
    batch: usize,
    concurrency: usize,
    top_k: usize,
}

/// Base roots composed into a realistic synthetic vocabulary (~8k distinct
/// terms — real corpora are zipfian over 10⁴–10⁶ terms, not a handful).
const ROOTS: &[&str] = &[
    "estate",
    "vector",
    "recall",
    "storage",
    "upgrade",
    "migration",
    "tokio",
    "signal",
    "daemon",
    "reranker",
    "network",
    "graph",
    "memory",
    "agent",
    "mesh",
    "warp",
    "connector",
    "drive",
    "mailbox",
    "index",
    "shard",
    "batch",
    "trend",
    "shape",
    "tag",
    "readiness",
    "reason",
    "flow",
    "hybrid",
    "lexical",
    "cosine",
    "fusion",
    "ingest",
    "backpressure",
];
const VOCAB_SIZE: u64 = 8192;

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

/// Draw one term: a root suffixed into one of `VOCAB_SIZE` distinct words,
/// with a zipf-ish skew (low suffixes are much more common).
fn synth_term(seed: &mut u64) -> String {
    let root = ROOTS[(xorshift(seed) % ROOTS.len() as u64) as usize];
    // Square a uniform draw to skew mass toward small suffixes.
    let u = (xorshift(seed) % 1000) as f64 / 1000.0;
    let suffix = ((u * u) * (VOCAB_SIZE as f64 / ROOTS.len() as f64)) as u64;
    format!("{root}{suffix}")
}

fn synth_doc(i: usize, seed: &mut u64) -> Document {
    let len = 24 + (xorshift(seed) % 40) as usize;
    let words: Vec<String> = (0..len).map(|_| synth_term(seed)).collect();
    Document::new(words.join(" ")).with_id(format!("doc-{i}"))
}

fn synth_query(seed: &mut u64) -> String {
    let len = 2 + (xorshift(seed) % 3) as usize;
    let words: Vec<String> = (0..len).map(|_| synth_term(seed)).collect();
    words.join(" ")
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_str(name: &str, default: &str) -> String {
    opt_arg_str(name).unwrap_or_else(|| default.to_string())
}

fn opt_arg_str(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn percentile(sorted_us: &[u128], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx] as f64 / 1000.0
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let docs = arg("--docs", 10_000);
    let queries = arg("--queries", 200);
    let batch = arg("--batch", 256);
    let concurrency = arg("--concurrency", 4);
    let top_k = arg("--top-k", 10);
    let store_kind = arg_str("--store", "mem");
    let tolerance_pct = arg("--tolerance", 25) as f64;
    let baseline_path = opt_arg_str("--baseline");
    let write_baseline_path = opt_arg_str("--write-baseline");
    let events_path = opt_arg_str("--events");

    if let Some(path) = &events_path {
        rrf_core::events::set_sink(Box::new(rrf_core::events::JsonlSink::open(path)?));
    }
    rrf_core::events::emit(
        "bench.start",
        serde_json::json!({ "store": store_kind, "docs": docs, "queries": queries }),
    );

    let embedder: Arc<dyn Embedder> = Arc::new(DeterministicEmbedder::new());

    // Keep the estate dir alive for the whole run.
    let tmp;
    let store: Arc<dyn Recall> = match store_kind.as_str() {
        "estate" => {
            tmp = tempfile::tempdir()?;
            let estate = connxism::Estate::open(tmp.path(), "bench")?;
            Arc::new(estate.recall())
        }
        _ => Arc::new(FlatRecall::new()),
    };

    println!(
        "# rrf-bench — store={store_kind} docs={docs} batch={batch} concurrency={concurrency}\n"
    );

    // ---- ingest: full machine (embed → index → persist) ----
    let handle = spawn_ingest(
        embedder.clone(),
        store.clone(),
        IngestConfig {
            batch_size: batch,
            concurrency,
            ..IngestConfig::default()
        },
    );
    let mut seed = 0x5EED_u64;
    let t0 = Instant::now();
    for i in 0..docs {
        handle.submit(synth_doc(i, &mut seed)).await?;
    }
    let stats = handle.finish().await?;
    let ingest_secs = t0.elapsed().as_secs_f64();

    assert_eq!(stats.indexed as usize, docs, "all docs must index");
    assert_eq!(store.len().await? as usize, docs);

    println!("## ingest");
    println!("| metric | value |");
    println!("|---|---|");
    println!("| documents | {docs} |");
    println!("| wall time | {ingest_secs:.2} s |");
    println!("| throughput | {:.0} docs/sec |", stats.docs_per_sec);
    println!("| batches | {} |", stats.batches);
    println!("| errors | {} |", stats.errors);

    // ---- query: hybrid latency ----
    let mut lat_us: Vec<u128> = Vec::with_capacity(queries);
    let mut qseed = 0xFACADE_u64;
    for _ in 0..queries {
        let q = synth_query(&mut qseed);
        let emb = embedder.embed_one(&q).await?;
        let t = Instant::now();
        let hits = store.hybrid_search(&q, &emb, top_k).await?;
        lat_us.push(t.elapsed().as_micros());
        assert!(!hits.is_empty(), "queries over a populated store must hit");
    }
    lat_us.sort_unstable();

    let p50 = percentile(&lat_us, 0.50);
    let p95 = percentile(&lat_us, 0.95);
    println!("\n## query (hybrid, top-{top_k}, {queries} queries)");
    println!("| percentile | latency |");
    println!("|---|---|");
    println!("| p50 | {p50:.2} ms |");
    println!("| p95 | {p95:.2} ms |");
    println!("| p99 | {:.2} ms |", percentile(&lat_us, 0.99));
    println!(
        "| throughput | {:.0} qps (sequential) |",
        1000.0 / p50.max(1e-9)
    );

    rrf_core::events::emit(
        "bench.result",
        serde_json::json!({
            "store": store_kind,
            "ingest_docs_per_sec": stats.docs_per_sec,
            "query_p50_ms": p50,
            "query_p95_ms": p95,
        }),
    );

    // ---- baseline configuration & tracking ----
    let config = BaselineConfig {
        store: store_kind.clone(),
        docs,
        queries,
        batch,
        concurrency,
        top_k,
    };
    let current = Baseline {
        config,
        recorded_at_ms: rrf_core::events::now_ms(),
        ingest_docs_per_sec: stats.docs_per_sec,
        query_p50_ms: p50,
        query_p95_ms: p95,
    };

    if let Some(path) = write_baseline_path {
        std::fs::write(&path, serde_json::to_string_pretty(&current)?)?;
        println!("\nbaseline recorded → {path}");
    }

    if let Some(path) = baseline_path {
        let stored: Baseline = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        if stored.config != current.config {
            eprintln!(
                "baseline config mismatch: stored {:?} vs run {:?}",
                stored.config, current.config
            );
            std::process::exit(2);
        }
        let regressions = compare(&stored, &current, tolerance_pct / 100.0);
        println!("\n## baseline check (±{tolerance_pct:.0}% vs {path})");
        println!("| metric | baseline | now | verdict |");
        println!("|---|---|---|---|");
        for line in &regressions.report {
            println!("{line}");
        }
        rrf_core::events::emit(
            "bench.baseline",
            serde_json::json!({ "path": path, "ok": regressions.failed.is_empty() }),
        );
        if !regressions.failed.is_empty() {
            eprintln!("REGRESSION: {}", regressions.failed.join(", "));
            std::process::exit(1);
        }
        println!("\nbaseline check passed.");
    }

    Ok(())
}

/// Outcome of a baseline comparison.
struct Comparison {
    report: Vec<String>,
    failed: Vec<String>,
}

/// Higher-is-better for throughput, lower-is-better for latency, each judged
/// against the shared fractional tolerance.
fn compare(stored: &Baseline, current: &Baseline, tol: f64) -> Comparison {
    let mut report = Vec::new();
    let mut failed = Vec::new();

    let mut check = |name: &str, base: f64, now: f64, higher_is_better: bool| {
        let ok = if higher_is_better {
            now >= base * (1.0 - tol)
        } else {
            now <= base * (1.0 + tol)
        };
        report.push(format!(
            "| {name} | {base:.2} | {now:.2} | {} |",
            if ok { "ok" } else { "REGRESSION" }
        ));
        if !ok {
            failed.push(name.to_string());
        }
    };

    check(
        "ingest docs/sec",
        stored.ingest_docs_per_sec,
        current.ingest_docs_per_sec,
        true,
    );
    check(
        "query p50 ms",
        stored.query_p50_ms,
        current.query_p50_ms,
        false,
    );
    check(
        "query p95 ms",
        stored.query_p95_ms,
        current.query_p95_ms,
        false,
    );

    Comparison { report, failed }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Baseline {
        Baseline {
            config: BaselineConfig {
                store: "mem".into(),
                docs: 1000,
                queries: 10,
                batch: 256,
                concurrency: 4,
                top_k: 10,
            },
            recorded_at_ms: 1,
            ingest_docs_per_sec: 100_000.0,
            query_p50_ms: 80.0,
            query_p95_ms: 90.0,
        }
    }

    #[test]
    fn within_tolerance_passes() {
        let mut now = base();
        now.ingest_docs_per_sec = 90_000.0; // −10% with 25% tolerance
        now.query_p50_ms = 95.0; // +19%
        assert!(compare(&base(), &now, 0.25).failed.is_empty());
    }

    #[test]
    fn regression_is_caught_both_directions() {
        let mut slow_ingest = base();
        slow_ingest.ingest_docs_per_sec = 50_000.0; // −50%
        assert_eq!(
            compare(&base(), &slow_ingest, 0.25).failed,
            vec!["ingest docs/sec"]
        );

        let mut slow_query = base();
        slow_query.query_p50_ms = 200.0; // +150%
        assert!(compare(&base(), &slow_query, 0.25)
            .failed
            .contains(&"query p50 ms".to_string()));
    }
}
