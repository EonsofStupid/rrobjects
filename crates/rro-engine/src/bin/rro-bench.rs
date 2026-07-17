//! `rro-bench` — the measurement harness. Real runs, real numbers.
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
//! cargo run --release --bin rro-bench -- --docs 50000 --queries 500 --store estate
//! ```
//!
//! **Baseline tracking:** `--write-baseline <path>` records this run's
//! configuration + numbers; `--baseline <path>` compares a later run against
//! it and exits non-zero on regression beyond `--tolerance <percent>`
//! (default 25 — shared containers are noisy; tighten on dedicated hardware).
//! `--events <path>` streams the run as JSONL for DuckDB.

use std::sync::Arc;
use std::time::Instant;

use model_registry::build_embedder;
use recall::FlatRecall;
use rro_core::{Document, Embedder, Recall};
use rro_engine::{spawn_ingest, IngestConfig, ObjectConfig, ReasonReadyObject};
use serde::{Deserialize, Serialize};

/// The recorded shape of a run: configuration + headline numbers.
#[derive(Debug, Serialize, Deserialize)]
struct Baseline {
    config: BaselineConfig,
    recorded_at_ms: u64,
    ingest_docs_per_sec: f64,
    query_p50_ms: f64,
    query_p95_ms: f64,
    /// Fraction of planted golden docs retrieved in top-k (protocol ≥ planted-v1).
    #[serde(default)]
    accuracy_at_k: f64,
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
    /// Corpus/metric protocol version; numbers are never compared across
    /// protocols.
    #[serde(default = "protocol_v0")]
    protocol: String,
}

fn protocol_v0() -> String {
    "v0".to_string()
}

/// Current corpus/metric protocol: each query carries a unique anchor term
/// (df = 1 across the corpus) planted in exactly one golden doc; accuracy@k
/// = fraction of queries whose golden doc is retrieved. A correct engine
/// scores 1.0; approximate or broken retrieval falls below.
const PROTOCOL: &str = "planted-v1";

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

/// One retrieval target: the query text and the id of its planted golden doc.
struct PlantedQuery {
    query: String,
    gold_id: String,
}

/// Build the planted-v1 corpus: `docs` noise documents plus one golden doc
/// per query. Each query gets a unique anchor term (`anchorN`, df = 1 in the
/// whole corpus — noise vocabulary can never produce it) plus two common
/// terms; its golden doc contains the anchor, the common terms, and noise.
fn build_corpus(docs: usize, queries: usize, seed: &mut u64) -> (Vec<Document>, Vec<PlantedQuery>) {
    let mut corpus: Vec<Document> = (0..docs).map(|i| synth_doc(i, seed)).collect();
    let mut planted = Vec::with_capacity(queries);

    for q in 0..queries {
        let anchor = format!("anchorq{q}");
        let common_a = synth_term(seed);
        let common_b = synth_term(seed);

        let mut words = vec![anchor.clone(), common_a.clone(), common_b.clone()];
        for _ in 0..20 {
            words.push(synth_term(seed));
        }
        let gold_id = format!("gold-{q}");
        // Insert goldens spread through the corpus, not appended at the end.
        let pos = (xorshift(seed) as usize) % (corpus.len() + 1);
        corpus.insert(pos, Document::new(words.join(" ")).with_id(gold_id.clone()));

        planted.push(PlantedQuery {
            query: format!("{anchor} {common_a} {common_b}"),
            gold_id,
        });
    }
    (corpus, planted)
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
        rro_core::events::set_sink(Box::new(rro_core::events::JsonlSink::open(path)?));
    }
    rro_core::events::emit(
        "bench.start",
        serde_json::json!({ "store": store_kind, "docs": docs, "queries": queries }),
    );

    let remote = opt_arg_str("--remote");
    let full_flow = std::env::args().any(|a| a == "--full-flow");

    // Honour RRO_EMBEDDER, defaulting to the weightless hash embedder.
    //
    // The default is deliberate and stays: this harness measures the *engine* —
    // ingest throughput, query p50/p95/p99 — and a real model would drown those
    // numbers in its own forward pass (measured elsewhere: engine ~3.9 ms, model
    // ~1124 ms, so 99.65% of a real wall clock is the model). Benchmarking the
    // engine through a 4B model measures the 4B model.
    //
    // But it used to be *hardcoded*, and `--export` shares this embedder. So the
    // command `real_vector_ef.rs` documents for producing real vectors —
    // `RRO_EMBEDDER=llamacpp cargo run --bin rro-bench -- --export ...` —
    // silently ignored the variable and wrote 384-d **hash** vectors. The test
    // that exists precisely because "the ANN gate has only ever run on synthetic
    // vectors" was being handed synthetic vectors by its own instructions.
    let ecfg = model_registry::EmbedderConfig::from_env()?;
    let embedder: Arc<dyn Embedder> = build_embedder(&ecfg).await?;
    if ecfg.kind != model_registry::EmbedderKind::Deterministic {
        eprintln!(
            "embedder: {} ({}) dim={} — NOTE: latency below now includes model time",
            ecfg.kind.as_str(),
            embedder.model_name(),
            embedder.dim()
        );
    }

    // planted-v1 corpus: noise docs + one golden doc per query.
    let mut seed = 0x5EED_u64;
    let (corpus, planted) = build_corpus(docs, queries, &mut seed);
    let total_docs = corpus.len();

    // --export <path>: dump the corpus + queries with precomputed embeddings
    // as JSONL so external baselines run on IDENTICAL inputs, then exit.
    if let Some(path) = opt_arg_str("--export") {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for d in &corpus {
            let v = embedder.embed_document_one(&d.text).await?;
            let line = serde_json::json!({
                "kind": "doc", "id": d.id.as_str(), "text": d.text, "vector": v.0,
            });
            writeln!(f, "{line}")?;
        }
        for pq in &planted {
            let v = embedder.embed_query_one(&pq.query).await?;
            let line = serde_json::json!({
                "kind": "query", "query": pq.query, "gold_id": pq.gold_id, "vector": v.0,
            });
            writeln!(f, "{line}")?;
        }
        f.flush()?;
        println!(
            "exported {} docs + {} queries → {path}",
            total_docs,
            planted.len()
        );
        return Ok(());
    }

    let mode = remote.clone().unwrap_or_else(|| "local".into());
    println!(
        "# rro-bench — protocol={PROTOCOL} store={store_kind} mode={mode} docs={total_docs} batch={batch} concurrency={concurrency}\n"
    );

    let (ingest_dps, ingest_secs, lat_us, hits): (f64, f64, Vec<u128>, usize) = match &remote {
        // ---- remote: full pipeline over the a2a layer-2 protocol ----
        Some(addr) => {
            let t0 = Instant::now();
            for chunk in corpus.chunks(batch) {
                let msg = rro_net::Message::request(
                    "bench",
                    "rro",
                    "index",
                    serde_json::json!({ "docs": chunk }),
                );
                let reply = rro_net::tcp::request(addr.as_str(), &msg).await?;
                anyhow::ensure!(
                    reply.body.get("total").is_some(),
                    "index reply missing total"
                );
            }
            let secs = t0.elapsed().as_secs_f64();

            let mut lat = Vec::with_capacity(queries);
            let mut hit = 0usize;
            for pq in &planted {
                let msg = rro_net::Message::request(
                    "bench",
                    "rro",
                    "ask",
                    serde_json::json!({ "query": pq.query }),
                );
                let t = Instant::now();
                let reply = rro_net::tcp::request(addr.as_str(), &msg).await?;
                lat.push(t.elapsed().as_micros());
                let found = reply.body["candidates"]
                    .as_array()
                    .map(|c| {
                        c.iter()
                            .any(|cand| cand["id"].as_str() == Some(pq.gold_id.as_str()))
                    })
                    .unwrap_or(false);
                hit += usize::from(found);
            }
            (total_docs as f64 / secs.max(1e-9), secs, lat, hit)
        }

        // ---- local: ingestion machine + store-level hybrid search ----
        None => {
            // Keep the estate dir AND the estate itself alive for the whole
            // run — Estate owns the out-of-band applier thread; dropping it
            // stops graph maintenance.
            let _tmp;
            let _estate_keeper;
            let store: Arc<dyn Recall> = match store_kind.as_str() {
                "estate" => {
                    let tmp = tempfile::tempdir()?;
                    let estate = connxism::Estate::open(tmp.path(), "bench")?;
                    let r: Arc<dyn Recall> = Arc::new(estate.recall());
                    _tmp = Some(tmp);
                    _estate_keeper = Some(estate);
                    r
                }
                _ => {
                    _tmp = None;
                    _estate_keeper = None;
                    Arc::new(FlatRecall::new())
                }
            };

            let handle = spawn_ingest(
                embedder.clone(),
                store.clone(),
                IngestConfig {
                    batch_size: batch,
                    concurrency,
                    ..IngestConfig::default()
                },
            );
            let t0 = Instant::now();
            for doc in corpus {
                handle.submit(doc).await?;
            }
            let stats = handle.finish().await?;
            let secs = t0.elapsed().as_secs_f64();

            assert_eq!(stats.indexed as usize, total_docs, "all docs must index");
            assert_eq!(store.len().await? as usize, total_docs);

            // Out-of-band index maintenance: wait for catch-up and report it
            // honestly (durable ingest and searchable-at-full-speed are two
            // different moments; both get printed).
            let tq = Instant::now();
            store.quiesce().await?;
            let catchup = tq.elapsed().as_secs_f64();
            println!("| index catch-up (quiesce) | {catchup:.2} s |");
            println!("| time to fully indexed | {:.2} s |", secs + catchup);

            let mut lat = Vec::with_capacity(queries);
            let mut hit = 0usize;
            if full_flow {
                // The whole engine as one: embed → hybrid recall → rerank →
                // classify per query, every stage evented (flow.stage).
                let flow = ReasonReadyObject::builder()
                    .embedder(embedder.clone())
                    .recall(store.clone())
                    .config(ObjectConfig {
                        recall_k: top_k.max(20),
                        rerank_k: top_k,
                    })
                    .build();
                for pq in &planted {
                    let t = Instant::now();
                    let result = flow.ask(&pq.query).await?;
                    lat.push(t.elapsed().as_micros());
                    let found = result
                        .candidates
                        .iter()
                        .any(|c| c.id.as_str() == pq.gold_id);
                    hit += usize::from(found);
                }
            } else {
                for pq in &planted {
                    let emb = embedder.embed_query_one(&pq.query).await?;
                    let t = Instant::now();
                    let results = store.hybrid_search(&pq.query, &emb, top_k).await?;
                    lat.push(t.elapsed().as_micros());
                    assert!(
                        !results.is_empty(),
                        "queries over a populated store must hit"
                    );
                    let found = results.iter().any(|c| c.id.as_str() == pq.gold_id);
                    hit += usize::from(found);
                }
            }
            (stats.docs_per_sec, secs, lat, hit)
        }
    };

    let accuracy = hits as f64 / queries.max(1) as f64;

    println!("## ingest");
    println!("| metric | value |");
    println!("|---|---|");
    println!("| documents | {total_docs} |");
    println!("| wall time | {ingest_secs:.2} s |");
    println!("| throughput | {ingest_dps:.0} docs/sec |");

    let mut lat_us = lat_us;
    lat_us.sort_unstable();
    let p50 = percentile(&lat_us, 0.50);
    let p95 = percentile(&lat_us, 0.95);
    println!("\n## query (hybrid, top-{top_k}, {queries} planted queries)");
    println!("| metric | value |");
    println!("|---|---|");
    println!("| **accuracy@{top_k} (golden retrieved)** | **{accuracy:.3}** |");
    println!("| p50 | {p50:.2} ms |");
    println!("| p95 | {p95:.2} ms |");
    println!("| p99 | {:.2} ms |", percentile(&lat_us, 0.99));
    println!(
        "| throughput | {:.0} qps (sequential) |",
        1000.0 / p50.max(1e-9)
    );

    rro_core::events::emit(
        "bench.result",
        serde_json::json!({
            "store": store_kind,
            "mode": mode,
            "protocol": PROTOCOL,
            "ingest_docs_per_sec": ingest_dps,
            "query_p50_ms": p50,
            "query_p95_ms": p95,
            "accuracy_at_k": accuracy,
        }),
    );

    // ---- baseline configuration & tracking ----
    let config = BaselineConfig {
        store: if remote.is_some() {
            format!("{store_kind}+a2a")
        } else {
            store_kind.clone()
        },
        docs,
        queries,
        batch,
        concurrency,
        top_k,
        protocol: PROTOCOL.to_string(),
    };
    let current = Baseline {
        config,
        recorded_at_ms: rro_core::events::now_ms(),
        ingest_docs_per_sec: ingest_dps,
        query_p50_ms: p50,
        query_p95_ms: p95,
        accuracy_at_k: accuracy,
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
        rro_core::events::emit(
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
    // Accuracy: judged with the same machinery so the report stays uniform.
    // Baselines recorded before planted-v1 carry 0.0 and always pass.
    check(
        "accuracy@k",
        stored.accuracy_at_k,
        current.accuracy_at_k,
        true,
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
                protocol: PROTOCOL.to_string(),
            },
            recorded_at_ms: 1,
            ingest_docs_per_sec: 100_000.0,
            query_p50_ms: 80.0,
            query_p95_ms: 90.0,
            accuracy_at_k: 1.0,
        }
    }

    #[test]
    fn planted_corpus_anchors_are_unique() {
        let mut seed = 7_u64;
        let (corpus, planted) = build_corpus(200, 20, &mut seed);
        assert_eq!(corpus.len(), 220);
        for pq in &planted {
            let anchor = pq.query.split(' ').next().unwrap();
            let holders: Vec<&str> = corpus
                .iter()
                .filter(|d| d.text.split(' ').any(|w| w == anchor))
                .map(|d| d.id.as_str())
                .collect();
            assert_eq!(holders, vec![pq.gold_id.as_str()], "anchor df must be 1");
        }
    }

    #[test]
    fn accuracy_regression_is_caught() {
        let mut bad = base();
        bad.accuracy_at_k = 0.5;
        assert!(compare(&base(), &bad, 0.25)
            .failed
            .contains(&"accuracy@k".to_string()));
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
