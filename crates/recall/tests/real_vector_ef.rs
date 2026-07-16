//! A5 — the ANN recall gate, on REAL vectors.
//!
//! `recall/src/ann.rs` gates `recall@10 >= 0.95` against exact search. That gate
//! has only ever run on **synthetic** vectors: `lcg`-generated uniform noise.
//! Real embeddings are nothing like uniform noise — they are anisotropic, they
//! concentrate, and their neighbourhood structure is what an ANN graph's `ef`
//! actually has to cope with. A gate tuned on noise says nothing about the
//! vectors this engine will really see.
//!
//! `#[ignore]` — needs a real-vector file, which CI has no way to produce:
//!
//! ```sh
//! # 1. export real embeddings (JSONL: {"kind":"doc","id":..,"vector":[..]})
//! RRO_EMBEDDER=llamacpp cargo run --release --bin rro-bench -- \
//!   --docs 5000 --export /tmp/real-vectors.jsonl
//!
//! # 2. sweep ef against the exact oracle
//! RRO_TEST_VECTORS=/tmp/real-vectors.jsonl \
//!   cargo test -p recall --test real_vector_ef -- --ignored --nocapture
//! ```
//!
//! The sweep prints the whole curve, not just a pass/fail, because the useful
//! output is "which `ef` buys the gate on real data, and what does it cost" —
//! a single boolean hides the trade the operator actually has to make.

use recall::{AnnConfig, AnnIndex};
use rro_core::{Embedding, Id};

/// Load `{"kind":"doc","id":..,"vector":[..]}` JSONL.
fn load_vectors(path: &str) -> Vec<(String, Embedding)> {
    let f = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} — export real vectors first"));
    let mut out = Vec::new();
    for line in f.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("kind").and_then(|k| k.as_str()) != Some("doc") {
            continue;
        }
        let Some(arr) = v.get("vector").and_then(|x| x.as_array()) else {
            continue;
        };
        let vec: Vec<f32> = arr.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect();
        let id = v
            .get("id")
            .and_then(|i| i.as_str())
            .unwrap_or_default()
            .to_string();
        if !vec.is_empty() && !id.is_empty() {
            out.push((id, Embedding(vec).normalized()));
        }
    }
    out
}

/// Exact top-k by cosine — the oracle the ANN is measured against.
fn exact_top_k(corpus: &[(String, Embedding)], q: &Embedding, k: usize) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = corpus
        .iter()
        .map(|(id, v)| (q.cosine(v), id.as_str()))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored.into_iter().take(k).map(|(_, id)| id.to_string()).collect()
}

fn vectors() -> Option<Vec<(String, Embedding)>> {
    let p = std::env::var("RRO_TEST_VECTORS").ok().filter(|s| !s.trim().is_empty())?;
    let v = load_vectors(&p);
    assert!(
        v.len() >= 500,
        "RRO_TEST_VECTORS={p} yielded only {} vectors — too few to say anything \
         about ANN recall",
        v.len()
    );
    Some(v)
}

/// THE gate, on real vectors: sweep `ef_search` and report the curve.
#[test]
#[ignore]
fn recall_at_10_on_real_vectors_across_ef() {
    let Some(corpus) = vectors() else {
        eprintln!("SKIP: set RRO_TEST_VECTORS to a real-vector JSONL export");
        return;
    };
    let dim = corpus[0].1.dim();
    println!("corpus: {} real vectors, dim={dim}", corpus.len());

    // Build once at the default ef_construction; sweep ef_search at query time.
    let t = std::time::Instant::now();
    let mut idx = AnnIndex::new(AnnConfig::default());
    for (id, v) in &corpus {
        idx.insert(Id::from(id.clone()), v);
    }
    println!("built in {:.1}s ({} nodes)", t.elapsed().as_secs_f64(), idx.len());

    // Queries: real vectors held out of nothing — a doc is its own best probe,
    // and the interesting question is whether the graph finds its true
    // neighbours, so probing WITH corpus vectors is the honest hard case.
    let probes: Vec<&(String, Embedding)> = corpus.iter().step_by(corpus.len() / 100).take(100).collect();

    println!("\n{:>6}  {:>10}  {:>12}", "ef", "recall@10", "us/query");
    let mut best_passing: Option<usize> = None;
    for ef in [16usize, 32, 64, 100, 128, 200, 256] {
        let mut found = 0usize;
        let mut total = 0usize;
        let t = std::time::Instant::now();
        for (_, q) in probes.iter() {
            let truth = exact_top_k(&corpus, q, 10);
            let got = idx.search(q, 10, ef);
            let got_ids: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
            for t in &truth {
                total += 1;
                if got_ids.contains(&t.as_str()) {
                    found += 1;
                }
            }
        }
        let recall = found as f64 / total as f64;
        let us = t.elapsed().as_micros() as f64 / probes.len() as f64;
        let mark = if recall >= 0.95 { " <- passes the 0.95 gate" } else { "" };
        println!("{ef:>6}  {recall:>10.4}  {us:>12.1}{mark}");
        if recall >= 0.95 && best_passing.is_none() {
            best_passing = Some(ef);
        }
    }

    match best_passing {
        Some(ef) => println!(
            "\nRESULT: recall@10 >= 0.95 on REAL {dim}-d vectors at ef={ef} \
             (default ef_search is {}).",
            AnnConfig::default().ef_search
        ),
        None => panic!(
            "recall@10 never reached 0.95 on real {dim}-d vectors at any ef up to 256. \
             The gate at ann.rs:533 holds only for synthetic vectors — that is a real \
             finding about the graph, not a flaky test. Report it; do not relax the gate."
        ),
    }
}

/// The default config must earn its default. If `ef_search: 64` does not hold the
/// gate on real vectors, the default is a lie and must change.
#[test]
#[ignore]
fn the_default_ef_holds_the_gate_on_real_vectors() {
    let Some(corpus) = vectors() else {
        eprintln!("SKIP: set RRO_TEST_VECTORS");
        return;
    };
    let cfg = AnnConfig::default();
    let mut idx = AnnIndex::new(cfg.clone());
    for (id, v) in &corpus {
        idx.insert(Id::from(id.clone()), v);
    }
    let probes: Vec<&(String, Embedding)> =
        corpus.iter().step_by((corpus.len() / 100).max(1)).take(100).collect();

    let mut found = 0usize;
    let mut total = 0usize;
    for (_, q) in probes.iter() {
        let truth = exact_top_k(&corpus, q, 10);
        let got = idx.search(q, 10, cfg.ef_search);
        let ids: Vec<&str> = got.iter().map(|(id, _)| id.as_str()).collect();
        for t in &truth {
            total += 1;
            if ids.contains(&t.as_str()) {
                found += 1;
            }
        }
    }
    let recall = found as f64 / total as f64;
    println!(
        "default ef_search={} -> recall@10 = {recall:.4} on {} real {}-d vectors",
        cfg.ef_search,
        corpus.len(),
        corpus[0].1.dim()
    );
    assert!(
        recall >= 0.95,
        "the DEFAULT ef_search={} gives recall@10={recall:.4} on real vectors, below \
         the 0.95 gate. The default was tuned on synthetic noise. Raise the default \
         (and pay the latency) or document the real number — do not leave a default \
         that only passes on fake data.",
        cfg.ef_search
    );
}
