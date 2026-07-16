//! Sprint 9 gate: a quantized estate (SQ8 graph + exact rescore from the
//! durable vectors) keeps recall against the full-precision ground truth.

use connxism::{Estate, EstateConfig};
use rrf_core::{Embedding, Recall, VectorRecord};

fn pseudo_vec(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn quantized_estate_recall_gate() {
    let n = 2048; // above the ANN routing threshold — the graph answers
    let dim = 64;
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open_with(
        dir.path(),
        "sq8",
        EstateConfig {
            quantized: true,
            ..EstateConfig::default()
        },
    )
    .unwrap();
    let recall = estate.recall();

    let mut vecs = Vec::with_capacity(n);
    let mut records = Vec::with_capacity(n);
    for i in 0..n {
        let e = Embedding(pseudo_vec(i as u64, dim));
        vecs.push(e.clone());
        records.push(VectorRecord::new(
            format!("doc{i}"),
            e,
            format!("entry {i}"),
        ));
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();

    let queries = 50;
    let mut found = 0usize;
    let mut total = 0usize;
    for qi in 0..queries {
        let q = Embedding(pseudo_vec(1_000_000 + qi as u64, dim));
        // Exact ground truth over the full-precision vectors.
        let mut truth: Vec<(usize, f32)> = vecs.iter().map(|v| q.cosine(v)).enumerate().collect();
        truth.sort_by(|a, b| b.1.total_cmp(&a.1));
        let truth: Vec<String> = truth
            .into_iter()
            .take(10)
            .map(|(i, _)| format!("doc{i}"))
            .collect();

        let hits = recall.search(&q, 10).await.unwrap();
        for t in &truth {
            total += 1;
            if hits.iter().any(|c| c.id.as_str() == t) {
                found += 1;
            }
        }
        // Rescored scores must be exact cosine, not quantized approximations.
        for c in &hits {
            let i: usize = c.id.as_str()[3..].parse().unwrap();
            let exact = q.cosine(&vecs[i]);
            assert!(
                (c.score - exact).abs() < 1e-5,
                "score for {} must be exact after rescore: {} vs {exact}",
                c.id,
                c.score
            );
        }
    }
    let recall_at_10 = found as f64 / total as f64;
    println!("SQ8 ESTATE GATE — recall@10 {recall_at_10:.3} (quantized graph + exact rescore)");
    assert!(
        recall_at_10 >= 0.90,
        "quantized estate recall@10 = {recall_at_10:.3}, gate is 0.90"
    );
}
