//! Timing probe for the ANN hot loop (build + search), full-precision store.
use recall::{AnnConfig, AnnIndex};
use rrf_core::{Embedding, Id};

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

fn main() {
    let (n, dim, queries) = (20_000, 384, 500);
    let mut idx = AnnIndex::new(AnnConfig::default());
    let t = std::time::Instant::now();
    for i in 0..n {
        idx.insert(
            Id::new(format!("v{i}")),
            &Embedding(pseudo_vec(i as u64, dim)),
        );
    }
    let build = t.elapsed();
    let t = std::time::Instant::now();
    let mut acc = 0usize;
    for qi in 0..queries {
        let q = Embedding(pseudo_vec(1_000_000 + qi as u64, dim));
        acc += idx.search(&q, 10, 64).len();
    }
    let search = t.elapsed();
    println!(
        "build {:.2}s | search {:.1}µs/query | sanity {acc}",
        build.as_secs_f64(),
        search.as_micros() as f64 / queries as f64
    );
}
