//! Phase 6b — paged vector storage. The graph's vectors live in a node-ordered
//! file read through a bounded page cache (`read_at`, no mmap, no unsafe), so a
//! graph far larger than RAM opens and searches with only its working set
//! resident.
//!
//! No mmap: memory-mapping a file cannot be made sound in safe Rust (external
//! truncation is UB), so — like redb, which removed its mmap backend for exactly
//! this reason — recall pages through an owned buffer cache instead. That keeps
//! the whole engine `#![forbid(unsafe_code)]`.

use recall::{AnnConfig, AnnIndex, Quantizer};
use rro_core::{Embedding, Id};

fn pseudo_vec(seed: u64, dim: usize) -> Embedding {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let v: Vec<f32> = (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect();
    Embedding(v)
}

fn build(n: usize, dim: usize, config: AnnConfig) -> AnnIndex {
    let mut idx = AnnIndex::new(config);
    for i in 0..n {
        idx.insert(Id::new(format!("v{i}")), &pseudo_vec(i as u64, dim));
    }
    idx
}

/// Persist `idx`'s structure + vectors, then reopen the vectors paged through a
/// cache of `cache_bytes`. The returned `TempDir` must be held for the graph's
/// lifetime (it backs the vector file).
fn persist_and_page(
    idx: &AnnIndex,
    config: AnnConfig,
    cache_bytes: usize,
) -> (AnnIndex, tempfile::TempDir) {
    let structure = idx.to_structure_bytes();
    let mut vecbytes = Vec::new();
    idx.write_vectors_to(&mut vecbytes).unwrap();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.vectors");
    std::fs::write(&path, &vecbytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let paged = AnnIndex::from_paged(&structure, file, config, cache_bytes).unwrap();
    (paged, dir)
}

/// The core 6b property: a graph reloaded with its vectors paged from disk
/// returns byte-identical search results to the in-RAM graph — while its resident
/// vector memory stays bounded by the cache budget, not the dataset size.
#[test]
fn paged_search_is_identical_and_ram_is_bounded() {
    // Dataset ≈ 1.5 MiB (3000 × 128 × 4); cache 256 KiB — a ~6× gap, so eviction
    // is real and "bounded" is a genuine claim.
    let dim = 128;
    let n = 3000;
    let idx = build(n, dim, AnnConfig::default());
    let dataset_bytes = n * dim * 4;

    let cache_bytes = 256 * 1024;
    let (paged, _dir) = persist_and_page(&idx, AnnConfig::default(), cache_bytes);
    assert_eq!(paged.len(), idx.len());

    for qi in 0..50 {
        let q = pseudo_vec(3_000_000 + qi as u64, dim);
        let a = idx.search(&q, 10, 128);
        let b = paged.search(&q, 10, 128);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.0.as_str(),
                y.0.as_str(),
                "paged ranking must match in-RAM"
            );
            assert!((x.1 - y.1).abs() < 1e-6, "paged scores must match in-RAM");
        }
    }

    // After all those queries, resident vector memory is still bounded by the
    // cache budget (plus a page's slack), nowhere near the whole dataset.
    let resident = paged.heap_vector_bytes();
    assert!(
        resident <= cache_bytes + 64 * 1024,
        "resident {resident} must stay within the cache budget {cache_bytes}"
    );
    assert!(
        resident < dataset_bytes / 2,
        "resident {resident} must be well under the dataset {dataset_bytes}"
    );
}

/// Quantized graphs page too: codes come from the file, params ride in the
/// structure blob. Reload reproduces the same ranking.
#[test]
fn paged_quantized_round_trips() {
    let config = AnnConfig {
        quantizer: Quantizer::Sq8,
        ..AnnConfig::default()
    };
    let idx = build(1500, 32, config.clone());

    let (paged, _dir) = persist_and_page(&idx, config, 128 * 1024);
    assert!(paged.is_quantized());

    for i in (0..1500u64).step_by(30) {
        let q = pseudo_vec(i, 32);
        let a = idx.search(&q, 5, 64);
        let b = paged.search(&q, 5, 64);
        let a_ids: Vec<_> = a.iter().map(|(id, _)| id.as_str().to_string()).collect();
        let b_ids: Vec<_> = b.iter().map(|(id, _)| id.as_str().to_string()).collect();
        assert_eq!(a_ids, b_ids, "paged quantized ranking must match in-RAM");
    }
}

/// After loading paged, new inserts land in the RAM tail and are immediately
/// searchable, while base nodes keep paging from disk — read-your-writes on top
/// of a paged base.
#[test]
fn inserts_after_paged_load_go_to_tail_and_are_searchable() {
    let idx = build(1000, 32, AnnConfig::default());
    let (mut paged, _dir) = persist_and_page(&idx, AnnConfig::default(), 64 * 1024);
    let before = paged.heap_vector_bytes();

    let newv = pseudo_vec(7_777_777, 32);
    paged.insert(Id::new("newnode"), &newv);
    assert!(
        paged.heap_vector_bytes() > before,
        "the appended vector must add to the RAM tail"
    );

    let hits = paged.search(&newv, 5, 128);
    assert!(
        hits.iter().any(|(id, _)| id.as_str() == "newnode"),
        "a vector inserted after paged load must be findable"
    );
    // A base node (still paged from disk) is also still findable.
    let hits0 = paged.search(&pseudo_vec(42, 32), 3, 64);
    assert!(hits0.iter().any(|(id, _)| id.as_str() == "v42"));
}

/// A sidecar whose size does not match the structure is rejected → the caller
/// rebuilds. Guards against a torn write or a structure/vectors skew.
#[test]
fn from_paged_rejects_mismatched_sidecar() {
    let idx = build(100, 16, AnnConfig::default());
    let structure = idx.to_structure_bytes();
    let mut vecbytes = Vec::new();
    idx.write_vectors_to(&mut vecbytes).unwrap();
    vecbytes.truncate(vecbytes.len() - 4); // one f32 short

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.vectors");
    std::fs::write(&path, &vecbytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    assert!(
        AnnIndex::from_paged(&structure, file, AnnConfig::default(), 1 << 20).is_none(),
        "a sidecar of the wrong length must be rejected"
    );

    // A precision mismatch (SQ8 structure under a full-vector config) is rejected.
    let sq = build(
        100,
        16,
        AnnConfig {
            quantizer: Quantizer::Sq8,
            ..AnnConfig::default()
        },
    );
    let sq_structure = sq.to_structure_bytes();
    let mut sq_vecs = Vec::new();
    sq.write_vectors_to(&mut sq_vecs).unwrap();
    let sq_path = dir.path().join("sq.vectors");
    std::fs::write(&sq_path, &sq_vecs).unwrap();
    let sq_file = std::fs::File::open(&sq_path).unwrap();
    assert!(
        AnnIndex::from_paged(&sq_structure, sq_file, AnnConfig::default(), 1 << 20).is_none(),
        "an SQ8 sidecar under a full-vector config must be rejected"
    );
}
