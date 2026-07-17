//! Phase 6b — mmap-backed vector storage. The graph's vectors live in a
//! memory-mapped sidecar file instead of the heap, so a large graph opens with
//! RSS tracking the working set rather than the whole dataset.
//!
//! These live in an integration test (a separate crate) on purpose: mapping a
//! file is inherently unsafe — the caller must guarantee the bytes do not change
//! under the map — and the `recall` library itself is `#![forbid(unsafe_code)]`.
//! So `recall` exposes only `from_mmap(structure, Arc<Mmap>, config)`, which is
//! safe, and the unsafe `Mmap::map` lives here (and, in production, in the
//! estate that owns the file).

use std::sync::Arc;

use memmap2::Mmap;
use recall::{AnnConfig, AnnIndex};
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

/// Persist `idx`'s structure + vectors, then map the vectors back and return the
/// reloaded graph. The returned `TempDir` must be held for the mmap's lifetime.
fn persist_and_mmap(idx: &AnnIndex, config: AnnConfig) -> (AnnIndex, tempfile::TempDir) {
    let structure = idx.to_structure_bytes();
    let mut vecbytes = Vec::new();
    idx.write_vectors(&mut vecbytes);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.vectors");
    std::fs::write(&path, &vecbytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    // SAFETY: the file is written once above and never mutated while mapped.
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let mapped = AnnIndex::from_mmap(&structure, Arc::new(mmap), config).unwrap();
    (mapped, dir)
}

/// The core 6b property: a graph reloaded with its vectors in an mmap returns
/// byte-identical search results to the in-RAM graph — while holding no vector
/// heap (the vectors live in the mapped file and page on demand).
#[test]
fn mmap_backed_search_is_identical_and_holds_no_vector_heap() {
    let idx = build(2000, 48, AnnConfig::default());
    assert!(
        idx.heap_vector_bytes() > 0,
        "an in-RAM graph holds its vectors on the heap"
    );

    let (mapped, _dir) = persist_and_mmap(&idx, AnnConfig::default());
    assert_eq!(
        mapped.heap_vector_bytes(),
        0,
        "an mmap-backed full-precision graph holds NO vector heap"
    );
    assert_eq!(mapped.len(), idx.len());

    for qi in 0..50 {
        let q = pseudo_vec(3_000_000 + qi as u64, 48);
        let a = idx.search(&q, 10, 128);
        let b = mapped.search(&q, 10, 128);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.0.as_str(), y.0.as_str(), "mmap ranking must match in-RAM");
            assert!((x.1 - y.1).abs() < 1e-6, "mmap scores must match in-RAM");
        }
    }
}

/// Quantized graphs split too: codes go to the mmap, params ride in the structure
/// blob. Reload reproduces the same ranking and holds only the (tiny) params on
/// the heap, not the codes.
#[test]
fn mmap_backed_quantized_round_trips() {
    let config = AnnConfig {
        quantized: true,
        ..AnnConfig::default()
    };
    let idx = build(1500, 32, config.clone());
    let heap_before = idx.heap_vector_bytes();

    let (mapped, _dir) = persist_and_mmap(&idx, config);
    assert!(mapped.is_quantized());
    // The codes (1500·32 = 48000 bytes) move to the mmap; the small params stay.
    // The heap must therefore drop by at least the whole code size.
    let after = mapped.heap_vector_bytes();
    assert!(
        heap_before - after >= 1500 * 32,
        "mmap must evict the codes from the heap (before {heap_before}, after {after})"
    );
    assert!(after > 0, "the params correctly remain resident");

    for i in (0..1500u64).step_by(30) {
        let q = pseudo_vec(i, 32);
        let a: Vec<_> = idx.search(&q, 5, 64);
        let b: Vec<_> = mapped.search(&q, 5, 64);
        let a_ids: Vec<_> = a.iter().map(|(id, _)| id.as_str().to_string()).collect();
        let b_ids: Vec<_> = b.iter().map(|(id, _)| id.as_str().to_string()).collect();
        assert_eq!(a_ids, b_ids, "mmap quantized ranking must match in-RAM");
    }
}

/// After loading mmap-backed, new inserts land in the RAM tail and are
/// immediately searchable, while base nodes stay in the mmap — read-your-writes
/// on top of a paged base.
#[test]
fn inserts_after_mmap_load_go_to_tail_and_are_searchable() {
    let idx = build(1000, 32, AnnConfig::default());
    let (mut mapped, _dir) = persist_and_mmap(&idx, AnnConfig::default());
    assert_eq!(mapped.heap_vector_bytes(), 0);

    let newv = pseudo_vec(7_777_777, 32);
    mapped.insert(Id::new("newnode"), &newv);
    assert!(
        mapped.heap_vector_bytes() > 0,
        "the appended vector must live in the RAM tail"
    );

    let hits = mapped.search(&newv, 5, 128);
    assert!(
        hits.iter().any(|(id, _)| id.as_str() == "newnode"),
        "a vector inserted after mmap load must be findable"
    );
    // A base node (still served from the mmap) is also still findable.
    let hits0 = mapped.search(&pseudo_vec(42, 32), 3, 64);
    assert!(hits0.iter().any(|(id, _)| id.as_str() == "v42"));
}

/// A sidecar whose size does not match the structure is rejected → the caller
/// rebuilds. Guards against a torn write or a structure/vectors skew.
#[test]
fn from_mmap_rejects_mismatched_sidecar() {
    let idx = build(100, 16, AnnConfig::default());
    let structure = idx.to_structure_bytes();
    let mut vecbytes = Vec::new();
    idx.write_vectors(&mut vecbytes);
    vecbytes.truncate(vecbytes.len() - 4); // one f32 short

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graph.vectors");
    std::fs::write(&path, &vecbytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    assert!(
        AnnIndex::from_mmap(&structure, Arc::new(mmap), AnnConfig::default()).is_none(),
        "a sidecar of the wrong length must be rejected"
    );

    // A precision mismatch (SQ8 blob under a full-vector config) is rejected too.
    let sq = build(
        100,
        16,
        AnnConfig {
            quantized: true,
            ..AnnConfig::default()
        },
    );
    let sq_structure = sq.to_structure_bytes();
    let mut sq_vecs = Vec::new();
    sq.write_vectors(&mut sq_vecs);
    let sq_path = dir.path().join("sq.vectors");
    std::fs::write(&sq_path, &sq_vecs).unwrap();
    let sq_file = std::fs::File::open(&sq_path).unwrap();
    let sq_mmap = unsafe { Mmap::map(&sq_file).unwrap() };
    assert!(
        AnnIndex::from_mmap(&sq_structure, Arc::new(sq_mmap), AnnConfig::default()).is_none(),
        "an SQ8 sidecar under a full-vector config must be rejected"
    );
}
