//! The ANN index: a layered navigable small-world graph, clean-authored.
//!
//! Exact search scans O(N); this graph answers in O(log N)-ish hops. The
//! construction is the classic layered small-world scheme: each node draws a
//! level from a geometric distribution; upper layers form coarse "highways",
//! layer 0 holds everyone. Search greedily descends the highways, then runs a
//! beam (`ef`) at layer 0.
//!
//! Design choices, stated:
//! - **Vectors are normalized on insert**; cosine similarity becomes a dot
//!   product, and internal ordering uses distance `1 − dot`.
//! - **Neighbor selection uses the diversity heuristic** (keep a candidate
//!   only if it is closer to the query than to any already-kept neighbor) —
//!   materially better recall than naive closest-M on clustered data.
//! - **Soft deletes**: removed nodes stay in the graph as tombstones (still
//!   traversable, never returned); compaction reclaims them later.
//! - **Rebuildable by contract**: the estate's durable vector column family
//!   is the source of truth; this graph can always be reconstructed from it
//!   (the two-phase pattern: durable intent first, index apply second).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use rrf_core::{Embedding, Id};

/// Tuning for the graph.
#[derive(Debug, Clone)]
pub struct AnnConfig {
    /// Max neighbors per node per layer (layer 0 gets `2 * m`).
    pub m: usize,
    /// Beam width while building.
    pub ef_construction: usize,
    /// Default beam width while searching (callers may pass larger).
    pub ef_search: usize,
    /// Store vectors as SQ8 codes (~4× smaller). Scores become approximate;
    /// callers holding the full-precision vectors elsewhere should rescore.
    pub quantized: bool,
}

impl Default for AnnConfig {
    fn default() -> Self {
        AnnConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 64,
            quantized: false,
        }
    }
}

/// Vector storage: full-precision f32 or SQ8 codes with per-vector params.
enum Store {
    Full(Vec<f32>),
    Sq8 {
        codes: Vec<u8>,
        params: Vec<crate::quant::SqParams>,
    },
}

impl Store {
    fn push(&mut self, v: &[f32]) {
        match self {
            Store::Full(vectors) => vectors.extend_from_slice(v),
            Store::Sq8 { codes, params } => {
                params.push(crate::quant::quantize_into(v, codes));
            }
        }
    }

    /// Dot of `node`'s stored vector with a full-precision query.
    fn dot_query(&self, node: u32, dim: usize, q: &[f32], qsum: f32) -> f32 {
        let start = node as usize * dim;
        match self {
            Store::Full(vectors) => rrf_core::simd::dot(&vectors[start..start + dim], q),
            Store::Sq8 { codes, params } => {
                crate::quant::dot_query(&codes[start..start + dim], &params[node as usize], q, qsum)
            }
        }
    }

    /// Dot between two stored vectors.
    fn dot_nodes(&self, a: u32, b: u32, dim: usize) -> f32 {
        let (sa, sb) = (a as usize * dim, b as usize * dim);
        match self {
            Store::Full(vectors) => {
                rrf_core::simd::dot(&vectors[sa..sa + dim], &vectors[sb..sb + dim])
            }
            Store::Sq8 { codes, params } => crate::quant::dot_codes(
                &codes[sa..sa + dim],
                &params[a as usize],
                &codes[sb..sb + dim],
                &params[b as usize],
            ),
        }
    }

    /// The (possibly lossy) full-precision vector of `node`.
    fn materialize(&self, node: u32, dim: usize) -> Vec<f32> {
        let start = node as usize * dim;
        match self {
            Store::Full(vectors) => vectors[start..start + dim].to_vec(),
            Store::Sq8 { codes, params } => {
                crate::quant::decode(&codes[start..start + dim], &params[node as usize])
            }
        }
    }

    /// Bytes held by vector storage.
    fn bytes(&self) -> usize {
        match self {
            Store::Full(vectors) => vectors.len() * 4,
            Store::Sq8 { codes, params } => {
                codes.len() + params.len() * std::mem::size_of::<crate::quant::SqParams>()
            }
        }
    }
}

/// Distance-ordered heap entry (min-heap via `Reverse` at use sites).
#[derive(PartialEq)]
struct Scored {
    dist: f32,
    node: u32,
}

impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

/// The layered small-world graph.
pub struct AnnIndex {
    config: AnnConfig,
    dim: Option<usize>,
    /// Flattened, normalized vector storage (node * dim), f32 or SQ8.
    store: Store,
    /// External ids by node.
    ids: Vec<Id>,
    /// External id → node.
    by_id: HashMap<Id, u32>,
    /// Tombstoned nodes (traversable, never returned).
    deleted: Vec<bool>,
    /// links[node][layer] = neighbor nodes.
    links: Vec<Vec<Vec<u32>>>,
    /// Highest occupied layer and its entry node.
    entry: Option<(u32, usize)>,
    /// Deterministic level RNG state.
    rng: u64,
    /// Live (non-tombstoned) count.
    live: usize,
}

impl AnnIndex {
    /// An empty graph.
    pub fn new(config: AnnConfig) -> Self {
        let store = if config.quantized {
            Store::Sq8 {
                codes: Vec::new(),
                params: Vec::new(),
            }
        } else {
            Store::Full(Vec::new())
        };
        AnnIndex {
            config,
            dim: None,
            store,
            ids: Vec::new(),
            by_id: HashMap::new(),
            deleted: Vec::new(),
            links: Vec::new(),
            entry: None,
            rng: 0x9E3779B97F4A7C15,
            live: 0,
        }
    }

    /// Live vector count.
    pub fn len(&self) -> usize {
        self.live
    }

    /// Whether the graph holds no live vectors.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Whether vector storage is SQ8 (scores approximate — rescore if the
    /// full-precision vectors are available elsewhere).
    pub fn is_quantized(&self) -> bool {
        self.config.quantized
    }

    /// Bytes held by vector storage (graph links excluded).
    pub fn vector_bytes(&self) -> usize {
        self.store.bytes()
    }

    fn dist_to(&self, node: u32, query: &[f32], qsum: f32) -> f32 {
        1.0 - self
            .store
            .dot_query(node, self.dim.unwrap_or(0), query, qsum)
    }

    fn next_level(&mut self) -> usize {
        // xorshift → uniform (0,1) → geometric level, capped.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let u = (self.rng >> 11) as f64 / (1u64 << 53) as f64;
        let ml = 1.0 / (self.config.m as f64).ln();
        ((-u.max(1e-12).ln() * ml) as usize).min(16)
    }

    /// Insert (or overwrite) an id with its vector. The vector is normalized
    /// internally; dimension is fixed by the first insert.
    pub fn insert(&mut self, id: Id, embedding: &Embedding) {
        // Overwrite = tombstone the old node, insert fresh.
        if let Some(&old) = self.by_id.get(&id) {
            if !self.deleted[old as usize] {
                self.deleted[old as usize] = true;
                self.live -= 1;
            }
        }

        let normalized = embedding.normalized();
        let v = normalized.as_slice();
        if self.dim.is_none() {
            self.dim = Some(v.len());
        }

        let node = self.ids.len() as u32;
        let level = self.next_level();
        self.store.push(v);
        self.ids.push(id.clone());
        self.by_id.insert(id, node);
        self.deleted.push(false);
        self.links.push(vec![Vec::new(); level + 1]);
        self.live += 1;

        let Some((mut cur, top)) = self.entry else {
            self.entry = Some((node, level));
            return;
        };

        let query: Vec<f32> = v.to_vec();
        let qsum: f32 = query.iter().sum();

        // Greedy descent through layers above the new node's level.
        for layer in ((level + 1)..=top).rev() {
            cur = self.greedy_at(&query, qsum, cur, layer);
        }

        // Beam-connect at each shared layer.
        let ef = self.config.ef_construction;
        for layer in (0..=level.min(top)).rev() {
            let found = self.beam(&query, qsum, cur, layer, ef, /*include_deleted*/ true);
            let max_links = if layer == 0 {
                self.config.m * 2
            } else {
                self.config.m
            };
            let chosen = self.select_diverse(&found, self.config.m);
            for &Scored { node: nb, .. } in &chosen {
                self.links[node as usize][layer].push(nb);
                self.links[nb as usize][layer].push(node);
                // Prune overflowing neighbor lists with the same heuristic.
                if self.links[nb as usize][layer].len() > max_links {
                    self.prune(nb, layer, max_links);
                }
            }
            if let Some(best) = chosen.first() {
                cur = best.node;
            }
        }

        if level > top {
            self.entry = Some((node, level));
        }
    }

    /// Tombstone an id (no-op if absent).
    pub fn remove(&mut self, id: &Id) {
        if let Some(&node) = self.by_id.get(id) {
            if !self.deleted[node as usize] {
                self.deleted[node as usize] = true;
                self.live -= 1;
            }
        }
    }

    /// Search: up to `k` live nearest ids with cosine similarity, best first.
    pub fn search(&self, query: &Embedding, k: usize, ef: usize) -> Vec<(Id, f32)> {
        let Some((mut cur, top)) = self.entry else {
            return Vec::new();
        };
        if k == 0 || self.dim != Some(query.dim()) {
            return Vec::new();
        }
        let q = query.normalized();
        let q = q.as_slice();
        let qsum: f32 = q.iter().sum();

        for layer in (1..=top).rev() {
            cur = self.greedy_at(q, qsum, cur, layer);
        }
        let ef = ef.max(self.config.ef_search).max(k);
        let found = self.beam(q, qsum, cur, 0, ef, /*include_deleted*/ false);

        found
            .into_iter()
            .take(k)
            .map(|s| (self.ids[s.node as usize].clone(), 1.0 - s.dist))
            .collect()
    }

    /// Greedy hill-climb at one layer: move to any closer neighbor until none.
    fn greedy_at(&self, query: &[f32], qsum: f32, start: u32, layer: usize) -> u32 {
        let mut cur = start;
        let mut cur_dist = self.dist_to(cur, query, qsum);
        loop {
            let mut improved = false;
            if let Some(neigh) = self.links[cur as usize].get(layer) {
                for &nb in neigh {
                    let d = self.dist_to(nb, query, qsum);
                    if d < cur_dist {
                        cur = nb;
                        cur_dist = d;
                        improved = true;
                    }
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// Beam search at one layer; returns candidates sorted nearest-first.
    /// Tombstones are traversed always, and included in results only during
    /// construction (`include_deleted`).
    fn beam(
        &self,
        query: &[f32],
        qsum: f32,
        start: u32,
        layer: usize,
        ef: usize,
        include_deleted: bool,
    ) -> Vec<Scored> {
        let mut visited = vec![false; self.ids.len()];
        visited[start as usize] = true;

        let start_dist = self.dist_to(start, query, qsum);
        // Candidates: min-heap by distance (explore closest first).
        let mut candidates: BinaryHeap<std::cmp::Reverse<Scored>> = BinaryHeap::new();
        candidates.push(std::cmp::Reverse(Scored {
            dist: start_dist,
            node: start,
        }));
        // Results: max-heap by distance (evict farthest).
        let mut results: BinaryHeap<Scored> = BinaryHeap::new();
        if include_deleted || !self.deleted[start as usize] {
            results.push(Scored {
                dist: start_dist,
                node: start,
            });
        }

        while let Some(std::cmp::Reverse(Scored { dist, node })) = candidates.pop() {
            let worst = results.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
            if dist > worst && results.len() >= ef {
                break;
            }
            if let Some(neigh) = self.links[node as usize].get(layer) {
                for &nb in neigh {
                    if visited[nb as usize] {
                        continue;
                    }
                    visited[nb as usize] = true;
                    let d = self.dist_to(nb, query, qsum);
                    let worst = results.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
                    if d < worst || results.len() < ef {
                        candidates.push(std::cmp::Reverse(Scored { dist: d, node: nb }));
                        if include_deleted || !self.deleted[nb as usize] {
                            results.push(Scored { dist: d, node: nb });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out = results.into_vec();
        out.sort();
        out
    }

    /// Diversity heuristic: keep a candidate only if it is closer to the
    /// query than to every already-kept neighbor.
    fn select_diverse(&self, sorted: &[Scored], m: usize) -> Vec<Scored> {
        let d = self.dim.unwrap_or(0);
        let mut kept: Vec<Scored> = Vec::with_capacity(m);
        for c in sorted {
            if kept.len() >= m {
                break;
            }
            let dominated = kept.iter().any(|s| {
                let dot = self.store.dot_nodes(c.node, s.node, d);
                (1.0 - dot) < c.dist
            });
            if !dominated {
                kept.push(Scored {
                    dist: c.dist,
                    node: c.node,
                });
            }
        }
        // Never under-fill: pad with the nearest remaining.
        if kept.len() < m {
            for c in sorted {
                if kept.len() >= m {
                    break;
                }
                if !kept.iter().any(|s| s.node == c.node) {
                    kept.push(Scored {
                        dist: c.dist,
                        node: c.node,
                    });
                }
            }
        }
        kept
    }

    /// Re-select a node's neighbor list down to `max_links`.
    fn prune(&mut self, node: u32, layer: usize, max_links: usize) {
        let query = self.store.materialize(node, self.dim.unwrap_or(0));
        let qsum: f32 = query.iter().sum();
        let mut scored: Vec<Scored> = self.links[node as usize][layer]
            .iter()
            .map(|&nb| Scored {
                dist: self.dist_to(nb, &query, qsum),
                node: nb,
            })
            .collect();
        scored.sort();
        let kept = self.select_diverse(&scored, max_links);
        self.links[node as usize][layer] = kept.into_iter().map(|s| s.node).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn build(n: usize, dim: usize) -> (AnnIndex, Vec<Embedding>) {
        let mut idx = AnnIndex::new(AnnConfig::default());
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            let e = Embedding(pseudo_vec(i as u64, dim));
            idx.insert(Id::new(format!("v{i}")), &e);
            vecs.push(e.normalized());
        }
        (idx, vecs)
    }

    fn exact_top_k(vecs: &[Embedding], q: &Embedding, k: usize) -> Vec<usize> {
        let qn = q.normalized();
        let mut scored: Vec<(usize, f32)> =
            vecs.iter().map(|v| v.cosine(&qn)).enumerate().collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn finds_exact_match_and_respects_k() {
        let (idx, vecs) = build(500, 32);
        let hits = idx.search(&vecs[123], 5, 64);
        assert_eq!(hits.len(), 5);
        assert_eq!(hits[0].0.as_str(), "v123");
        assert!(hits[0].1 > 0.999);
    }

    #[test]
    fn recall_at_10_vs_exact_is_high() {
        let n = 5000;
        let dim = 64;
        let (idx, vecs) = build(n, dim);

        let queries = 100;
        let mut found = 0usize;
        let mut total = 0usize;
        for qi in 0..queries {
            let q = Embedding(pseudo_vec(1_000_000 + qi as u64, dim));
            let truth = exact_top_k(&vecs, &q, 10);
            let ann: Vec<String> = idx
                .search(&q, 10, 128)
                .into_iter()
                .map(|(id, _)| id.as_str().to_string())
                .collect();
            for t in truth {
                total += 1;
                if ann.iter().any(|id| id == &format!("v{t}")) {
                    found += 1;
                }
            }
        }
        let recall = found as f64 / total as f64;
        assert!(recall >= 0.95, "recall@10 = {recall:.3}, gate is 0.95");
    }

    #[test]
    fn quantized_recall_gate_and_memory() {
        let n = 5000;
        let dim = 64;
        let mut idx = AnnIndex::new(AnnConfig {
            quantized: true,
            ..AnnConfig::default()
        });
        let mut vecs = Vec::with_capacity(n);
        for i in 0..n {
            let e = Embedding(pseudo_vec(i as u64, dim));
            idx.insert(Id::new(format!("v{i}")), &e);
            vecs.push(e.normalized());
        }
        let full_bytes = n * dim * 4;
        let sq_bytes = idx.vector_bytes();
        assert!(idx.is_quantized());
        assert!(
            sq_bytes * 3 < full_bytes,
            "SQ8 must shrink vector memory ≥3×: {sq_bytes} vs {full_bytes}"
        );

        let queries = 100;
        let mut found = 0usize;
        let mut total = 0usize;
        for qi in 0..queries {
            let q = Embedding(pseudo_vec(1_000_000 + qi as u64, dim));
            let truth = exact_top_k(&vecs, &q, 10);
            let ann: Vec<String> = idx
                .search(&q, 10, 128)
                .into_iter()
                .map(|(id, _)| id.as_str().to_string())
                .collect();
            for t in truth {
                total += 1;
                if ann.iter().any(|id| id == &format!("v{t}")) {
                    found += 1;
                }
            }
        }
        let recall = found as f64 / total as f64;
        println!(
            "SQ8 GATE — recall@10 {recall:.3}, vector bytes {sq_bytes} vs full {full_bytes} ({:.1}x smaller)",
            full_bytes as f64 / sq_bytes as f64
        );
        assert!(
            recall >= 0.90,
            "quantized recall@10 = {recall:.3}, gate is 0.90"
        );
    }

    #[test]
    fn tombstones_never_return_and_overwrite_wins() {
        let (mut idx, vecs) = build(200, 16);
        idx.remove(&"v10".into());
        let hits = idx.search(&vecs[10], 10, 64);
        assert!(hits.iter().all(|(id, _)| id.as_str() != "v10"));
        assert_eq!(idx.len(), 199);

        // Overwrite: v11 gets v20's vector; searching v20's vector returns v11.
        let new = vecs[20].clone();
        idx.insert("v11".into(), &new);
        assert_eq!(idx.len(), 199, "overwrite must not grow live count");
        let hits = idx.search(&new, 3, 64);
        assert!(hits.iter().any(|(id, _)| id.as_str() == "v11"));
    }

    #[test]
    fn empty_and_dim_mismatch_are_safe() {
        let idx = AnnIndex::new(AnnConfig::default());
        assert!(idx.search(&Embedding(vec![1.0, 0.0]), 5, 32).is_empty());
        let (idx, _) = build(50, 8);
        assert!(idx.search(&Embedding(vec![1.0; 16]), 5, 32).is_empty());
    }
}
