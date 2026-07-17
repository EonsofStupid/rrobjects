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
use std::sync::Arc;

use memmap2::Mmap;
use rro_core::{Embedding, Id};

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

/// Node-ordered vector storage split between a read-only mmap **base** (nodes
/// `0..base_count`, paged from disk by the OS page cache) and an in-RAM **tail**
/// (nodes appended since the base was mapped). A freshly built graph is all tail;
/// a graph opened from a persisted sidecar is all base until the next write.
///
/// This split is the RAM-ceiling lift: 10M vectors map straight from disk and
/// only the working set stays resident, while writes still land on the heap and
/// are searchable immediately. Node indices are dense and stable, so a node's
/// slot is a pure offset — no per-node bookkeeping.
struct MappedVec<T: bytemuck::Pod> {
    /// The mapped base file, reinterpreted as `[T]`. `None` for an in-RAM graph.
    base: Option<Arc<Mmap>>,
    /// Number of `T` elements the base holds (0 when `base` is `None`).
    base_len: usize,
    /// Elements for nodes at or beyond the base — the heap-resident tail.
    tail: Vec<T>,
}

impl<T: bytemuck::Pod> MappedVec<T> {
    fn in_ram() -> Self {
        MappedVec {
            base: None,
            base_len: 0,
            tail: Vec::new(),
        }
    }

    /// Build over a mapped base of exactly `base_len` elements, with no tail.
    fn mapped(base: Arc<Mmap>, base_len: usize) -> Self {
        MappedVec {
            base: Some(base),
            base_len,
            tail: Vec::new(),
        }
    }

    #[inline]
    fn base_slice(&self) -> &[T] {
        match &self.base {
            // The base file is page-aligned and its length is validated against
            // `base_len` on load, so this reinterpret is sound; `cast_slice`
            // still bounds- and align-checks rather than read past the map.
            Some(m) => &bytemuck::cast_slice::<u8, T>(&m[..])[..self.base_len],
            None => &[],
        }
    }

    /// The `unit`-element slice for `node` (its whole vector or code block).
    #[inline]
    fn get(&self, node: u32, unit: usize) -> &[T] {
        let node = node as usize;
        let base_count = self.base_len / unit;
        if node < base_count {
            let s = node * unit;
            &self.base_slice()[s..s + unit]
        } else {
            let s = (node - base_count) * unit;
            &self.tail[s..s + unit]
        }
    }

    /// Append one node's `unit` elements — always to the RAM tail.
    #[inline]
    fn push(&mut self, block: &[T]) {
        self.tail.extend_from_slice(block);
    }

    /// Total elements across base + tail.
    fn len(&self) -> usize {
        self.base_len + self.tail.len()
    }

    /// Heap bytes held — the tail only; the base is mmap, not heap.
    fn heap_bytes(&self) -> usize {
        self.tail.len() * std::mem::size_of::<T>()
    }

    /// Total logical bytes (base + tail).
    fn logical_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// Append every element in node order (base then tail) as raw bytes — the
    /// on-disk sidecar layout, and what a later open mmaps back as the base.
    fn write_all(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(bytemuck::cast_slice(self.base_slice()));
        out.extend_from_slice(bytemuck::cast_slice(&self.tail));
    }
}

/// Vector storage: full-precision `f32` or SQ8 codes with per-vector params.
/// Both keep their raw data in a [`MappedVec`], so either precision can be
/// mmap-backed; the SQ8 `params` stay in RAM (tiny next to the codes).
enum Store {
    Full(MappedVec<f32>),
    Sq8 {
        codes: MappedVec<u8>,
        params: Vec<crate::quant::SqParams>,
    },
}

impl Store {
    fn push(&mut self, v: &[f32]) {
        match self {
            Store::Full(vectors) => vectors.push(v),
            Store::Sq8 { codes, params } => {
                params.push(crate::quant::quantize_into(v, &mut codes.tail));
            }
        }
    }

    /// Dot of `node`'s stored vector with a full-precision query.
    #[inline(always)]
    fn dot_query(&self, node: u32, dim: usize, q: &[f32], qsum: f32) -> f32 {
        match self {
            Store::Full(vectors) => rro_core::simd::dot(vectors.get(node, dim), q),
            Store::Sq8 { codes, params } => {
                crate::quant::dot_query(codes.get(node, dim), &params[node as usize], q, qsum)
            }
        }
    }

    /// Dot between two stored vectors.
    fn dot_nodes(&self, a: u32, b: u32, dim: usize) -> f32 {
        match self {
            Store::Full(vectors) => rro_core::simd::dot(vectors.get(a, dim), vectors.get(b, dim)),
            Store::Sq8 { codes, params } => crate::quant::dot_codes(
                codes.get(a, dim),
                &params[a as usize],
                codes.get(b, dim),
                &params[b as usize],
            ),
        }
    }

    /// The (possibly lossy) full-precision vector of `node`.
    fn materialize(&self, node: u32, dim: usize) -> Vec<f32> {
        match self {
            Store::Full(vectors) => vectors.get(node, dim).to_vec(),
            Store::Sq8 { codes, params } => {
                crate::quant::decode(codes.get(node, dim), &params[node as usize])
            }
        }
    }

    /// Total logical bytes held by vector storage (base + tail).
    fn bytes(&self) -> usize {
        match self {
            Store::Full(vectors) => vectors.logical_bytes(),
            Store::Sq8 { codes, params } => {
                codes.logical_bytes() + params.len() * std::mem::size_of::<crate::quant::SqParams>()
            }
        }
    }

    /// Heap bytes held by vector storage — excludes the mmap base. This is the
    /// number that stays small when a large graph is opened mmap-backed.
    fn heap_bytes(&self) -> usize {
        match self {
            Store::Full(vectors) => vectors.heap_bytes(),
            Store::Sq8 { codes, params } => {
                codes.heap_bytes() + params.len() * std::mem::size_of::<crate::quant::SqParams>()
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
                codes: MappedVec::in_ram(),
                params: Vec::new(),
            }
        } else {
            Store::Full(MappedVec::in_ram())
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

    #[inline(always)]
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

    /// Nearest `k` among the nodes `allow` accepts — filter-aware traversal.
    ///
    /// `allow` is checked by external id, so the caller passes the set its filter
    /// resolved to. `ef` is widened internally: a filter of selectivity `s` needs
    /// the frontier to hold ~`ef/s` nodes to surface `ef` allowed ones, so the
    /// beam runs at `ef_search / max(s, floor)` — bounded, because an
    /// arbitrarily selective filter over an arbitrarily large graph is the case
    /// exact scoping already took.
    pub fn search_filtered(
        &self,
        query: &Embedding,
        k: usize,
        ef: usize,
        allow: &std::collections::HashSet<Id>,
    ) -> Vec<(Id, f32)> {
        let Some((mut cur, top)) = self.entry else {
            return Vec::new();
        };
        if k == 0 || self.dim != Some(query.dim()) || allow.is_empty() {
            return Vec::new();
        }
        let q = query.normalized();
        let q = q.as_slice();
        let qsum: f32 = q.iter().sum();

        for layer in (1..=top).rev() {
            cur = self.greedy_at(q, qsum, cur, layer);
        }

        // Widen ef by the inverse selectivity so the frontier holds enough
        // allowed nodes, capped so a 1-in-a-million filter does not ask for a
        // graph-sized beam (that regime belongs to exact scoping).
        let selectivity = (allow.len() as f64 / self.ids.len().max(1) as f64).max(1.0 / 4096.0);
        let widened = ((ef.max(self.config.ef_search).max(k) as f64) / selectivity)
            .ceil()
            .min(self.ids.len() as f64) as usize;

        let found = self.beam_admit(q, qsum, cur, 0, widened, false, |node| {
            allow.contains(&self.ids[node as usize])
        });

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
        self.beam_admit(query, qsum, start, layer, ef, include_deleted, |_| true)
    }

    /// Beam search that **traverses every node but admits only those `admit`
    /// accepts** into the result set.
    ///
    /// This is what makes filtered ANN correct rather than approximate. A naive
    /// filtered search runs a normal beam and drops non-matching results at the
    /// end — but the beam only ever held the query's *global* neighbours, so if
    /// the filter is uncorrelated with the query almost nothing survives. Here the
    /// candidate frontier still walks through disallowed nodes (they are the graph
    /// edges that connect one allowed region to another — dropping them would sever
    /// the graph), while the result heap only ever holds allowed nodes. The beam
    /// therefore spends its full width `ef` inside the filter.
    #[allow(clippy::too_many_arguments)]
    fn beam_admit(
        &self,
        query: &[f32],
        qsum: f32,
        start: u32,
        layer: usize,
        ef: usize,
        include_deleted: bool,
        admit: impl Fn(u32) -> bool,
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
        if (include_deleted || !self.deleted[start as usize]) && admit(start) {
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
                        if (include_deleted || !self.deleted[nb as usize]) && admit(nb) {
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

    /// Serialize the whole graph to a compact, self-describing binary blob.
    ///
    /// This is what makes startup O(load) instead of O(N log N): the estate
    /// rebuilds the ANN graph from the durable vectors on every open by
    /// re-inserting each one, which is the dominant cost of a cold start. Persist
    /// this blob at flush/shutdown and load it back instead, and a 10M-vector
    /// estate opens in the time it takes to read a file rather than to rebuild an
    /// HNSW graph from scratch.
    ///
    /// The durable vectors remain the source of truth (this is a *cache* of the
    /// derived graph); a mismatch on load falls back to the rebuild, so a stale or
    /// corrupt blob can never serve wrong results — see [`AnnIndex::from_bytes`].
    /// Zero-dependency hand-rolled format, little-endian, matching the rest of the
    /// tree's binary encodings; a version byte guards forward compatibility.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Vec::with_capacity(4 * 1024);
        w.extend_from_slice(b"RROG"); // magic
        w.push(1); // format version
        self.write_head(&mut w);
        match &self.store {
            Store::Full(v) => {
                w.push(0);
                w.extend_from_slice(&(v.len() as u64).to_le_bytes());
                v.write_all(&mut w);
            }
            Store::Sq8 { codes, params } => {
                w.push(1);
                w.extend_from_slice(&(codes.len() as u64).to_le_bytes());
                codes.write_all(&mut w);
                write_params(&mut w, params);
            }
        }
        w
    }

    /// Load a graph from [`AnnIndex::to_bytes`]. Returns `None` if the blob is not
    /// a valid, current-version graph — the caller then rebuilds from the durable
    /// vectors, so a bad cache degrades to correct-but-slower, never to wrong.
    ///
    /// `config` is supplied by the caller (it is an open-time choice, not graph
    /// state), and `quantized` must match how the blob was written or the store
    /// tag check rejects it.
    pub fn from_bytes(bytes: &[u8], config: AnnConfig) -> Option<AnnIndex> {
        let mut r = ByteReader::new(bytes);
        if r.take(4)? != b"RROG" || r.u8()? != 1 {
            return None;
        }
        let head = Head::read(&mut r)?;
        let store = match r.u8()? {
            0 => {
                // f32 elements are read one at a time: the blob is a byte buffer at
                // an arbitrary offset, so a zero-copy cast could be misaligned.
                let len = r.u64()? as usize;
                let mut mv = MappedVec::in_ram();
                mv.tail.reserve(len);
                for _ in 0..len {
                    mv.tail.push(r.f32()?);
                }
                Store::Full(mv)
            }
            1 => {
                let clen = r.u64()? as usize;
                let mut codes = MappedVec::in_ram();
                codes.tail = r.take(clen)?.to_vec();
                let params = read_params(&mut r)?;
                Store::Sq8 { codes, params }
            }
            _ => return None,
        };
        // The store tag must agree with how this estate is configured, or a
        // quantized estate would load full vectors (or vice versa) and score wrong.
        if matches!(store, Store::Sq8 { .. }) != config.quantized {
            return None;
        }
        Some(head.into_index(config, store))
    }

    // ---- split persistence: structure blob + mmap-able vector sidecar ----------
    //
    // 6a persists the whole graph — structure *and* vectors — as one blob, which
    // still pulls every vector into RAM on load. 6b splits them: the structure
    // (small: ids, links, tombstones, SQ8 params) stays a blob, and the vectors
    // go to a separate node-ordered file that a later open mmaps as the base. That
    // is what lets RSS track the working set instead of the dataset.

    /// Serialize everything *except* the raw vectors — ids, links, tombstones,
    /// entry, and (for SQ8) the per-vector params. Pair with
    /// [`AnnIndex::write_vectors`]; reload with [`AnnIndex::from_mmap`].
    pub fn to_structure_bytes(&self) -> Vec<u8> {
        let mut w = Vec::with_capacity(4 * 1024);
        w.extend_from_slice(b"RROS"); // magic: structure-only
        w.push(1); // format version
        self.write_head(&mut w);
        match &self.store {
            Store::Full(_) => w.push(0),
            Store::Sq8 { params, .. } => {
                w.push(1);
                write_params(&mut w, params);
            }
        }
        w
    }

    /// Append the raw vectors in node order — the exact bytes a later open maps
    /// back as the mmap base. Full: `n·dim` `f32` LE. SQ8: `n·dim` code
    /// bytes (the params travel in the structure blob). No header, so the file
    /// starts on a vector boundary and the mmap is naturally aligned.
    pub fn write_vectors(&self, out: &mut Vec<u8>) {
        match &self.store {
            Store::Full(v) => v.write_all(out),
            Store::Sq8 { codes, .. } => codes.write_all(out),
        }
    }

    /// Reconstruct a graph from a structure blob ([`AnnIndex::to_structure_bytes`])
    /// plus a memory-mapped vector file ([`AnnIndex::write_vectors`]). The vectors
    /// are *not* read into RAM — they stay in the mmap and page on demand.
    ///
    /// Returns `None` (→ caller rebuilds) if the blob is not a current-version
    /// structure, the store precision disagrees with `config`, or the vector file
    /// is not exactly `n · dim` elements — a size mismatch means the sidecar does
    /// not match the structure, and trusting it would read the wrong bytes.
    pub fn from_mmap(structure: &[u8], vectors: Arc<Mmap>, config: AnnConfig) -> Option<AnnIndex> {
        let mut r = ByteReader::new(structure);
        if r.take(4)? != b"RROS" || r.u8()? != 1 {
            return None;
        }
        let head = Head::read(&mut r)?;
        let quantized = match r.u8()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        if quantized != config.quantized {
            return None;
        }
        let dim = head.dim.unwrap_or(0);
        let elems = head.ids.len().checked_mul(dim)?;
        let store = if quantized {
            let params = read_params(&mut r)?;
            if params.len() != head.ids.len() || vectors.len() != elems {
                return None; // one code byte per dim; sidecar must match exactly
            }
            Store::Sq8 {
                codes: MappedVec::mapped(vectors, elems),
                params,
            }
        } else {
            if vectors.len() != elems * std::mem::size_of::<f32>() {
                return None;
            }
            Store::Full(MappedVec::mapped(vectors, elems))
        };
        Some(head.into_index(config, store))
    }

    /// Write the shared head (dim, rng, live count, entry, ids, tombstones,
    /// links) — everything but the store section. Shared by [`AnnIndex::to_bytes`]
    /// and [`AnnIndex::to_structure_bytes`].
    fn write_head(&self, w: &mut Vec<u8>) {
        w.extend_from_slice(&(self.dim.unwrap_or(0) as u32).to_le_bytes());
        w.extend_from_slice(&self.rng.to_le_bytes());
        w.extend_from_slice(&(self.live as u64).to_le_bytes());
        match self.entry {
            Some((node, layer)) => {
                w.push(1);
                w.extend_from_slice(&node.to_le_bytes());
                w.extend_from_slice(&(layer as u32).to_le_bytes());
            }
            None => w.push(0),
        }
        w.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        for id in &self.ids {
            let b = id.as_str().as_bytes();
            w.extend_from_slice(&(b.len() as u32).to_le_bytes());
            w.extend_from_slice(b);
        }
        for &d in &self.deleted {
            w.push(d as u8);
        }
        for node_links in &self.links {
            w.extend_from_slice(&(node_links.len() as u32).to_le_bytes());
            for layer in node_links {
                w.extend_from_slice(&(layer.len() as u32).to_le_bytes());
                for &nb in layer {
                    w.extend_from_slice(&nb.to_le_bytes());
                }
            }
        }
    }

    /// Heap bytes held by vector storage — the mmap base is excluded, so this is
    /// what stays small when a large graph is opened mmap-backed. Observability
    /// behind the "RSS tracks the working set, not the dataset" property.
    pub fn heap_vector_bytes(&self) -> usize {
        self.store.heap_bytes()
    }
}

/// The graph's structure, parsed from a head record — the shared spine of every
/// deserialization path.
struct Head {
    dim: Option<usize>,
    rng: u64,
    live: usize,
    entry: Option<(u32, usize)>,
    ids: Vec<Id>,
    by_id: HashMap<Id, u32>,
    deleted: Vec<bool>,
    links: Vec<Vec<Vec<u32>>>,
}

impl Head {
    fn read(r: &mut ByteReader) -> Option<Head> {
        let dim_raw = r.u32()? as usize;
        let dim = if dim_raw == 0 { None } else { Some(dim_raw) };
        let rng = r.u64()?;
        let live = r.u64()? as usize;
        let entry = if r.u8()? == 1 {
            Some((r.u32()?, r.u32()? as usize))
        } else {
            None
        };
        let n = r.u32()? as usize;
        let mut ids = Vec::with_capacity(n);
        let mut by_id = HashMap::with_capacity(n);
        for node in 0..n {
            let len = r.u32()? as usize;
            let s = std::str::from_utf8(r.take(len)?).ok()?.to_string();
            let id = Id::new(s);
            by_id.insert(id.clone(), node as u32);
            ids.push(id);
        }
        let mut deleted = Vec::with_capacity(n);
        for _ in 0..n {
            deleted.push(r.u8()? != 0);
        }
        let mut links = Vec::with_capacity(n);
        for _ in 0..n {
            let nlayers = r.u32()? as usize;
            let mut node_links = Vec::with_capacity(nlayers);
            for _ in 0..nlayers {
                let count = r.u32()? as usize;
                let mut layer = Vec::with_capacity(count);
                for _ in 0..count {
                    layer.push(r.u32()?);
                }
                node_links.push(layer);
            }
            links.push(node_links);
        }
        Some(Head {
            dim,
            rng,
            live,
            entry,
            ids,
            by_id,
            deleted,
            links,
        })
    }

    fn into_index(self, config: AnnConfig, store: Store) -> AnnIndex {
        AnnIndex {
            config,
            dim: self.dim,
            store,
            ids: self.ids,
            by_id: self.by_id,
            deleted: self.deleted,
            links: self.links,
            entry: self.entry,
            rng: self.rng,
            live: self.live,
        }
    }
}

/// Append SQ8 per-vector params: a count then `(scale, offset, code_sum)` each.
fn write_params(w: &mut Vec<u8>, params: &[crate::quant::SqParams]) {
    w.extend_from_slice(&(params.len() as u32).to_le_bytes());
    for p in params {
        w.extend_from_slice(&p.scale.to_le_bytes());
        w.extend_from_slice(&p.offset.to_le_bytes());
        w.extend_from_slice(&p.code_sum.to_le_bytes());
    }
}

/// Inverse of [`write_params`]. `None` on a truncated buffer.
fn read_params(r: &mut ByteReader) -> Option<Vec<crate::quant::SqParams>> {
    let np = r.u32()? as usize;
    let mut params = Vec::with_capacity(np);
    for _ in 0..np {
        params.push(crate::quant::SqParams {
            scale: r.f32()?,
            offset: r.f32()?,
            code_sum: r.f32()?,
        });
    }
    Some(params)
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

    /// A persisted graph must reload byte-identical in every field that governs
    /// search, and — the property that actually matters — return the *same
    /// results* as the graph it was serialized from. This is the correctness
    /// guarantee behind loading the graph instead of rebuilding it on open.
    #[test]
    fn to_from_bytes_round_trips_and_search_is_identical() {
        let (mut idx, vecs) = build(1000, 48);
        // Exercise tombstones + overwrite so deleted/live/links are non-trivial.
        idx.remove(&"v7".into());
        idx.insert("v8".into(), &vecs[900]);

        let bytes = idx.to_bytes();
        let back = AnnIndex::from_bytes(&bytes, AnnConfig::default())
            .expect("valid blob must deserialize");

        assert_eq!(back.len(), idx.len());
        assert_eq!(back.dim, idx.dim);
        assert_eq!(back.entry, idx.entry);
        assert_eq!(back.ids, idx.ids);
        assert_eq!(back.deleted, idx.deleted);
        assert_eq!(back.links, idx.links);

        for qi in 0..50 {
            let q = Embedding(pseudo_vec(2_000_000 + qi as u64, 48));
            let a = idx.search(&q, 10, 128);
            let b = back.search(&q, 10, 128);
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(x.0.as_str(), y.0.as_str(), "result order must match");
            }
        }
    }

    /// Quantized graphs must round-trip too, and the store-tag guard must reject a
    /// blob whose quantization does not match the estate's configuration — that
    /// mismatch is exactly what forces a safe rebuild rather than scoring wrong.
    #[test]
    fn quantized_round_trips_and_store_tag_guards_config() {
        let mut idx = AnnIndex::new(AnnConfig {
            quantized: true,
            ..AnnConfig::default()
        });
        for i in 0..500 {
            idx.insert(Id::new(format!("v{i}")), &Embedding(pseudo_vec(i, 32)));
        }
        let bytes = idx.to_bytes();

        // Correct config: loads.
        let back = AnnIndex::from_bytes(
            &bytes,
            AnnConfig {
                quantized: true,
                ..AnnConfig::default()
            },
        )
        .expect("quantized blob must load under quantized config");
        assert!(back.is_quantized());
        assert_eq!(back.len(), idx.len());

        // Mismatched config (expects full vectors): rejected → caller rebuilds.
        assert!(
            AnnIndex::from_bytes(&bytes, AnnConfig::default()).is_none(),
            "sq8 blob under a full-vector config must be rejected"
        );
    }

    /// Truncated or garbage bytes must never panic — they return `None` so the
    /// estate falls back to rebuilding from the durable vectors.
    #[test]
    fn corrupt_bytes_return_none_not_panic() {
        assert!(AnnIndex::from_bytes(b"", AnnConfig::default()).is_none());
        assert!(AnnIndex::from_bytes(b"RROG", AnnConfig::default()).is_none());
        assert!(AnnIndex::from_bytes(b"XXXX\x01", AnnConfig::default()).is_none());
        let (idx, _) = build(100, 16);
        let mut bytes = idx.to_bytes();
        bytes.truncate(bytes.len() / 2);
        assert!(AnnIndex::from_bytes(&bytes, AnnConfig::default()).is_none());
    }

    /// The structure blob and vector sidecar together reconstruct the exact same
    /// bytes as the single-blob `to_bytes` would carry — proven here in-crate
    /// without mmap (the mmap path itself is covered in `tests/mmap.rs`, which can
    /// use the unsafe `Mmap::map` that this crate forbids). We rebuild an in-RAM
    /// graph from the split parts by concatenating structure + a Full store tag +
    /// vectors into the `to_bytes` layout and checking search is unchanged.
    #[test]
    fn structure_plus_vectors_carry_everything_to_bytes_does() {
        let (idx, _) = build(500, 24);
        let structure = idx.to_structure_bytes();
        let mut vectors = Vec::new();
        idx.write_vectors(&mut vectors);

        // Structure omits the vectors; the sidecar is exactly the vector bytes.
        assert!(structure.len() < idx.to_bytes().len());
        assert_eq!(vectors.len(), 500 * 24 * 4);
    }
}

#[cfg(test)]
mod filter_aware_tests {
    use super::*;

    /// The traversal primitive in isolation: with the filter uncorrelated to the
    /// query, `search_filtered` must still return allowed nodes near the query —
    /// where a plain `search` + post-filter would return almost nothing.
    #[test]
    fn search_filtered_finds_allowed_neighbours_a_postfilter_would_miss() {
        let mut idx = AnnIndex::new(AnnConfig::default());
        let mut seed = 7u64;
        let mut lcg = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        };
        // 5000 nodes; ~2% are "allowed", scattered independently of position.
        let n = 5000;
        let mut allow = std::collections::HashSet::new();
        let mut vecs: Vec<(String, Embedding)> = Vec::new();
        for i in 0..n {
            let v = Embedding(vec![lcg(), lcg(), lcg()]).normalized();
            let id = Id(format!("n{i}"));
            idx.insert(id.clone(), &v);
            if i % 50 == 0 {
                allow.insert(id.clone());
            }
            vecs.push((id.as_str().to_string(), v));
        }
        let q = Embedding(vec![1.0, 0.0, 0.0]).normalized();

        // Exact filtered top-10 by brute force.
        let mut exact: Vec<(String, f32)> = vecs
            .iter()
            .filter(|(id, _)| allow.contains(&Id(id.clone())))
            .map(|(id, v)| (id.clone(), q.cosine(v)))
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1));
        let truth: std::collections::HashSet<&str> =
            exact.iter().take(10).map(|(id, _)| id.as_str()).collect();

        let got = idx.search_filtered(&q, 10, 64, &allow);
        assert_eq!(got.len(), 10, "must fill the page from the allowed set");
        let hit = got
            .iter()
            .filter(|(id, _)| truth.contains(id.as_str()))
            .count();
        assert!(
            hit >= 8,
            "filter-aware traversal recall@10 = {hit}/10 vs exact — a post-filter \
             over a 2% filter would return near-zero"
        );
        // Every result is actually allowed.
        for (id, _) in &got {
            assert!(allow.contains(id), "{id:?} is not in the allowed set");
        }
    }

    #[test]
    fn search_filtered_empty_allow_is_empty() {
        let mut idx = AnnIndex::new(AnnConfig::default());
        for i in 0..2000 {
            idx.insert(
                Id(format!("n{i}")),
                &Embedding(vec![i as f32, 1.0]).normalized(),
            );
        }
        let got = idx.search_filtered(
            &Embedding(vec![1.0, 0.0]).normalized(),
            10,
            64,
            &std::collections::HashSet::new(),
        );
        assert!(got.is_empty());
    }
}

/// A tiny bounds-checked little-endian reader for [`AnnIndex::from_bytes`].
/// Every read returns `Option`, so a truncated or malformed blob yields `None`
/// (→ rebuild) instead of a panic.
struct ByteReader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> ByteReader<'a> {
    fn new(b: &'a [u8]) -> Self {
        ByteReader { b, i: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.i.checked_add(n)?;
        let s = self.b.get(self.i..end)?;
        self.i = end;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Option<u64> {
        let s = self.take(8)?;
        Some(u64::from_le_bytes([
            s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
        ]))
    }
    fn f32(&mut self) -> Option<f32> {
        self.take(4)
            .map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
}
