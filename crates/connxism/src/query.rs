//! Query execution: one spec ([`rro_core::EstateQuery`], pure data in the
//! core contract), every retrieval capability of the estate.
//!
//! Filter execution is **three-strategy**, chosen by the exact cardinality the
//! payload index already resolves (no estimation): **exact scoping** (score the
//! whole matched id-set when it is ≤ `EXACT_SCOPE_MAX`), **filter-aware graph
//! traversal** (walk the HNSW graph admitting only allowed nodes, when the
//! matched set is larger), and **post-filter** (over-fetch + hydrate + retain)
//! only when the filter cannot be resolved from indexes at all. Facets, filtered
//! counts, and cursor-paged **scroll** live beside it on the estate.

use rro_core::{Candidate, Embedding, EstateQuery, Metadata, Result};

use crate::estate::{rocks_err, Estate};
use crate::keys::CF_DOCS;
use crate::model::StoredDoc;
use crate::store::ConnXRecall;

/// How hard filtered searches over-fetch before post-filtering.
const FILTER_OVERFETCH: usize = 8;

/// Match count up to which an indexed filter is scored **exactly** over its
/// resolved id set.
///
/// Exact cosine over the matched set is one point-lookup + one dot per id —
/// measured at ~2.7 ms for 5,000 ids — and it is correct at every selectivity.
/// The old value was 4,096, which was not a cost limit but a *correctness cliff*:
/// a filter matching 4,097 docs fell to a global post-filter that returned the
/// query's global neighbours intersected with the filter, i.e. almost nothing
/// for an uncorrelated filter. 65,536 keeps exact scoring in the sub-100 ms band
/// while covering essentially every real filtered query; the rare larger matched
/// set takes the filter-aware graph traversal, which is also correct.
const EXACT_SCOPE_MAX: usize = 65_536;

/// The standard reciprocal-rank-fusion constant (same as the hybrid path).
const FUSION_RRF_K: f32 = 60.0;

impl ConnXRecall {
    /// Execute a typed query. Strategy, in order: prefetch stages (if any)
    /// gather the id universe; an explicit scope wins/intersects; otherwise
    /// a fully-indexed filter resolves its exact id-set first and scores
    /// inside it; otherwise hybrid (ANN + BM25, fused) with over-fetch +
    /// post-filter.
    pub async fn query(&self, q: EstateQuery) -> Result<Vec<Candidate>> {
        if let Some(cap) = self.quota_max_top_k() {
            if q.top_k > cap {
                return Err(rro_core::RroError::Quota(format!(
                    "top_k {} exceeds max_top_k {cap}",
                    q.top_k
                )));
            }
        }
        self.query_depth(q, 0).await
    }

    /// Prefetch stages may nest; three levels is a pipeline, more is a bug.
    const MAX_PREFETCH_DEPTH: usize = 3;

    fn query_depth(
        &self,
        mut q: EstateQuery,
        depth: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<Candidate>>> + Send + '_>>
    {
        Box::pin(async move {
            // Prefetch: run each stage (recursively), union their ids, and
            // fold the union into the scope the outer query executes in.
            if !q.prefetch.is_empty() {
                if depth >= Self::MAX_PREFETCH_DEPTH {
                    return Err(rro_core::RroError::Recall(format!(
                        "prefetch nesting exceeds {} levels",
                        Self::MAX_PREFETCH_DEPTH
                    )));
                }
                let stages = std::mem::take(&mut q.prefetch);
                let mut union: Vec<String> = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for stage in stages {
                    let mut inner = stage.query;
                    inner.top_k = stage.limit;
                    inner.with_payload = false;
                    for c in self.query_depth(inner, depth + 1).await? {
                        if seen.insert(c.id.as_str().to_string()) {
                            union.push(c.id.as_str().to_string());
                        }
                    }
                }
                q.scope = Some(match q.scope.take() {
                    None => union,
                    Some(explicit) => {
                        let allow: std::collections::HashSet<&str> =
                            explicit.iter().map(String::as_str).collect();
                        union
                            .into_iter()
                            .filter(|id| allow.contains(id.as_str()))
                            .collect()
                    }
                });
            }
            self.query_flat(q).await
        })
    }

    /// The non-recursive body (prefetch already folded into `scope`).
    async fn query_flat(&self, q: EstateQuery) -> Result<Vec<Candidate>> {
        let top_k = q.top_k;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let text = q.text.clone().unwrap_or_default();
        let vector = match &q.vector {
            Some(v) => v.clone(),
            None => Embedding(Vec::new()),
        };

        let dsl = q.effective_filter();
        // Pagination ranks to offset+k depth, then skips.
        let want = top_k.saturating_add(q.offset);
        // Filters post-filter and MaxSim reorders — both need a candidate
        // set deeper than the page for the winners to be *in* it.
        let fetch = if dsl.is_empty() && q.multi.is_none() {
            want
        } else {
            want.saturating_mul(FILTER_OVERFETCH)
        };

        // A named collection restricts the id universe exactly like an
        // explicit scope; both given → the intersection.
        let scope: Option<Vec<String>> = match (&q.collection, &q.scope) {
            (None, s) => s.clone(),
            (Some(coll), s) => {
                let coll = self.resolve_alias(coll).await?;
                let members = self.collection_members_blocking(&coll).await?;
                match s {
                    None => Some(members),
                    Some(explicit) => {
                        let allow: std::collections::HashSet<&str> =
                            members.iter().map(String::as_str).collect();
                        Some(
                            explicit
                                .iter()
                                .filter(|id| allow.contains(id.as_str()))
                                .cloned()
                                .collect(),
                        )
                    }
                }
            }
        };

        // `prefiltered` = the candidate set already satisfies the filter
        // exactly (resolved from payload indexes); no post-filter needed.
        // `allowed` = the id universe a scope or prefilter restricts to —
        // any extra ranking fused in below must respect it too.
        let mut prefiltered = false;
        let mut allowed: Option<std::collections::HashSet<String>> = None;
        let mut results = match &scope {
            Some(scope) => {
                allowed = Some(scope.iter().cloned().collect());
                self.scoped_search(&text, &vector, fetch, scope.clone())
                    .await?
            }
            None if !dsl.is_empty() => match crate::filter::ids_where(&self.db, &dsl)? {
                // Fully-indexed filter: `ids` is the EXACT matching set, and its
                // length is the exact cardinality — no estimation needed.
                //
                // Score it exactly whenever that is affordable. Exact cosine over
                // the set is a point-lookup per id (~0.5 µs warm) plus a dot
                // product — measured at 2.7 ms for 5,000 ids — and it is correct
                // at *every* selectivity. The old code capped this at 4,096 and
                // fell to global post-filter above it, which is the bug this
                // phase exists to kill: a filter matching 5,000 of 200,000 docs
                // (2.5%) exceeded the cap, ran a global ANN fetching k×8=80
                // candidates, and `retain` kept only the ~2 that happened to fall
                // in the bucket — a top-10 request answered with 1 result,
                // silently. Filtered ANN must return the filtered nearest
                // neighbours, not the global ones that survive a filter.
                Some(ids) if ids.len() <= EXACT_SCOPE_MAX => {
                    prefiltered = true;
                    allowed = Some(ids.iter().cloned().collect());
                    if ids.is_empty() {
                        Vec::new()
                    } else {
                        self.scoped_search(&text, &vector, want, ids).await?
                    }
                }
                // A genuinely huge matched set (> EXACT_SCOPE_MAX). Exact scoring
                // would be O(matches) and slow, but global post-filter would be
                // *wrong* — so this is the filter-aware ANN path: walk the graph
                // and keep only allowed nodes, so the beam spends its whole width
                // inside the filter instead of on global neighbours that will be
                // thrown away. Correct at any selectivity, sub-linear in the
                // matched set.
                Some(ids) => {
                    prefiltered = true;
                    let allow: std::collections::HashSet<String> = ids.into_iter().collect();
                    allowed = Some(allow.clone());
                    self.filter_aware_search(&text, &vector, want, &allow, q.fusion, q.fusion_mode)
                        .await?
                }
                // Filter not resolvable from indexes at all — the only option is
                // over-fetch + post-filter, with the over-fetch scaled up so the
                // page is likely to fill even for a selective predicate.
                None => {
                    self.unscoped(
                        &text,
                        &vector,
                        q.vector.is_some(),
                        fetch,
                        q.fusion,
                        q.fusion_mode,
                    )
                    .await?
                }
            },
            None => match (&q.using, &q.vector) {
                // Named space: the dense ranking is exact cosine inside that
                // space; a lexical ranking (if text) fuses in as usual.
                (Some(space), Some(v)) => {
                    self.named_hybrid(space, &text, v, fetch, q.fusion, q.fusion_mode)
                        .await?
                }
                _ => {
                    self.unscoped(
                        &text,
                        &vector,
                        q.vector.is_some(),
                        fetch,
                        q.fusion,
                        q.fusion_mode,
                    )
                    .await?
                }
            },
        };

        // Weighted sparse: a third ranking, fused by reciprocal rank fusion
        // with whatever the dense/lexical strategy produced.
        if let Some(sv) = &q.sparse {
            if !sv.is_empty() {
                let mut sparse = self.sparse_search(sv, fetch).await?;
                if let Some(allowed) = &allowed {
                    sparse.retain(|c| allowed.contains(c.id.as_str()));
                }
                if !sparse.is_empty() {
                    let scored = [
                        results
                            .iter()
                            .map(|c| (c.id.as_str().to_string(), c.score))
                            .collect::<Vec<_>>(),
                        sparse
                            .iter()
                            .map(|c| (c.id.as_str().to_string(), c.score))
                            .collect::<Vec<_>>(),
                    ];
                    let fused = crate::index::fuse(
                        q.fusion_mode,
                        &scored,
                        &q.fusion.as_slice(),
                        FUSION_RRF_K,
                    );
                    let mut out = Vec::with_capacity(fetch.min(fused.len()));
                    for (id, score) in fused.into_iter().take(fetch) {
                        if let Some(doc) = self.doc(&id).await? {
                            let mut c = Candidate::new(doc.id, doc.text, score);
                            c.metadata = doc.metadata;
                            out.push(c);
                        }
                    }
                    results = out;
                }
            }
        }

        // Late interaction: rescore the fetch-deep candidate set by MaxSim
        // against stored token vectors (docs without any keep first-phase
        // order after the scored ones).
        if let Some(tokens) = &q.multi {
            if !tokens.is_empty() {
                results = self.maxsim_rescore(results, tokens).await?;
                // Hydrate winners that came from lean paths.
                for c in results.iter_mut() {
                    if c.text.is_empty() {
                        if let Some(doc) = self.doc(c.id.as_str()).await? {
                            c.text = doc.text;
                            if c.metadata.is_empty() {
                                c.metadata = doc.metadata;
                            }
                        }
                    }
                }
            }
        }

        if !dsl.is_empty() && !prefiltered {
            // Lexical-path candidates may carry empty payloads; hydrate before
            // filtering so the clauses see real metadata.
            for c in results.iter_mut() {
                if c.metadata.is_empty() {
                    if let Some(doc) = self.doc(c.id.as_str()).await? {
                        c.metadata = doc.metadata;
                        if c.text.is_empty() {
                            c.text = doc.text;
                        }
                    }
                }
            }
            results.retain(|c| dsl.matches(&c.metadata));
        }
        if let Some(t) = q.score_threshold {
            results.retain(|c| c.score >= t);
        }
        // Pagination: skip the offset, take the page.
        if q.offset > 0 {
            results = results.into_iter().skip(q.offset).collect();
        }
        results.truncate(top_k);
        if !q.with_payload {
            for c in results.iter_mut() {
                c.text.clear();
                c.metadata = Metadata::new();
            }
        }
        if q.with_vectors {
            for c in results.iter_mut() {
                c.vector = self.vector_of(c.id.as_str()).await?;
            }
        }
        if q.highlight {
            if let Some(qt) = &q.text {
                let analyzer = self.analyzer_ref();
                for c in results.iter_mut() {
                    if !c.text.is_empty() {
                        c.highlights = analyzer.highlight(&c.text, qt);
                    }
                }
            }
        }
        Ok(results)
    }

    /// Resolve a collection alias to its target (or pass the name through).
    async fn resolve_alias(&self, name: &str) -> Result<String> {
        let db = self.db.clone();
        let name = name.to_string();
        tokio::task::spawn_blocking(move || {
            let aliases: std::collections::BTreeMap<String, String> = db
                .get_json(crate::keys::CF_META, crate::keys::META_ALIASES)?
                .unwrap_or_default();
            Ok(aliases.get(&name).cloned().unwrap_or(name))
        })
        .await
        .map_err(|e| rro_core::RroError::Recall(format!("join: {e}")))?
    }

    /// One collection's member ids, off the blocking pool.
    async fn collection_members_blocking(&self, name: &str) -> Result<Vec<String>> {
        let db = self.db.clone();
        let name = name.to_string();
        tokio::task::spawn_blocking(move || {
            let handle = db.cf(crate::keys::CF_COLL)?;
            let prefix = crate::keys::coll_prefix(&name);
            let mut out = Vec::new();
            for item in db.0.iterator_cf(
                handle,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            ) {
                let (k, _) = item.map_err(crate::estate::rocks_err)?;
                if !k.starts_with(&prefix) {
                    break;
                }
                out.push(String::from_utf8_lossy(&k[prefix.len()..]).into_owned());
            }
            Ok(out)
        })
        .await
        .map_err(|e| rro_core::RroError::Recall(format!("join: {e}")))?
    }

    /// The default retrieval path: hybrid when a vector is present, lexical
    /// otherwise. `weights` is the query's fusion decision and must be threaded
    /// through — this is the path almost every query takes, so a knob that
    /// misses it is a knob that does nothing.
    async fn unscoped(
        &self,
        text: &str,
        vector: &Embedding,
        has_vector: bool,
        fetch: usize,
        weights: rro_core::HybridWeights,
        mode: rro_core::FusionMode,
    ) -> Result<Vec<Candidate>> {
        if has_vector {
            self.hybrid_weighted(text, vector, fetch, weights, mode)
                .await
        } else {
            self.lexical_search(text, fetch).await
        }
    }

    /// Dense ranking from a named space, fused with lexical when text is
    /// present, winners hydrated from the doc store.
    /// Hybrid over a *named* vector space. `weights` is the query's fusion
    /// decision — the named path must honour it exactly like the default path,
    /// or `using:` would silently change how your rankings are fused.
    async fn named_hybrid(
        &self,
        space: &str,
        text: &str,
        vector: &Embedding,
        fetch: usize,
        weights: rro_core::HybridWeights,
        mode: rro_core::FusionMode,
    ) -> Result<Vec<Candidate>> {
        let dense = self.named_search(space, vector, fetch).await?;
        let lexical = if text.is_empty() {
            Vec::new()
        } else {
            self.lexical_search(text, fetch).await?
        };
        if lexical.is_empty() {
            let mut out = Vec::with_capacity(dense.len());
            for mut c in dense {
                if let Some(doc) = self.doc(c.id.as_str()).await? {
                    c.text = doc.text;
                    c.metadata = doc.metadata;
                }
                out.push(c);
            }
            return Ok(out);
        }
        let scored = [
            dense
                .iter()
                .map(|c| (c.id.as_str().to_string(), c.score))
                .collect::<Vec<_>>(),
            lexical
                .iter()
                .map(|c| (c.id.as_str().to_string(), c.score))
                .collect::<Vec<_>>(),
        ];
        let fused = crate::index::fuse(mode, &scored, &weights.as_slice(), FUSION_RRF_K);
        let mut out = Vec::with_capacity(fetch.min(fused.len()));
        for (id, score) in fused.into_iter().take(fetch) {
            if let Some(doc) = self.doc(&id).await? {
                let mut c = Candidate::new(doc.id, doc.text, score);
                c.metadata = doc.metadata;
                out.push(c);
            }
        }
        Ok(out)
    }
}

fn matches_filter(metadata: &Metadata, filter: &Metadata) -> bool {
    filter.iter().all(|(k, v)| metadata.get(k) == Some(v))
}

impl Estate {
    /// Facet: value → document count for a metadata field. **Index-first**
    /// when the field has a payload index and its rows carry decodable
    /// tags (string / bool): the rows sort by typed value, so
    /// counting distinct values is one run-length prefix scan with zero
    /// doc reads. Numeric/datetime/uuid/geo/other tags (original JSON
    /// spelling not reconstructible from the canonical key) and unindexed
    /// fields fall back to the exact doc scan.
    pub fn facet(&self, field: &str) -> Result<std::collections::BTreeMap<String, u64>> {
        if crate::filter::indexed_fields(&self.db)?
            .iter()
            .any(|f| f == field)
        {
            if let Some(counts) = self.facet_from_index(field)? {
                return Ok(counts);
            }
        }
        let handle = self.db.cf(CF_DOCS)?;
        let mut out = std::collections::BTreeMap::new();
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            let doc: StoredDoc = serde_json::from_slice(&v)?;
            if let Some(value) = doc.metadata.get(field) {
                let key = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                *out.entry(key).or_insert(0) += 1;
            }
        }
        Ok(out)
    }

    /// One prefix scan over `field`'s index rows, counting per distinct
    /// typed value. Returns `None` (fall back to the doc scan) on any tag
    /// whose surface form the key can't reconstruct.
    fn facet_from_index(
        &self,
        field: &str,
    ) -> Result<Option<std::collections::BTreeMap<String, u64>>> {
        let handle = self.db.cf(crate::keys::CF_PIDX)?;
        let prefix = crate::keys::pidx_field_prefix(field);
        let mut out = std::collections::BTreeMap::new();
        for item in self.db.0.iterator_cf(
            handle,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        ) {
            let (k, _) = item.map_err(rocks_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            let rest = &k[prefix.len()..];
            let Some((tag, payload)) = rest.split_first() else {
                continue;
            };
            // payload = typed-value bytes + SEP + doc_id; the value part
            // ends at the LAST separator (doc ids never contain NUL).
            let Some(sep_pos) = payload.iter().rposition(|&b| b == crate::keys::SEP) else {
                continue;
            };
            let value = &payload[..sep_pos];
            let surface = match *tag {
                crate::keys::PIDX_STR => String::from_utf8_lossy(value).into_owned(),
                crate::keys::PIDX_BOOL => (value == [1]).to_string(),
                // Numbers, datetimes, uuids, geo, other: the key holds a
                // canonical encoding, not the JSON source spelling ("2.0"
                // vs "2") — reconstructing would silently change facet
                // keys, so these fall back to the exact doc scan.
                _ => return Ok(None),
            };
            *out.entry(surface).or_insert(0) += 1;
        }
        Ok(Some(out))
    }

    /// The distinct values of a metadata field (facet keys) — index-first
    /// where the facet is.
    pub fn distinct(&self, field: &str) -> Result<Vec<String>> {
        Ok(self.facet(field)?.into_keys().collect())
    }

    /// Count documents matching a metadata equality filter (empty filter =
    /// exact total from the counter, free).
    pub fn count(&self, filter: &Metadata) -> Result<u64> {
        if filter.is_empty() {
            return self.db.get_u64(crate::keys::META_DOC_COUNT);
        }
        let handle = self.db.cf(CF_DOCS)?;
        let mut n = 0u64;
        for item in self.db.0.iterator_cf(handle, rocksdb::IteratorMode::Start) {
            let (_, v) = item.map_err(rocks_err)?;
            let doc: StoredDoc = serde_json::from_slice(&v)?;
            if matches_filter(&doc.metadata, filter) {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Scroll: cursor-paged document listing (id order). Pass the last id of
    /// the previous page as `after`; returns up to `limit` docs.
    pub fn scroll(&self, after: Option<&str>, limit: usize) -> Result<Vec<StoredDoc>> {
        let handle = self.db.cf(CF_DOCS)?;
        let mode = match after {
            Some(a) => rocksdb::IteratorMode::From(a.as_bytes(), rocksdb::Direction::Forward),
            None => rocksdb::IteratorMode::Start,
        };
        let mut out = Vec::new();
        for item in self.db.0.iterator_cf(handle, mode) {
            let (k, v) = item.map_err(rocks_err)?;
            if let Some(a) = after {
                if k.as_ref() == a.as_bytes() {
                    continue; // strictly after the cursor
                }
            }
            out.push(serde_json::from_slice(&v)?);
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }
}
