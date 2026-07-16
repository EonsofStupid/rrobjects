//! The typed query surface: one spec, every retrieval capability.
//!
//! [`EstateQuery`] is the builder consumers drive (and the seam a future
//! RRQL parser compiles into): hybrid text+vector search, metadata equality
//! **filters** (over-fetch + post-filter on the indexed paths, exact on
//! scoped paths), optional routed **scope**, and top-k. Facets, filtered
//! counts, and cursor-paged **scroll** live beside it on the estate.

use serde::{Deserialize, Serialize};

use rrf_core::{Candidate, Embedding, Metadata, Recall as _, Result};

use crate::estate::{rocks_err, Estate};
use crate::keys::CF_DOCS;
use crate::model::StoredDoc;
use crate::store::ConnXRecall;

/// How hard filtered searches over-fetch before post-filtering.
const FILTER_OVERFETCH: usize = 8;

/// A typed retrieval request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EstateQuery {
    /// Query text (drives the lexical half and, via the caller's embedder,
    /// usually the vector too).
    pub text: Option<String>,
    /// Dense query vector.
    pub vector: Option<Embedding>,
    /// Results wanted.
    pub top_k: usize,
    /// Metadata equality filter: every key must match exactly.
    pub filter: Metadata,
    /// Restrict to these ids (e.g. a routed neighborhood). Exact scoring.
    pub scope: Option<Vec<String>>,
}

impl EstateQuery {
    /// A hybrid query for the top `k`.
    pub fn hybrid(text: impl Into<String>, vector: Embedding, k: usize) -> Self {
        EstateQuery {
            text: Some(text.into()),
            vector: Some(vector),
            top_k: k,
            ..EstateQuery::default()
        }
    }

    /// Add a metadata equality condition.
    pub fn must(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.filter.insert(key.into(), value);
        self
    }

    /// Restrict to a routed scope.
    pub fn within(mut self, scope: Vec<String>) -> Self {
        self.scope = Some(scope);
        self
    }
}

fn matches_filter(metadata: &Metadata, filter: &Metadata) -> bool {
    filter.iter().all(|(k, v)| metadata.get(k) == Some(v))
}

impl ConnXRecall {
    /// Execute a typed query: scoped exact when a scope is given, otherwise
    /// hybrid (ANN + BM25, fused) with over-fetch + post-filter when a
    /// metadata filter is present.
    pub async fn query(&self, q: EstateQuery) -> Result<Vec<Candidate>> {
        let top_k = q.top_k;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let text = q.text.clone().unwrap_or_default();
        let vector = match &q.vector {
            Some(v) => v.clone(),
            None => Embedding(Vec::new()),
        };

        let fetch = if q.filter.is_empty() {
            top_k
        } else {
            top_k.saturating_mul(FILTER_OVERFETCH)
        };

        let mut results = match &q.scope {
            Some(scope) => {
                self.scoped_search(&text, &vector, fetch, scope.clone())
                    .await?
            }
            None if q.vector.is_some() => self.hybrid_search(&text, &vector, fetch).await?,
            None => self.lexical_search(&text, fetch).await?,
        };

        if !q.filter.is_empty() {
            // Lexical-path candidates may carry empty payloads; hydrate before
            // filtering so equality checks see real metadata.
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
            results.retain(|c| matches_filter(&c.metadata, &q.filter));
        }
        results.truncate(top_k);
        Ok(results)
    }
}

impl Estate {
    /// Facet: value → document count for a metadata field. v1 scans the doc
    /// column family (secondary indexes take over in the P3 tail — this is
    /// exact today, and honest about its cost).
    pub fn facet(&self, field: &str) -> Result<std::collections::BTreeMap<String, u64>> {
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
