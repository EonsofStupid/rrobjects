//! The persistent recall store: vectors + BM25 postings + payloads in one
//! estate, hybrid-searchable.
//!
//! [`ConnXRecall`] implements [`rro_core::Recall`]. `search` is dense cosine;
//! `hybrid_search` fuses dense and lexical rankings with reciprocal rank
//! fusion. All RocksDB work runs on the blocking pool so the tokio runtime
//! never stalls. Postings writes are blind puts (one row per (term, doc)),
//! but the estate counters (doc count, token totals, shape census) are
//! read-modify-write, so writers serialize behind an async mutex.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, RwLock as StdRwLock};

use async_trait::async_trait;
use recall::AnnIndex;
use rro_core::{Candidate, Embedding, Id, Recall, Result, RroError, VectorRecord};
use tokio::sync::Mutex;

use crate::estate::{rocks_err, Db, Estate};
use crate::index::{bm25_scores, reciprocal_rank_fusion, Bm25Params, Posting, Postings};
use crate::keys::{
    self, CF_DOCS, CF_FEED, CF_META, CF_PIDX, CF_TERMS, CF_VECS, META_DOC_COUNT, META_ESTATE,
    META_FEED_SEQ, META_SHAPES, META_TOTAL_TOKENS,
};
use crate::model::{EstateInfo, Shape, StoredDoc};

/// How much of each ranking feeds the fusion stage.
const FUSION_DEPTH_FACTOR: usize = 4;
/// The standard reciprocal-rank-fusion constant.
const RRF_K: f32 = 60.0;

/// Corpus size below which dense search scans exactly instead of using the
/// graph (tiny corpora: the scan is faster and exact).
const ANN_MIN_CORPUS: usize = 1024;

/// One write in a [`ConnXRecall::transaction`] batch — the ops that compose
/// atomically. Applying a sequence of these commits all of them or none: an
/// error on any op rolls the whole batch back, and nothing durable lands until
/// the last one succeeds.
pub enum WriteOp {
    /// Upsert a batch of records (same semantics as [`Recall::upsert`]).
    Upsert(Vec<VectorRecord>),
    /// Remove one document by id.
    Remove(Id),
}

/// Persistent, hybrid (dense + lexical) recall over an estate.
#[derive(Clone)]
pub struct ConnXRecall {
    pub(crate) db: Db,
    ann: Arc<StdRwLock<AnnIndex>>,
    pending: Arc<crate::pending::Pending>,
    feed_notify: Arc<tokio::sync::Notify>,
    quotas: Arc<crate::estate::Quotas>,
    lexical_stats: bool,
    analyzer: Arc<rro_core::text::Analyzer>,
    writer: Arc<Mutex<()>>,
    params: Bm25Params,
    /// How the graph quantizes its vectors. Lossy modes rescore graph hits
    /// exactly from the durable vectors (scores must never be approximate at the
    /// API surface without saying so), and BQ additionally over-fetches wider —
    /// its 1-bit codes surface fewer true neighbours per candidate.
    quantizer: recall::Quantizer,
}

impl Estate {
    /// The estate's recall store (shares this estate's database and graph).
    pub fn recall(&self) -> ConnXRecall {
        ConnXRecall {
            db: self.db.clone(),
            ann: self.ann.clone(),
            pending: self.pending.clone(),
            feed_notify: self.feed_notify.clone(),
            quotas: Arc::new(self.quotas.clone()),
            lexical_stats: self.lexical_stats,
            analyzer: Arc::new(self.info().analyzer.clone()),
            writer: Arc::new(Mutex::new(())),
            params: Bm25Params::default(),
            quantizer: self.quantizer,
        }
    }
}

/// Map candidates to `(id, score)` for the fusion functions.
fn scored_ids(cands: &[Candidate]) -> Vec<(String, f32)> {
    cands
        .iter()
        .map(|c| (c.id.as_str().to_string(), c.score))
        .collect()
}

impl ConnXRecall {
    /// Hybrid dense+lexical recall, fusing with explicit per-arm `weights`.
    ///
    /// This is the real implementation; the [`Recall::hybrid_search`] trait
    /// method delegates here with plain 1:1 weights. The split exists because
    /// the trait is a narrow port with no query object, while fusion is a
    /// per-query decision — see [`rro_core::EstateQuery::fusion`].
    pub async fn hybrid_weighted(
        &self,
        query_text: &str,
        query: &Embedding,
        top_k: usize,
        weights: rro_core::HybridWeights,
        mode: rro_core::FusionMode,
    ) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let ann = self.ann.clone();
        let pending = self.pending.clone();
        let params = self.params;
        let weights = weights.as_slice();
        let q = query.clone();
        let terms = self.analyzer.analyze(query_text);
        let depth = top_k.saturating_mul(FUSION_DEPTH_FACTOR).max(top_k);
        let quantizer = self.quantizer;

        tokio::task::spawn_blocking(move || {
            // Two rankings over the same estate…
            let dense = dense_blocking(&db, &ann, &pending, &q, depth, false, quantizer)?;
            let lexical = if terms.is_empty() {
                Vec::new()
            } else {
                lexical_topk_blocking(&db, params, &terms, depth)?
            };

            // …fused by the chosen strategy. Scored lists carry `(id, score)` so
            // DBSF can use the magnitudes; RRF ignores them and keeps the order.
            let scored = [scored_ids(&dense), scored_ids(&lexical)];
            let fused = crate::index::fuse(mode, &scored, &weights, RRF_K);

            let mut out = Vec::with_capacity(top_k);
            for (doc_id, score) in fused.into_iter().take(top_k) {
                if let Some(doc) = db.get_json::<StoredDoc>(CF_DOCS, doc_id.as_bytes())? {
                    let mut c = Candidate::new(doc.id, doc.text, score);
                    c.metadata = doc.metadata;
                    out.push(c);
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// A batch of write operations applied as one atomic transaction.
    pub async fn transaction(&self, ops: Vec<WriteOp>) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        // Serialize writers: the transaction threads the estate's counters, and
        // two concurrent transactions reading the same pre-commit counter would
        // both write a stale value. The whole transaction runs under this guard.
        let _guard = self.writer.lock().await;
        let db = self.db.clone();
        let pending = self.pending.clone();
        let analyzer = self.analyzer.clone();
        let lexical_stats = self.lexical_stats;
        tokio::task::spawn_blocking(move || {
            let mut tx = crate::txn::Transaction::begin(&db, &pending)?;
            for op in ops {
                match op {
                    // Any Err here returns before `commit`, so `tx` drops and the
                    // whole batch rolls back — nothing durable, graph untouched.
                    WriteOp::Upsert(records) => {
                        upsert_into(&mut tx, &db, &analyzer, records, lexical_stats)?
                    }
                    WriteOp::Remove(id) => {
                        remove_into(&mut tx, &db, &analyzer, id.as_str(), lexical_stats)?
                    }
                }
            }
            tx.commit()
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))??;
        self.feed_notify.notify_waiters();
        Ok(())
    }

    /// Fetch a stored document by id.
    pub async fn doc(&self, id: &str) -> Result<Option<StoredDoc>> {
        let db = self.db.clone();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes()))
            .await
            .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// Weighted sparse search: exact accumulated dot product between the
    /// sparse query and every document carrying any of its dimensions —
    /// sorted prefix scans per query dimension over the sparse postings.
    pub async fn sparse_search(
        &self,
        query: &rro_core::SparseVector,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        if query.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let q = query.clone();
        tokio::task::spawn_blocking(move || {
            let sparse_cf = db.cf(keys::CF_SPARSE)?;
            let mut scores: HashMap<String, f32> = HashMap::new();
            for (dim, qw) in q.iter() {
                let prefix = keys::sparse_prefix(dim);
                for item in db.0.iterator_cf(
                    sparse_cf,
                    rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
                ) {
                    let (k, v) = item.map_err(rocks_err)?;
                    if !k.starts_with(&prefix) {
                        break;
                    }
                    let doc_id = String::from_utf8_lossy(&k[prefix.len()..]).into_owned();
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&v[..4.min(v.len())]);
                    *scores.entry(doc_id).or_insert(0.0) += qw * f32::from_le_bytes(b);
                }
            }
            let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            ranked.truncate(top_k);
            Ok(ranked
                .into_iter()
                .map(|(id, score)| Candidate::new(id, String::new(), score))
                .collect())
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// The stored dense vector of one document, if present.
    pub async fn vector_of(&self, id: &str) -> Result<Option<Embedding>> {
        let db = self.db.clone();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || {
            let vecs_cf = db.cf(CF_VECS)?;
            Ok(db
                .0
                .get_cf(vecs_cf, id.as_bytes())
                .map_err(rocks_err)?
                .map(|b| Embedding(keys::decode_vec(&b))))
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// Search matrix: pairwise cosine similarity among `ids` (upper
    /// triangle, i < j, input order). Unknown ids are skipped.
    pub async fn similarity_matrix(&self, ids: &[String]) -> Result<Vec<(String, String, f32)>> {
        let db = self.db.clone();
        let ids = ids.to_vec();
        tokio::task::spawn_blocking(move || {
            let vecs_cf = db.cf(CF_VECS)?;
            let mut known: Vec<(String, Embedding)> = Vec::with_capacity(ids.len());
            for id in &ids {
                if let Some(b) = db.0.get_cf(vecs_cf, id.as_bytes()).map_err(rocks_err)? {
                    known.push((id.clone(), Embedding(keys::decode_vec(&b))));
                }
            }
            let mut out = Vec::with_capacity(known.len() * known.len().saturating_sub(1) / 2);
            for i in 0..known.len() {
                for j in (i + 1)..known.len() {
                    out.push((
                        known[i].0.clone(),
                        known[j].0.clone(),
                        known[i].1.cosine(&known[j].1),
                    ));
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// Exact cosine search inside one **named vector space** (a sorted
    /// prefix scan over that space's rows). Names are independent spaces
    /// with independent dimensionalities.
    pub async fn named_search(
        &self,
        space: &str,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let space = space.to_string();
        let q = query.clone();
        tokio::task::spawn_blocking(move || {
            let nvecs_cf = db.cf(keys::CF_NVECS)?;
            let prefix = keys::nvec_prefix(&space);
            let mut scored: Vec<(String, f32)> = Vec::new();
            for item in db.0.iterator_cf(
                nvecs_cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            ) {
                let (k, v) = item.map_err(rocks_err)?;
                if !k.starts_with(&prefix) {
                    break;
                }
                let doc_id = String::from_utf8_lossy(&k[prefix.len()..]).into_owned();
                let emb = Embedding(keys::decode_vec(&v));
                scored.push((doc_id, q.cosine(&emb)));
            }
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(top_k);
            Ok(scored
                .into_iter()
                .map(|(id, score)| Candidate::new(id, String::new(), score))
                .collect())
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// Rescore candidates by MaxSim (late interaction) against their stored
    /// token vectors. Docs with token vectors sort first by MaxSim; docs
    /// without any keep their relative first-phase order after them.
    pub async fn maxsim_rescore(
        &self,
        mut candidates: Vec<Candidate>,
        query_tokens: &[Embedding],
    ) -> Result<Vec<Candidate>> {
        if query_tokens.is_empty() || candidates.is_empty() {
            return Ok(candidates);
        }
        let db = self.db.clone();
        let ids: Vec<String> = candidates
            .iter()
            .map(|c| c.id.as_str().to_string())
            .collect();
        let q: Vec<Embedding> = query_tokens.to_vec();
        let scores: Vec<Option<f32>> = tokio::task::spawn_blocking(move || {
            let mvecs_cf = db.cf(keys::CF_MVECS)?;
            let mut out = Vec::with_capacity(ids.len());
            for id in &ids {
                let s = match db.0.get_cf(mvecs_cf, id.as_bytes()).map_err(rocks_err)? {
                    Some(bytes) => {
                        let doc_tokens: Vec<Embedding> = keys::decode_multi(&bytes)
                            .into_iter()
                            .map(Embedding)
                            .collect();
                        Some(rro_core::maxsim(&q, &doc_tokens))
                    }
                    None => None,
                };
                out.push(s);
            }
            Ok::<_, RroError>(out)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))??;

        for (c, s) in candidates.iter_mut().zip(&scores) {
            if let Some(s) = s {
                c.score = *s;
            }
        }
        // Stable partition: MaxSim-scored docs (sorted) first, the rest keep
        // their first-phase order.
        let mut with: Vec<Candidate> = Vec::new();
        let mut without: Vec<Candidate> = Vec::new();
        for (c, s) in candidates.into_iter().zip(&scores) {
            if s.is_some() {
                with.push(c);
            } else {
                without.push(c);
            }
        }
        with.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.id.as_str().cmp(b.id.as_str()))
        });
        with.extend(without);
        Ok(with)
    }

    /// Lexical (BM25) search over the persistent inverted index.
    pub async fn lexical_search(&self, query: &str, top_k: usize) -> Result<Vec<Candidate>> {
        let db = self.db.clone();
        let params = self.params;
        let terms = self.analyzer.analyze(query);
        if terms.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }
        tokio::task::spawn_blocking(move || lexical_topk_blocking(&db, params, &terms, top_k))
            .await
            .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// Hybrid recall **inside a scope** — the treasure half of the fusion
    /// law. The scope is a routed neighborhood (`Estate::traverse`); dense
    /// scoring is *exact* over it (point lookups, no ANN approximation) and
    /// lexical BM25 is filtered to it, fused as usual. Ids in the scope that
    /// aren't documents are ignored.
    /// Filter-aware dense search: nearest `top_k` among `allow`, via graph
    /// traversal that admits only allowed nodes (see `AnnIndex::search_filtered`).
    ///
    /// This is the correct answer for a matched set too large to score exactly:
    /// it walks the HNSW graph but only ever collects nodes the filter accepts,
    /// so the result is the *filtered* nearest neighbours rather than the global
    /// ones that happen to survive a post-filter. When text is present a scoped
    /// BM25 ranking over the same allowed set fuses in, exactly like the hybrid
    /// path. Below the ANN minimum corpus the graph is not built, so this falls back
    /// to exact scoped scoring — which is also correct, just O(matches).
    pub async fn filter_aware_search(
        &self,
        query_text: &str,
        query: &Embedding,
        top_k: usize,
        allow: &std::collections::HashSet<String>,
        weights: rro_core::HybridWeights,
        mode: rro_core::FusionMode,
    ) -> Result<Vec<Candidate>> {
        if top_k == 0 || allow.is_empty() {
            return Ok(Vec::new());
        }
        // Small corpus: no graph. Exact scoping is correct and cheap.
        let has_graph = { self.ann.read().expect("ann lock").len() >= ANN_MIN_CORPUS };
        if !has_graph {
            return self
                .scoped_search(query_text, query, top_k, allow.iter().cloned().collect())
                .await;
        }

        let ann = self.ann.clone();
        let db = self.db.clone();
        let params = self.params;
        let q = query.clone();
        let terms = self.analyzer.analyze(query_text);
        let allow_ids: std::collections::HashSet<rro_core::Id> =
            allow.iter().map(|s| rro_core::Id(s.clone())).collect();
        let allow_str = allow.clone();
        let weights = weights.as_slice();

        tokio::task::spawn_blocking(move || {
            // Filter-aware dense ranking off the graph — keep the scores for DBSF.
            let dense: Vec<(String, f32)> = {
                let graph = ann.read().expect("ann lock");
                graph
                    .search_filtered(&q, top_k, top_k.max(64), &allow_ids)
                    .into_iter()
                    .map(|(id, s)| (id.as_str().to_string(), s))
                    .collect()
            };

            // Lexical ranking, restricted to the allowed set.
            let lexical: Vec<(String, f32)> = if terms.is_empty() {
                Vec::new()
            } else {
                let mut lex = lexical_blocking(&db, params, &terms, top_k * 4)?;
                lex.retain(|c| allow_str.contains(c.id.as_str()));
                lex.into_iter()
                    .map(|c| (c.id.as_str().to_string(), c.score))
                    .collect()
            };

            let fused: Vec<String> = if lexical.is_empty() {
                dense.into_iter().map(|(id, _)| id).collect()
            } else {
                let scored = [dense, lexical];
                crate::index::fuse(mode, &scored, &weights, RRF_K)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            };

            // Hydrate winners exactly from the durable vectors + doc store.
            let vecs_cf = db.cf(CF_VECS)?;
            let mut out = Vec::with_capacity(top_k.min(fused.len()));
            for id in fused.into_iter().take(top_k) {
                let score = match db.0.get_cf(vecs_cf, id.as_bytes()).map_err(rocks_err)? {
                    Some(bytes) => q.cosine(&Embedding(keys::decode_vec(&bytes))),
                    None => continue,
                };
                let mut c = Candidate::new(id.clone(), String::new(), score);
                if let Some(doc) = db.get_json::<crate::model::StoredDoc>(CF_DOCS, id.as_bytes())? {
                    c.text = doc.text;
                    c.metadata = doc.metadata;
                }
                out.push(c);
            }
            out.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.id.as_str().cmp(b.id.as_str()))
            });
            Ok(out)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    /// Exact hybrid recall restricted to `scope`: exact cosine over the scope by
    /// point-lookup, fused with a scoped BM25 ranking. Correct at any size; the
    /// cost is O(scope).
    pub async fn scoped_search(
        &self,
        query_text: &str,
        query: &Embedding,
        top_k: usize,
        scope: Vec<String>,
    ) -> Result<Vec<Candidate>> {
        if top_k == 0 || scope.is_empty() {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let params = self.params;
        let q = query.clone();
        let terms = self.analyzer.analyze(query_text);

        tokio::task::spawn_blocking(move || {
            use std::collections::HashSet;
            let in_scope: HashSet<&str> = scope.iter().map(String::as_str).collect();

            // Dense: exact cosine over the scope by point lookup.
            let vecs_cf = db.cf(CF_VECS)?;
            let mut dense: Vec<(String, f32)> = Vec::new();
            for id in &scope {
                if let Some(bytes) = db.0.get_cf(vecs_cf, id.as_bytes()).map_err(rocks_err)? {
                    let emb = Embedding(keys::decode_vec(&bytes));
                    dense.push((id.clone(), q.cosine(&emb)));
                }
            }
            dense.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            // Lexical: BM25, filtered to the scope.
            let mut lexical = if terms.is_empty() {
                Vec::new()
            } else {
                lexical_blocking(&db, params, &terms, usize::MAX)?
            };
            lexical.retain(|c| in_scope.contains(c.id.as_str()));

            // Fuse and fetch winners' payloads.
            let lists = [
                dense.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                lexical
                    .iter()
                    .map(|c| c.id.as_str().to_string())
                    .collect::<Vec<_>>(),
            ];
            let fused = reciprocal_rank_fusion(&lists, RRF_K);

            let mut out = Vec::with_capacity(top_k);
            for (doc_id, score) in fused.into_iter().take(top_k) {
                if let Some(doc) = db.get_json::<StoredDoc>(CF_DOCS, doc_id.as_bytes())? {
                    let mut c = Candidate::new(doc.id, doc.text, score);
                    c.metadata = doc.metadata;
                    out.push(c);
                }
            }
            Ok(out)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }
}

#[async_trait]
impl Recall for ConnXRecall {
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        // Quotas first: batch size and per-doc payload bytes reject at
        // the boundary, before any write.
        if let Some(cap) = self.quotas.max_batch {
            if records.len() > cap {
                return Err(RroError::Quota(format!(
                    "batch of {} exceeds max_batch {cap}",
                    records.len()
                )));
            }
        }
        if let Some(cap) = self.quotas.max_payload_bytes {
            for r in &records {
                let bytes = serde_json::to_vec(&r.metadata)?.len();
                if bytes > cap {
                    return Err(RroError::Quota(format!(
                        "payload of {bytes} bytes on `{}` exceeds max_payload_bytes {cap}",
                        r.id.as_str()
                    )));
                }
            }
        }
        // Serialize writers: counters/census are read-modify-write.
        let _guard = self.writer.lock().await;
        let db = self.db.clone();
        let pending = self.pending.clone();
        let analyzer = self.analyzer.clone();
        let max_docs = self.quotas.max_docs;
        let lexical_stats = self.lexical_stats;
        tokio::task::spawn_blocking(move || {
            // Doc cap: net-new docs counted inside the serialized writer,
            // so the check is race-free.
            if let Some(cap) = max_docs {
                let existing = db.get_u64(META_DOC_COUNT)?;
                let net_new = records
                    .iter()
                    .filter(|r| {
                        db.get_json::<StoredDoc>(CF_DOCS, r.id.as_str().as_bytes())
                            .map(|d| d.is_none())
                            .unwrap_or(true)
                    })
                    .count() as u64;
                if existing + net_new > cap {
                    return Err(RroError::Quota(format!(
                        "{existing} docs + {net_new} new exceeds max_docs {cap}"
                    )));
                }
            }
            // One implicit single-statement transaction: the durable batch and
            // the counter deltas land atomically, and the graph ops enqueue for
            // the out-of-band applier only after that batch commits. This is the
            // exact same path an explicit `BEGIN … COMMIT` takes with more
            // statements — there is no second write implementation.
            let mut tx = crate::txn::Transaction::begin(&db, &pending)?;
            upsert_into(&mut tx, &db, &analyzer, records, lexical_stats)?;
            tx.commit()
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))??;
        // Wake push-stream watchers: the feed rows are committed.
        self.feed_notify.notify_waiters();
        Ok(())
    }

    async fn search(&self, query: &Embedding, top_k: usize) -> Result<Vec<Candidate>> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.clone();
        let ann = self.ann.clone();
        let pending = self.pending.clone();
        let q = query.clone();
        let quantizer = self.quantizer;
        tokio::task::spawn_blocking(move || {
            dense_blocking(&db, &ann, &pending, &q, top_k, true, quantizer)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    async fn hybrid_search(
        &self,
        query_text: &str,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        // The port carries no query, so it fuses 1:1 (plain RRF). Callers that
        // have an EstateQuery go through `hybrid_weighted` and honour its
        // `fusion` weights.
        self.hybrid_weighted(
            query_text,
            query,
            top_k,
            rro_core::HybridWeights::default(),
            rro_core::FusionMode::Rrf,
        )
        .await
    }

    async fn len(&self) -> Result<usize> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || db.get_u64(META_DOC_COUNT).map(|n| n as usize))
            .await
            .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }

    async fn remove(&self, id: &Id) -> Result<()> {
        let _guard = self.writer.lock().await;
        let db = self.db.clone();
        let pending = self.pending.clone();
        let analyzer = self.analyzer.clone();
        let lexical_stats = self.lexical_stats;
        let id = id.clone();
        tokio::task::spawn_blocking(move || {
            // One implicit single-statement transaction: durable deletes commit
            // atomically, then the tombstone enqueues for the applier.
            let mut tx = crate::txn::Transaction::begin(&db, &pending)?;
            remove_into(&mut tx, &db, &analyzer, id.as_str(), lexical_stats)?;
            tx.commit()
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))??;
        self.feed_notify.notify_waiters();
        Ok(())
    }

    async fn quiesce(&self) -> Result<()> {
        let pending = self.pending.clone();
        tokio::task::spawn_blocking(move || {
            pending.quiesce();
            Ok(())
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))?
    }
}

// ---- blocking internals (run on the blocking pool) ----------------------------

/// Apply an upsert into a transaction: durable index writes into `tx.batch`,
/// counter deltas onto `tx`'s threaded counters, and one deferred graph op per
/// record. The transaction commits (or cancels) them as a unit.
///
/// `db` is passed alongside `tx` purely so column-family handles can be resolved
/// without holding an immutable borrow of `tx` across its own mutation — both
/// reference the same estate, and `&Db` is shared, so this aliasing is sound.
///
/// Schema registration (the estate's dimension, named-space dims, the collection
/// registry) is written immediately, not through the batch, and so is **not**
/// rolled back by a CANCEL. This is deliberate and safe: those are monotonic
/// declarations, and a canceled registration leaves at most an empty collection
/// name or a learned dimension with no rows — which every index reads as "no
/// members", i.e. consistent. The *data* indexes (postings, payload, sparse,
/// named/multi vectors, collection membership, doc/token counts, changefeed) all
/// live in `tx.batch` and roll back exactly.
fn upsert_into(
    tx: &mut crate::txn::Transaction,
    db: &Db,
    analyzer: &rro_core::text::Analyzer,
    records: Vec<VectorRecord>,
    lexical_stats: bool,
) -> Result<()> {
    // Dimension guard: fixed by the first upsert, enforced forever after.
    let mut info: EstateInfo = db
        .get_json(CF_META, META_ESTATE)?
        .ok_or_else(|| RroError::Recall("estate not initialized".into()))?;
    let dim = records[0].embedding.dim();
    match info.dim {
        None => {
            info.dim = Some(dim);
            db.put_json(CF_META, META_ESTATE, &info)?;
        }
        Some(expected) if expected != dim => {
            return Err(RroError::DimMismatch { expected, got: dim });
        }
        _ => {}
    }
    for r in &records {
        if r.embedding.dim() != dim {
            return Err(RroError::DimMismatch {
                expected: dim,
                got: r.embedding.dim(),
            });
        }
    }

    tx.touch_counters();
    // Postings are one row per (term, doc): every index write below is a
    // blind put/delete — no read-modify-write, flat cost as terms grow.
    let docs_cf = db.cf(CF_DOCS)?;
    let vecs_cf = db.cf(CF_VECS)?;
    let terms_cf = db.cf(CF_TERMS)?;
    let feed_cf = db.cf(CF_FEED)?;
    let pidx_cf = db.cf(CF_PIDX)?;
    let sparse_cf = db.cf(keys::CF_SPARSE)?;
    let nvecs_cf = db.cf(keys::CF_NVECS)?;
    let mvecs_cf = db.cf(keys::CF_MVECS)?;
    let coll_cf = db.cf(keys::CF_COLL)?;
    let tdf_cf = db.cf(keys::CF_TDF)?;
    let indexed_fields = crate::filter::indexed_fields(db)?;
    // df deltas are NETTED per batch: one merge operand per term per
    // WriteBatch instead of one per (term, doc). Same atomicity, same
    // counts — but hot terms stop accumulating thousands of merge
    // operands per flush (found as a −56% ingest regression by the
    // Sprint-28 gate).
    let mut df_delta: HashMap<String, i64> = HashMap::new();

    // Auto-register any new collection names (writers already serialize).
    let mut registry: Vec<String> = db
        .get_json(CF_META, keys::META_COLLECTIONS)?
        .unwrap_or_default();
    let mut registry_dirty = false;
    for r in &records {
        if let Some(c) = &r.collection {
            if !registry.iter().any(|x| x == c) {
                registry.push(c.clone());
                registry_dirty = true;
            }
        }
    }
    if registry_dirty {
        registry.sort();
        db.put_json(CF_META, keys::META_COLLECTIONS, &registry)?;
    }

    // Named spaces: each name's dimensionality is fixed by its first vector.
    let mut named_dims_dirty = false;
    for r in &records {
        for (name, v) in &r.named {
            match info.named_dims.get(name) {
                None => {
                    info.named_dims.insert(name.clone(), v.dim());
                    named_dims_dirty = true;
                }
                Some(&expected) if expected != v.dim() => {
                    return Err(RroError::DimMismatch {
                        expected,
                        got: v.dim(),
                    });
                }
                _ => {}
            }
        }
        // Late-interaction token vectors must agree among themselves.
        if let Some(first) = r.multi.first() {
            if r.multi.iter().any(|t| t.dim() != first.dim()) {
                return Err(RroError::Recall(
                    "multi-vector token dims disagree within one record".into(),
                ));
            }
        }
    }
    if named_dims_dirty {
        db.put_json(CF_META, META_ESTATE, &info)?;
    }

    for r in records {
        let id = r.id.as_str().to_string();

        // Overwrite semantics: retract the old version's postings (lexical
        // and sparse), payload index rows, and counters.
        if let Some(old) = db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes())? {
            let mut seen = std::collections::HashSet::new();
            for term in analyzer.analyze(&old.text) {
                tx.batch.delete_cf(terms_cf, keys::term_key(&term, &id));
                if seen.insert(term.clone()) {
                    *df_delta.entry(term).or_insert(0) -= 1;
                }
            }
            for dim in &old.sparse_dims {
                tx.batch.delete_cf(sparse_cf, keys::sparse_key(*dim, &id));
            }
            for space in &old.named_spaces {
                tx.batch.delete_cf(nvecs_cf, keys::nvec_key(space, &id));
            }
            if old.multi_len > 0 {
                tx.batch.delete_cf(mvecs_cf, id.as_bytes());
            }
            if let Some(c) = &old.collection {
                tx.batch.delete_cf(coll_cf, keys::coll_key(c, &id));
            }
            for field in &indexed_fields {
                if let Some(v) = old.metadata.get(field) {
                    tx.batch.delete_cf(pidx_cf, keys::pidx_key(field, v, &id));
                }
            }
            tx.total_tokens = tx.total_tokens.saturating_sub(old.token_len as u64);
            if let Some(n) = tx.shapes.get_mut(&old.shape.key()) {
                *n = n.saturating_sub(1);
            }
            tx.doc_count = tx.doc_count.saturating_sub(1);
        }

        let tokens = analyzer.analyze(&r.text);
        let token_len = tokens.len() as u32;
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in tokens {
            *tf.entry(t).or_insert(0) += 1;
        }
        for (term, f) in tf {
            // Binary posting value: [tf u32 LE][len u32 LE].
            let mut v = [0u8; 8];
            v[..4].copy_from_slice(&f.to_le_bytes());
            v[4..].copy_from_slice(&token_len.to_le_bytes());
            tx.batch.put_cf(terms_cf, keys::term_key(&term, &id), v);
            *df_delta.entry(term).or_insert(0) += 1;
        }

        let shape = Shape::of(&r.metadata);
        *tx.shapes.entry(shape.key()).or_insert(0) += 1;
        tx.doc_count += 1;
        tx.total_tokens += token_len as u64;

        // Payload index rows for indexed fields — blind puts, same tx.batch.
        for field in &indexed_fields {
            if let Some(v) = r.metadata.get(field) {
                tx.batch.put_cf(pidx_cf, keys::pidx_key(field, v, &id), []);
            }
        }

        // Weighted sparse postings — one row per (dim, doc), blind puts.
        let mut sparse_dims = Vec::new();
        if let Some(sv) = &r.sparse {
            sparse_dims.reserve(sv.nnz());
            for (dim, w) in sv.iter() {
                tx.batch
                    .put_cf(sparse_cf, keys::sparse_key(dim, &id), w.to_le_bytes());
                sparse_dims.push(dim);
            }
        }

        // Named vectors: one row per (space, doc) — blind puts.
        let mut named_spaces = Vec::with_capacity(r.named.len());
        for (name, v) in &r.named {
            tx.batch.put_cf(
                nvecs_cf,
                keys::nvec_key(name, &id),
                keys::encode_vec(v.as_slice()),
            );
            named_spaces.push(name.clone());
        }

        // Late-interaction token vectors: one row per doc.
        let multi_len = r.multi.len() as u32;
        if multi_len > 0 {
            let raw: Vec<Vec<f32>> = r.multi.iter().map(|e| e.0.clone()).collect();
            tx.batch
                .put_cf(mvecs_cf, id.as_bytes(), keys::encode_multi(&raw));
        }

        if let Some(c) = &r.collection {
            tx.batch.put_cf(coll_cf, keys::coll_key(c, &id), []);
        }

        let doc = StoredDoc {
            id: id.clone(),
            text: r.text,
            metadata: r.metadata,
            tags: Vec::new(),
            shape,
            token_len,
            connector_id: None,
            sparse_dims,
            named_spaces,
            multi_len,
            collection: r.collection,
        };
        tx.batch
            .put_cf(docs_cf, id.as_bytes(), serde_json::to_vec(&doc)?);
        tx.batch.put_cf(
            vecs_cf,
            id.as_bytes(),
            keys::encode_vec(r.embedding.as_slice()),
        );
        // Deferred: the vector enters the ANN graph only when the tx commits.
        tx.push_graph(crate::txn::GraphOp::Upsert(
            rro_core::Id(id.clone()),
            r.embedding.clone(),
        ));

        // Changefeed row, atomic with the write itself.
        let change = crate::model::Change {
            seq: tx.feed_seq,
            op: crate::model::ChangeOp::Upsert,
            doc_id: id.clone(),
            at: crate::model::now_ms(),
        };
        tx.batch.put_cf(
            feed_cf,
            tx.feed_seq.to_be_bytes(),
            serde_json::to_vec(&change)?,
        );
        tx.feed_seq += 1;
    }

    if lexical_stats {
        for (term, delta) in df_delta {
            if delta != 0 {
                tx.batch
                    .merge_cf(tdf_cf, term.as_bytes(), delta.to_le_bytes());
            }
        }
    }

    Ok(())
}

/// Decode a posting value: 8-byte binary (tf, len — the current format)
/// or the JSON rows estates wrote before the binary format existed.
fn decode_posting(v: &[u8]) -> Result<Posting> {
    if v.len() == 8 {
        Ok(Posting {
            tf: u32::from_le_bytes(v[..4].try_into().expect("4 bytes")),
            len: u32::from_le_bytes(v[4..].try_into().expect("4 bytes")),
        })
    } else {
        Ok(serde_json::from_slice(v)?)
    }
}

/// Prefix-scan a term's postings rows.
fn scan_postings(db: &Db, term: &str) -> Result<Postings> {
    let terms_cf = db.cf(CF_TERMS)?;
    let prefix = keys::term_prefix(term);
    let mut out = Postings::new();
    for item in db.0.iterator_cf(
        terms_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    ) {
        let (k, v) = item.map_err(rocks_err)?;
        if !k.starts_with(&prefix) {
            break;
        }
        let doc_id = String::from_utf8_lossy(&k[prefix.len()..]).into_owned();
        out.push((doc_id, decode_posting(&v)?));
    }
    Ok(out)
}

/// Dense search: the ANN graph above [`ANN_MIN_CORPUS`], exact scan below.
/// Graph results are merged with the **pending overlay** (not-yet-applied
/// upserts scored exactly, pending removals masked) so a committed write is
/// searchable before the applier reaches it. When `fetch_payload` is false,
/// candidates carry ids and scores only (fusion fetches winners' payloads).
fn dense_blocking(
    db: &Db,
    ann: &Arc<StdRwLock<AnnIndex>>,
    pending: &Arc<crate::pending::Pending>,
    query: &Embedding,
    top_k: usize,
    fetch_payload: bool,
    quantizer: recall::Quantizer,
) -> Result<Vec<Candidate>> {
    use recall::Quantizer;
    // Lossy graphs return approximate scores; over-fetch, then rescore the
    // candidates exactly from the durable vectors before cutting to k. BQ is
    // far coarser than SQ8 (1 bit vs 1 byte per dim), so it over-fetches wider
    // and widens `ef` too — otherwise its top-k misses too many true neighbours
    // for rescore to recover them. (Measured: at `×2`/`ef 64` BQ recall@10 is
    // ~0.64; at `×8`/`ef 200` it clears the estate gate.)
    let rescore = quantizer.is_lossy();
    let (factor, ef_floor) = match quantizer {
        Quantizer::None => (1, 64),
        Quantizer::Sq8 => (2, 64),
        Quantizer::Bq => (8, 200),
    };
    let fetch = top_k.saturating_mul(factor);
    let mut scored: Vec<(String, f32)>;
    {
        let graph = ann.read().expect("ann lock");
        if graph.len() >= ANN_MIN_CORPUS {
            scored = graph
                .search(query, fetch, fetch.max(ef_floor))
                .into_iter()
                .map(|(id, score)| (id.as_str().to_string(), score))
                .collect();

            // Overlay: pending wins by id; removals mask stale graph hits.
            let (ups, dels) = pending.overlay(query);
            if !ups.is_empty() || !dels.is_empty() {
                use std::collections::HashSet;
                let masked: HashSet<&str> = dels.iter().map(|d| d.as_str()).collect();
                let overlaid: HashSet<&str> = ups.iter().map(|(id, _)| id.as_str()).collect();
                scored.retain(|(id, _)| {
                    !masked.contains(id.as_str()) && !overlaid.contains(id.as_str())
                });
                scored.extend(ups.into_iter().map(|(id, s)| (id.as_str().to_string(), s)));
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(fetch);
            }

            if rescore {
                let vecs_cf = db.cf(CF_VECS)?;
                for (id, s) in scored.iter_mut() {
                    if let Some(bytes) = db.0.get_cf(vecs_cf, id.as_bytes()).map_err(rocks_err)? {
                        *s = query.cosine(&Embedding(keys::decode_vec(&bytes)));
                    }
                }
                scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(top_k);
            }
        } else {
            // Tiny corpus: exact scan of the durable vectors — already
            // current (writes land in `vecs` before enqueueing), no overlay.
            let vecs_cf = db.cf(CF_VECS)?;
            scored = Vec::new();
            for item in db.0.iterator_cf(vecs_cf, rocksdb::IteratorMode::Start) {
                let (k, v) = item.map_err(rocks_err)?;
                let emb = Embedding(keys::decode_vec(&v));
                scored.push((String::from_utf8_lossy(&k).into_owned(), query.cosine(&emb)));
            }
            scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(top_k);
        }
    }

    let mut out = Vec::with_capacity(scored.len());
    for (id, score) in scored {
        if fetch_payload {
            if let Some(doc) = db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes())? {
                let mut c = Candidate::new(doc.id, doc.text, score);
                c.metadata = doc.metadata;
                out.push(c);
                continue;
            }
        }
        out.push(Candidate::new(id, String::new(), score));
    }
    Ok(out)
}

/// One term's BM25 contribution for a posting.
#[inline(always)]
fn bm25_term(params: Bm25Params, idf: f32, p: Posting, avgdl: f32) -> f32 {
    let f = p.tf as f32;
    let dl = p.len as f32;
    idf * (f * (params.k1 + 1.0)) / (f + params.k1 * (1.0 - params.b + params.b * dl / avgdl))
}

/// Max-score lexical top-k (authored from the Turtle–Flood concept):
/// document frequencies come from the blind `tdf` counters, giving each
/// term an idf and a per-doc score upper bound BEFORE any scan. Terms are
/// processed in descending upper bound; once the current k-th accumulator
/// exceeds the summed bounds of the unprocessed terms, no unseen document
/// can reach the top-k — the remaining (typically most common) terms are
/// resolved by POINT LOOKUPS for the accumulated candidates instead of
/// full postings scans. Exact by construction: accumulators only grow,
/// and every candidate's final score is completed before ranking.
/// Falls back to the full scorer on estates without df stats.
fn lexical_topk_blocking(
    db: &Db,
    params: Bm25Params,
    terms: &[String],
    top_k: usize,
) -> Result<Vec<Candidate>> {
    let meta_cf = db.cf(CF_META)?;
    if db
        .0
        .get_cf(meta_cf, keys::META_LEXSTATS)
        .map_err(rocks_err)?
        .is_none()
    {
        return lexical_blocking(db, params, terms, top_k);
    }
    let n_docs = db.get_u64(META_DOC_COUNT)?;
    let total_tokens = db.get_u64(META_TOTAL_TOKENS)?;
    if n_docs == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    let n = n_docs as f32;
    let avgdl = (total_tokens as f32 / n).max(1.0);

    // Term stats: df from the merged counters → idf → upper bound.
    let tdf_cf = db.cf(keys::CF_TDF)?;
    let mut infos: Vec<(String, f32, f32)> = Vec::new(); // (term, idf, ub)
    let mut seen = std::collections::HashSet::new();
    for t in terms {
        if !seen.insert(t.clone()) {
            continue;
        }
        let df =
            db.0.get_cf(tdf_cf, t.as_bytes())
                .map_err(rocks_err)?
                .map(|b| {
                    let mut a = [0u8; 8];
                    a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
                    i64::from_le_bytes(a)
                })
                .unwrap_or(0);
        if df <= 0 {
            continue; // term absent from the corpus
        }
        let idf = (((n - df as f32 + 0.5) / (df as f32 + 0.5)) + 1.0)
            .ln()
            .max(0.0);
        infos.push((t.clone(), idf, idf * (params.k1 + 1.0)));
    }
    if infos.is_empty() {
        return Ok(Vec::new());
    }
    // Highest upper bound first: rare, informative terms scan first.
    infos.sort_by(|a, b| b.2.total_cmp(&a.2));

    let terms_cf = db.cf(CF_TERMS)?;
    let mut acc: HashMap<String, f32> = HashMap::new();
    let mut kth_floor = 0.0f32;

    let mut idx = 0usize;
    while idx < infos.len() {
        let remaining_ub: f32 = infos[idx..].iter().map(|x| x.2).sum();
        if acc.len() >= top_k && kth_floor > remaining_ub {
            break; // no unseen doc can reach the top-k
        }
        let (term, idf, _) = &infos[idx];
        let prefix = keys::term_prefix(term);
        for item in db.0.iterator_cf(
            terms_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        ) {
            let (k, v) = item.map_err(rocks_err)?;
            if !k.starts_with(&prefix) {
                break;
            }
            let doc_id = String::from_utf8_lossy(&k[prefix.len()..]).into_owned();
            *acc.entry(doc_id).or_insert(0.0) +=
                bm25_term(params, *idf, decode_posting(&v)?, avgdl);
        }
        idx += 1;
        // Refresh the k-th floor (a valid lower bound: accumulators only grow).
        if acc.len() >= top_k {
            let mut vals: Vec<f32> = acc.values().copied().collect();
            let kidx = top_k - 1;
            vals.select_nth_unstable_by(kidx, |a, b| b.total_cmp(a));
            kth_floor = vals[kidx];
        }
    }

    // Pruned tail: complete every candidate's score by point lookups.
    for (term, idf, _) in &infos[idx..] {
        for (doc_id, score) in acc.iter_mut() {
            if let Some(v) =
                db.0.get_cf(terms_cf, keys::term_key(term, doc_id))
                    .map_err(rocks_err)?
            {
                *score += bm25_term(params, *idf, decode_posting(&v)?, avgdl);
            }
        }
    }

    let mut ranked: Vec<(String, f32)> = acc.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_k);
    Ok(ranked
        .into_iter()
        .map(|(id, score)| Candidate::new(id, String::new(), score))
        .collect())
}

fn lexical_blocking(
    db: &Db,
    params: Bm25Params,
    terms: &[String],
    top_k: usize,
) -> Result<Vec<Candidate>> {
    let n_docs = db.get_u64(META_DOC_COUNT)?;
    let total_tokens = db.get_u64(META_TOTAL_TOKENS)?;
    let avgdl = if n_docs == 0 {
        1.0
    } else {
        total_tokens as f32 / n_docs as f32
    };

    let mut term_postings: Vec<(String, Postings)> = Vec::with_capacity(terms.len());
    for t in terms {
        term_postings.push((t.clone(), scan_postings(db, t)?));
    }

    let scores = bm25_scores(params, n_docs, avgdl, &term_postings);
    let mut ranked: Vec<(String, f32)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(top_k);

    let mut out = Vec::with_capacity(ranked.len());
    for (id, score) in ranked {
        out.push(Candidate::new(id, String::new(), score));
    }
    Ok(out)
}

/// Apply a removal into a transaction (see [`upsert_into`] for the shape). The
/// deferred graph op is a tombstone, applied to the ANN index at commit.
fn remove_into(
    tx: &mut crate::txn::Transaction,
    db: &Db,
    analyzer: &rro_core::text::Analyzer,
    id: &str,
    lexical_stats: bool,
) -> Result<()> {
    let Some(old) = db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes())? else {
        return Ok(());
    };
    tx.touch_counters();
    let terms_cf = db.cf(CF_TERMS)?;
    let tdf_cf = db.cf(keys::CF_TDF)?;
    let mut seen = std::collections::HashSet::new();
    for term in analyzer.analyze(&old.text) {
        tx.batch.delete_cf(terms_cf, keys::term_key(&term, id));
        if lexical_stats && seen.insert(term.clone()) {
            tx.batch
                .merge_cf(tdf_cf, term.as_bytes(), (-1i64).to_le_bytes());
        }
    }
    let pidx_cf = db.cf(CF_PIDX)?;
    for field in crate::filter::indexed_fields(db)? {
        if let Some(v) = old.metadata.get(&field) {
            tx.batch.delete_cf(pidx_cf, keys::pidx_key(&field, v, id));
        }
    }
    let sparse_cf = db.cf(keys::CF_SPARSE)?;
    for dim in &old.sparse_dims {
        tx.batch.delete_cf(sparse_cf, keys::sparse_key(*dim, id));
    }
    let nvecs_cf = db.cf(keys::CF_NVECS)?;
    for space in &old.named_spaces {
        tx.batch.delete_cf(nvecs_cf, keys::nvec_key(space, id));
    }
    if old.multi_len > 0 {
        tx.batch.delete_cf(db.cf(keys::CF_MVECS)?, id.as_bytes());
    }
    if let Some(c) = &old.collection {
        tx.batch
            .delete_cf(db.cf(keys::CF_COLL)?, keys::coll_key(c, id));
    }
    tx.batch.delete_cf(db.cf(CF_DOCS)?, id.as_bytes());
    tx.batch.delete_cf(db.cf(CF_VECS)?, id.as_bytes());

    // Changefeed row, atomic with the removal.
    let change = crate::model::Change {
        seq: tx.feed_seq,
        op: crate::model::ChangeOp::Remove,
        doc_id: id.to_string(),
        at: crate::model::now_ms(),
    };
    tx.batch.put_cf(
        db.cf(CF_FEED)?,
        tx.feed_seq.to_be_bytes(),
        serde_json::to_vec(&change)?,
    );
    tx.feed_seq += 1;

    tx.doc_count = tx.doc_count.saturating_sub(1);
    tx.total_tokens = tx.total_tokens.saturating_sub(old.token_len as u64);
    if let Some(n) = tx.shapes.get_mut(&old.shape.key()) {
        *n = n.saturating_sub(1);
    }
    tx.push_graph(crate::txn::GraphOp::Remove(rro_core::Id(id.to_string())));
    Ok(())
}

/// One-statement removal (the estate's direct-delete path uses this).
pub(crate) fn remove_blocking(
    db: &Db,
    pending: &crate::pending::Pending,
    analyzer: &rro_core::text::Analyzer,
    id: &str,
    lexical_stats: bool,
) -> Result<()> {
    let mut tx = crate::txn::Transaction::begin(db, pending)?;
    remove_into(&mut tx, db, analyzer, id, lexical_stats)?;
    tx.commit()
}

impl ConnXRecall {
    /// Merge keys into a document's metadata (existing keys overwrite).
    pub async fn set_payload(&self, id: &str, patch: rro_core::Metadata) -> Result<()> {
        self.mutate_payload(id, move |m| {
            for (k, v) in patch {
                m.insert(k, v);
            }
        })
        .await
    }

    /// Replace a document's metadata entirely.
    pub async fn overwrite_payload(&self, id: &str, meta: rro_core::Metadata) -> Result<()> {
        self.mutate_payload(id, move |m| *m = meta).await
    }

    /// Remove specific keys from a document's metadata.
    pub async fn delete_payload_keys(&self, id: &str, keys: Vec<String>) -> Result<()> {
        self.mutate_payload(id, move |m| {
            for k in &keys {
                m.remove(k);
            }
        })
        .await
    }

    /// Clear a document's metadata entirely.
    pub async fn clear_payload(&self, id: &str) -> Result<()> {
        self.mutate_payload(id, |m| m.clear()).await
    }

    /// The shared payload-mutation path: one WriteBatch carrying the
    /// rewritten doc, exact pidx retraction/rewrite for indexed fields,
    /// the shape-census adjustment, and a changefeed row — atomic.
    async fn mutate_payload(
        &self,
        id: &str,
        f: impl FnOnce(&mut rro_core::Metadata) + Send + 'static,
    ) -> Result<()> {
        let _guard = self.writer.lock().await;
        let db = self.db.clone();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || {
            let Some(mut doc) = db.get_json::<StoredDoc>(CF_DOCS, id.as_bytes())? else {
                return Err(RroError::Recall(format!("no such document: {id}")));
            };
            let mut batch = rocksdb::WriteBatch::default();
            let pidx_cf = db.cf(CF_PIDX)?;
            let indexed = crate::filter::indexed_fields(&db)?;

            // Retract old index rows, adjust the census out.
            for field in &indexed {
                if let Some(v) = doc.metadata.get(field) {
                    batch.delete_cf(pidx_cf, keys::pidx_key(field, v, &id));
                }
            }
            let mut shapes: BTreeMap<String, u64> =
                db.get_json(CF_META, META_SHAPES)?.unwrap_or_default();
            if let Some(n) = shapes.get_mut(&doc.shape.key()) {
                *n = n.saturating_sub(1);
            }

            // Apply the mutation, re-derive the shape, rewrite rows.
            f(&mut doc.metadata);
            doc.shape = Shape::of(&doc.metadata);
            *shapes.entry(doc.shape.key()).or_insert(0) += 1;
            for field in &indexed {
                if let Some(v) = doc.metadata.get(field) {
                    batch.put_cf(pidx_cf, keys::pidx_key(field, v, &id), []);
                }
            }
            batch.put_cf(db.cf(CF_DOCS)?, id.as_bytes(), serde_json::to_vec(&doc)?);

            // Changefeed row, atomic with the mutation.
            let mut feed_seq = db.get_u64(META_FEED_SEQ)?;
            let change = crate::model::Change {
                seq: feed_seq,
                op: crate::model::ChangeOp::Upsert,
                doc_id: id.clone(),
                at: crate::model::now_ms(),
            };
            batch.put_cf(
                db.cf(CF_FEED)?,
                feed_seq.to_be_bytes(),
                serde_json::to_vec(&change)?,
            );
            feed_seq += 1;
            let meta_cf = db.cf(CF_META)?;
            batch.put_cf(meta_cf, META_FEED_SEQ, feed_seq.to_le_bytes());
            batch.put_cf(meta_cf, META_SHAPES, serde_json::to_vec(&shapes)?);

            db.write(batch)
        })
        .await
        .map_err(|e| RroError::Recall(format!("join: {e}")))??;
        self.feed_notify.notify_waiters();
        Ok(())
    }
}

impl ConnXRecall {
    /// The configured query-size cap, if any (quota enforcement).
    pub(crate) fn quota_max_top_k(&self) -> Option<usize> {
        self.quotas.max_top_k
    }

    /// The estate's analyzer (highlighting, diagnostics).
    pub(crate) fn analyzer_ref(&self) -> &rro_core::text::Analyzer {
        &self.analyzer
    }
}
