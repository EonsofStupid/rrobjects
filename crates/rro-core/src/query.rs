//! The typed query contract: what any consumer sends to any recall store.
//!
//! These are pure data types — no storage, no execution. The estate (or any
//! remote node, over the a2a wire) executes them; clients build them. That
//! split is why a thin client can speak the full query plane without
//! depending on a storage engine.

use serde::{Deserialize, Serialize};

use crate::types::{Embedding, Metadata, SparseVector};

/// One testable condition over a metadata field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Condition {
    /// The field equals this value exactly.
    Eq {
        /// Metadata field name.
        key: String,
        /// Required value.
        value: serde_json::Value,
    },
    /// The field equals any of these values.
    Any {
        /// Metadata field name.
        key: String,
        /// Accepted values.
        values: Vec<serde_json::Value>,
    },
    /// The field is a number inside the given (half-)open interval.
    Range {
        /// Metadata field name.
        key: String,
        /// Exclusive lower bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gt: Option<f64>,
        /// Inclusive lower bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gte: Option<f64>,
        /// Exclusive upper bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lt: Option<f64>,
        /// Inclusive upper bound.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lte: Option<f64>,
    },
    /// The field is an RFC3339 timestamp inside the given (half-)open
    /// interval (bounds are RFC3339 strings; comparison is by instant, so
    /// mixed offsets compare correctly).
    DateRange {
        /// Metadata field name.
        key: String,
        /// Exclusive lower bound (RFC3339).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gt: Option<String>,
        /// Inclusive lower bound (RFC3339).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gte: Option<String>,
        /// Exclusive upper bound (RFC3339).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lt: Option<String>,
        /// Inclusive upper bound (RFC3339).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lte: Option<String>,
    },
    /// The field is a `{lat, lon}` point within `radius_m` meters of the
    /// center (true haversine).
    GeoRadius {
        /// Metadata field name.
        key: String,
        /// Center latitude (degrees).
        lat: f64,
        /// Center longitude (degrees).
        lon: f64,
        /// Radius in meters.
        radius_m: f64,
    },
    /// The field is a `{lat, lon}` point inside the box. The box must not
    /// cross the antimeridian (v1 limit).
    GeoBox {
        /// Metadata field name.
        key: String,
        /// Southern edge (min latitude, degrees).
        lat_min: f64,
        /// Western edge (min longitude, degrees).
        lon_min: f64,
        /// Northern edge (max latitude, degrees).
        lat_max: f64,
        /// Eastern edge (max longitude, degrees).
        lon_max: f64,
    },
    /// The field is present (any value).
    Exists {
        /// Metadata field name.
        key: String,
    },
}

impl Condition {
    /// Equality condition.
    pub fn eq(key: impl Into<String>, value: serde_json::Value) -> Self {
        Condition::Eq {
            key: key.into(),
            value,
        }
    }

    /// Match-any condition.
    pub fn any(key: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        Condition::Any {
            key: key.into(),
            values,
        }
    }

    /// Inclusive numeric range `[gte, lte]` (pass `None` to leave a side open).
    pub fn range(key: impl Into<String>, gte: Option<f64>, lte: Option<f64>) -> Self {
        Condition::Range {
            key: key.into(),
            gt: None,
            gte,
            lt: None,
            lte,
        }
    }

    /// Inclusive datetime range `[gte, lte]` in RFC3339 (pass `None` to
    /// leave a side open).
    pub fn date_range(
        key: impl Into<String>,
        gte: Option<impl Into<String>>,
        lte: Option<impl Into<String>>,
    ) -> Self {
        Condition::DateRange {
            key: key.into(),
            gt: None,
            gte: gte.map(Into::into),
            lt: None,
            lte: lte.map(Into::into),
        }
    }

    /// Radius condition: within `radius_m` meters of `(lat, lon)`.
    pub fn geo_radius(key: impl Into<String>, lat: f64, lon: f64, radius_m: f64) -> Self {
        Condition::GeoRadius {
            key: key.into(),
            lat,
            lon,
            radius_m,
        }
    }

    /// Box condition: inside `[lat_min..lat_max] × [lon_min..lon_max]`.
    pub fn geo_box(
        key: impl Into<String>,
        lat_min: f64,
        lon_min: f64,
        lat_max: f64,
        lon_max: f64,
    ) -> Self {
        Condition::GeoBox {
            key: key.into(),
            lat_min,
            lon_min,
            lat_max,
            lon_max,
        }
    }

    /// Existence condition.
    pub fn exists(key: impl Into<String>) -> Self {
        Condition::Exists { key: key.into() }
    }

    /// The metadata field this condition reads.
    pub fn key(&self) -> &str {
        match self {
            Condition::Eq { key, .. }
            | Condition::Any { key, .. }
            | Condition::Range { key, .. }
            | Condition::DateRange { key, .. }
            | Condition::GeoRadius { key, .. }
            | Condition::GeoBox { key, .. }
            | Condition::Exists { key } => key,
        }
    }

    /// Whether `metadata` satisfies this condition.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self {
            Condition::Eq { key, value } => metadata.get(key) == Some(value),
            Condition::Any { key, values } => metadata
                .get(key)
                .map(|v| values.contains(v))
                .unwrap_or(false),
            Condition::Range {
                key,
                gt,
                gte,
                lt,
                lte,
            } => match metadata.get(key).and_then(|v| v.as_f64()) {
                Some(x) => {
                    gt.map(|b| x > b).unwrap_or(true)
                        && gte.map(|b| x >= b).unwrap_or(true)
                        && lt.map(|b| x < b).unwrap_or(true)
                        && lte.map(|b| x <= b).unwrap_or(true)
                }
                None => false,
            },
            Condition::DateRange {
                key,
                gt,
                gte,
                lt,
                lte,
            } => {
                let parse = |s: &String| crate::time::rfc3339_to_epoch_ms(s);
                match metadata
                    .get(key)
                    .and_then(|v| v.as_str())
                    .and_then(crate::time::rfc3339_to_epoch_ms)
                {
                    Some(x) => {
                        gt.as_ref().and_then(parse).map(|b| x > b).unwrap_or(true)
                            && gte.as_ref().and_then(parse).map(|b| x >= b).unwrap_or(true)
                            && lt.as_ref().and_then(parse).map(|b| x < b).unwrap_or(true)
                            && lte.as_ref().and_then(parse).map(|b| x <= b).unwrap_or(true)
                    }
                    None => false,
                }
            }
            Condition::GeoRadius {
                key,
                lat,
                lon,
                radius_m,
            } => match metadata.get(key).and_then(crate::geo::point_of) {
                Some((plat, plon)) => crate::geo::haversine_m(*lat, *lon, plat, plon) <= *radius_m,
                None => false,
            },
            Condition::GeoBox {
                key,
                lat_min,
                lon_min,
                lat_max,
                lon_max,
            } => match metadata.get(key).and_then(crate::geo::point_of) {
                Some((plat, plon)) => {
                    plat >= *lat_min && plat <= *lat_max && plon >= *lon_min && plon <= *lon_max
                }
                None => false,
            },
            Condition::Exists { key } => metadata.contains_key(key),
        }
    }
}

/// A boolean combination of [`Condition`]s: every `must` holds, at least one
/// `should` holds (when any are given), no `must_not` holds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    /// Every condition must hold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must: Vec<Condition>,
    /// At least one must hold, when any are given.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub should: Vec<Condition>,
    /// None may hold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub must_not: Vec<Condition>,
}

impl Filter {
    /// An empty filter (matches everything).
    pub fn new() -> Self {
        Filter::default()
    }

    /// Add a `must` clause.
    pub fn must(mut self, c: Condition) -> Self {
        self.must.push(c);
        self
    }

    /// Add a `should` clause.
    pub fn should(mut self, c: Condition) -> Self {
        self.should.push(c);
        self
    }

    /// Add a `must_not` clause.
    pub fn must_not(mut self, c: Condition) -> Self {
        self.must_not.push(c);
        self
    }

    /// Whether no clauses are present.
    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.should.is_empty() && self.must_not.is_empty()
    }

    /// Whether `metadata` satisfies the whole filter.
    pub fn matches(&self, metadata: &Metadata) -> bool {
        self.must.iter().all(|c| c.matches(metadata))
            && (self.should.is_empty() || self.should.iter().any(|c| c.matches(metadata)))
            && !self.must_not.iter().any(|c| c.matches(metadata))
    }

    /// Every field name any clause reads.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.must
            .iter()
            .chain(&self.should)
            .chain(&self.must_not)
            .map(Condition::key)
    }
}

fn default_true() -> bool {
    true
}

/// One prefetch stage: gather `limit` candidates with an inner query;
/// the outer query rescores inside the union of its prefetches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prefetch {
    /// The inner retrieval (may itself carry prefetches — depth-capped
    /// by executors).
    pub query: EstateQuery,
    /// How many candidates this stage contributes.
    pub limit: usize,
}

/// How much say each retriever gets in hybrid fusion.
///
/// Plain RRF is this struct at 1:1 — an assumption that dense and lexical
/// retrieval are equally trustworthy on your corpus. They usually are not, and
/// the equal-vote default is what makes fused results score *below* the better
/// arm alone: a lexical-only hit at rank 1 contributes `1/(60+1)` and outranks a
/// dense hit at rank 2 at `1/(60+2)`, so the weaker retriever outvotes the
/// stronger one on its own turf.
///
/// This lives on the **query**, not on the estate: the right weight is a
/// property of what is being asked, not of what is stored. "Find `E0521`" wants
/// the lexical arm; "why does fusion regress" does not. Measured on nfcorpus,
/// the best lexical weight is ~0 — see `docs/BENCHMARKS_REAL.md` Finding 1.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HybridWeights {
    /// Vote scale for the dense (vector) ranking.
    pub dense: f32,
    /// Vote scale for the lexical (BM25) ranking.
    pub lexical: f32,
}

impl Default for HybridWeights {
    /// 1:1 — identical to plain RRF, so this knob changes nobody's results
    /// until they ask it to. Deliberately **not** a value tuned on a benchmark:
    /// baking in one corpus's answer is overfitting shipped as a default.
    fn default() -> Self {
        HybridWeights {
            dense: 1.0,
            lexical: 1.0,
        }
    }
}

impl HybridWeights {
    /// As the `[dense, lexical]` vector the fusion takes — the order both
    /// fusion call sites build their lists in.
    pub fn as_slice(self) -> [f32; 2] {
        [self.dense, self.lexical]
    }
}

/// A typed retrieval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstateQuery {
    /// Query text (drives the lexical half and, via the caller's embedder,
    /// usually the vector too).
    pub text: Option<String>,
    /// Dense query vector.
    pub vector: Option<Embedding>,
    /// Weighted sparse query vector — fused with the dense/lexical rankings
    /// by executors that maintain a sparse index.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseVector>,
    /// Route the dense half to a named vector space instead of the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub using: Option<String>,
    /// Restrict to one named collection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    /// Late-interaction query token vectors: candidates are rescored by
    /// MaxSim against their stored token vectors (docs without any sort
    /// after those that have them, in first-phase order).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi: Option<Vec<Embedding>>,
    /// Results wanted.
    pub top_k: usize,
    /// Metadata equality filter: every key must match exactly (legacy form;
    /// merged into `dsl` as `must` equality clauses at execution).
    #[serde(default)]
    pub filter: Metadata,
    /// The typed filter DSL: must / should / must_not clauses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsl: Option<Filter>,
    /// Restrict to these ids (e.g. a routed neighborhood). Exact scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<Vec<String>>,
    /// Drop candidates scoring below this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f32>,
    /// Carry text + metadata on results (`false` returns ids and scores only).
    #[serde(default = "default_true")]
    pub with_payload: bool,
    /// Skip this many ranked results before taking `top_k` (pagination).
    #[serde(default)]
    pub offset: usize,
    /// Carry each winner's stored dense vector on the candidate.
    #[serde(default)]
    pub with_vectors: bool,
    /// Carry analyzer-aware highlight spans of the query terms on each
    /// winner's text.
    #[serde(default)]
    pub highlight: bool,
    /// Prefetch pipeline: each stage gathers candidates by its own signal;
    /// the outer query rescores exactly inside their union.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefetch: Vec<Prefetch>,
    /// How much say each retriever gets when the dense and lexical rankings are
    /// fused. Defaults to 1:1 (plain RRF).
    #[serde(default)]
    pub fusion: HybridWeights,
}

impl Default for EstateQuery {
    fn default() -> Self {
        EstateQuery {
            text: None,
            vector: None,
            sparse: None,
            using: None,
            collection: None,
            multi: None,
            top_k: 0,
            filter: Metadata::new(),
            dsl: None,
            scope: None,
            score_threshold: None,
            with_payload: true,
            offset: 0,
            with_vectors: false,
            highlight: false,
            prefetch: Vec::new(),
            fusion: HybridWeights::default(),
        }
    }
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

    /// A text-only query for the top `k` (the executor embeds it, or answers
    /// lexically if it has no embedder).
    pub fn text(text: impl Into<String>, k: usize) -> Self {
        EstateQuery {
            text: Some(text.into()),
            top_k: k,
            ..EstateQuery::default()
        }
    }

    /// Add a metadata equality condition.
    pub fn must(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.filter.insert(key.into(), value);
        self
    }

    /// Attach a weighted sparse query vector (fused with the other rankings).
    pub fn sparse_vector(mut self, sparse: SparseVector) -> Self {
        self.sparse = Some(sparse);
        self
    }

    /// Route the dense half to a named vector space.
    pub fn using(mut self, name: impl Into<String>) -> Self {
        self.using = Some(name.into());
        self
    }

    /// Restrict to one named collection.
    pub fn in_collection(mut self, name: impl Into<String>) -> Self {
        self.collection = Some(name.into());
        self
    }

    /// Rescore candidates by MaxSim against these query token vectors.
    pub fn multi_query(mut self, vectors: Vec<Embedding>) -> Self {
        self.multi = Some(vectors);
        self
    }

    /// Attach a typed filter (must / should / must_not clauses).
    pub fn filtered(mut self, filter: Filter) -> Self {
        self.dsl = Some(filter);
        self
    }

    /// Restrict to a routed scope.
    pub fn within(mut self, scope: Vec<String>) -> Self {
        self.scope = Some(scope);
        self
    }

    /// Drop candidates scoring below `t`.
    pub fn threshold(mut self, t: f32) -> Self {
        self.score_threshold = Some(t);
        self
    }

    /// Return ids and scores only (no text, no metadata).
    pub fn ids_only(mut self) -> Self {
        self.with_payload = false;
        self
    }

    /// Skip the first `n` ranked results (pagination).
    pub fn offset(mut self, n: usize) -> Self {
        self.offset = n;
        self
    }

    /// Carry each winner's stored dense vector on the candidate.
    pub fn with_vectors(mut self) -> Self {
        self.with_vectors = true;
        self
    }

    /// Carry highlight spans on each winner's text.
    pub fn highlighted(mut self) -> Self {
        self.highlight = true;
        self
    }

    /// Add a prefetch stage (gather `limit` candidates with `query`; the
    /// outer query rescores inside the union of all prefetches).
    pub fn prefetch(mut self, query: EstateQuery, limit: usize) -> Self {
        self.prefetch.push(Prefetch { query, limit });
        self
    }

    /// The effective filter: DSL clauses plus legacy equality pairs.
    pub fn effective_filter(&self) -> Filter {
        let mut dsl = self.dsl.clone().unwrap_or_default();
        for (k, v) in &self.filter {
            dsl.must.push(Condition::eq(k.clone(), v.clone()));
        }
        dsl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, serde_json::Value)]) -> Metadata {
        let mut m = Metadata::new();
        for (k, v) in pairs {
            m.insert((*k).to_string(), v.clone());
        }
        m
    }

    #[test]
    fn clause_semantics() {
        let m = meta(&[
            ("team", serde_json::json!("ops")),
            ("priority", serde_json::json!(3)),
        ]);

        assert!(Filter::new()
            .must(Condition::eq("team", serde_json::json!("ops")))
            .must(Condition::range("priority", Some(1.0), Some(4.0)))
            .matches(&m));

        assert!(!Filter::new()
            .must_not(Condition::eq("team", serde_json::json!("ops")))
            .matches(&m));

        // should: at least one must hold when any are given.
        assert!(Filter::new()
            .should(Condition::eq("team", serde_json::json!("eng")))
            .should(Condition::exists("priority"))
            .matches(&m));
        assert!(!Filter::new()
            .should(Condition::eq("team", serde_json::json!("eng")))
            .matches(&m));

        // range on a missing / non-numeric field never matches.
        assert!(!Condition::range("missing", Some(0.0), None).matches(&m));
        assert!(!Condition::range("team", Some(0.0), None).matches(&m));
    }

    #[test]
    fn serde_roundtrip() {
        let f = Filter::new()
            .must(Condition::eq("team", serde_json::json!("ops")))
            .should(Condition::any(
                "kind",
                vec![serde_json::json!("doc"), serde_json::json!("mail")],
            ))
            .must_not(Condition::range("priority", None, Some(1.0)));
        let q = EstateQuery::text("rollout", 5).filtered(f).threshold(0.2);
        let json = serde_json::to_string(&q).unwrap();
        let back: EstateQuery = serde_json::from_str(&json).unwrap();
        let f = back.dsl.unwrap();
        assert_eq!(f.must.len(), 1);
        assert_eq!(f.should.len(), 1);
        assert_eq!(f.must_not.len(), 1);
        assert_eq!(back.score_threshold, Some(0.2));
        assert!(back.with_payload, "payload defaults on over the wire");
    }
}
