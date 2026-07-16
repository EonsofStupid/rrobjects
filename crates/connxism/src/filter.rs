//! The filter DSL: typed conditions over document metadata.
//!
//! [`Filter`] carries three clause lists with the usual boolean semantics —
//! every `must` holds, at least one `should` holds (when any are given), no
//! `must_not` holds. Conditions cover equality, match-any, numeric ranges,
//! and field existence. The whole structure is serde-serializable so it can
//! ride the a2a wire unchanged.
//!
//! Execution is two-strategy, chosen per query:
//! - **filter-first** when every referenced field has a payload secondary
//!   index (`Estate::create_payload_index`): the exact matching id-set is
//!   resolved from sorted index scans, then scored exactly inside it;
//! - **post-filter** otherwise: over-fetch, hydrate payloads, retain matches.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use rrf_core::{Metadata, Result};

use crate::estate::{rocks_err, Db};
use crate::keys::{self, CF_META, CF_PIDX, META_PIDX, SEP};

/// One testable condition over a metadata field.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
            Condition::Exists { key } => metadata.contains_key(key),
        }
    }
}

/// A boolean combination of [`Condition`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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

    fn keys(&self) -> impl Iterator<Item = &str> {
        self.must
            .iter()
            .chain(&self.should)
            .chain(&self.must_not)
            .map(Condition::key)
    }
}

/// The estate's payload-indexed field names.
pub(crate) fn indexed_fields(db: &Db) -> Result<Vec<String>> {
    Ok(db
        .get_json::<Vec<String>>(CF_META, META_PIDX)?
        .unwrap_or_default())
}

/// Resolve the exact id-set matching `filter` from the payload indexes.
/// Returns `None` when the filter can't be answered from indexes alone
/// (an unindexed field, or only `must_not` clauses — the complement of an
/// index scan is not enumerable cheaply). Ids come back sorted.
pub(crate) fn ids_where(db: &Db, filter: &Filter) -> Result<Option<Vec<String>>> {
    if filter.must.is_empty() && filter.should.is_empty() {
        return Ok(None);
    }
    let indexed = indexed_fields(db)?;
    if !filter.keys().all(|k| indexed.iter().any(|f| f == k)) {
        return Ok(None);
    }

    let mut acc: Option<HashSet<String>> = None;
    for c in &filter.must {
        let ids = ids_for_condition(db, c)?;
        acc = Some(match acc {
            None => ids,
            Some(prev) => prev.intersection(&ids).cloned().collect(),
        });
        if acc.as_ref().map(HashSet::is_empty).unwrap_or(false) {
            return Ok(Some(Vec::new()));
        }
    }
    if !filter.should.is_empty() {
        let mut union = HashSet::new();
        for c in &filter.should {
            union.extend(ids_for_condition(db, c)?);
        }
        acc = Some(match acc {
            None => union,
            Some(prev) => prev.intersection(&union).cloned().collect(),
        });
    }
    let mut set = acc.unwrap_or_default();
    for c in &filter.must_not {
        for id in ids_for_condition(db, c)? {
            set.remove(&id);
        }
    }
    let mut out: Vec<String> = set.into_iter().collect();
    out.sort();
    Ok(Some(out))
}

/// All doc ids matching one condition, from its field's index rows.
fn ids_for_condition(db: &Db, c: &Condition) -> Result<HashSet<String>> {
    match c {
        Condition::Eq { key, value } => scan_value(db, key, value),
        Condition::Any { key, values } => {
            let mut out = HashSet::new();
            for v in values {
                out.extend(scan_value(db, key, v)?);
            }
            Ok(out)
        }
        Condition::Range {
            key,
            gt,
            gte,
            lt,
            lte,
        } => scan_range(db, key, *gt, *gte, *lt, *lte),
        Condition::Exists { key } => scan_field(db, key),
    }
}

/// Prefix-scan every doc id carrying exactly `value` in `field`.
fn scan_value(db: &Db, field: &str, value: &serde_json::Value) -> Result<HashSet<String>> {
    let handle = db.cf(CF_PIDX)?;
    let prefix = keys::pidx_value_prefix(field, value);
    let mut out = HashSet::new();
    for item in db.0.iterator_cf(
        handle,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    ) {
        let (k, _) = item.map_err(rocks_err)?;
        if !k.starts_with(&prefix) {
            break;
        }
        out.insert(String::from_utf8_lossy(&k[prefix.len()..]).into_owned());
    }
    Ok(out)
}

/// Ordered scan of `field`'s numeric rows between the bounds — starts at the
/// lower bound and stops at the upper; only matching rows are touched.
fn scan_range(
    db: &Db,
    field: &str,
    gt: Option<f64>,
    gte: Option<f64>,
    lt: Option<f64>,
    lte: Option<f64>,
) -> Result<HashSet<String>> {
    let handle = db.cf(CF_PIDX)?;
    let num_prefix = keys::pidx_num_prefix(field);
    let lower = gte.or(gt).unwrap_or(f64::NEG_INFINITY);
    let mut start = num_prefix.clone();
    start.extend_from_slice(&keys::encode_f64_sortable(lower));

    let mut out = HashSet::new();
    for item in db.0.iterator_cf(
        handle,
        rocksdb::IteratorMode::From(&start, rocksdb::Direction::Forward),
    ) {
        let (k, _) = item.map_err(rocks_err)?;
        if !k.starts_with(&num_prefix) {
            break;
        }
        let val_start = num_prefix.len();
        let Some(bytes) = k.get(val_start..val_start + 8) else {
            continue;
        };
        let x = keys::decode_f64_sortable(bytes.try_into().expect("8-byte slice"));
        if lt.map(|b| x >= b).unwrap_or(false) || lte.map(|b| x > b).unwrap_or(false) {
            break; // rows sort by value; past the upper bound means done
        }
        if gt.map(|b| x <= b).unwrap_or(false) || gte.map(|b| x < b).unwrap_or(false) {
            continue; // at the boundary of an exclusive lower bound
        }
        // key layout: prefix + 8 value bytes + SEP + doc_id
        let Some(&sep) = k.get(val_start + 8) else {
            continue;
        };
        if sep != SEP {
            continue;
        }
        out.insert(String::from_utf8_lossy(&k[val_start + 9..]).into_owned());
    }
    Ok(out)
}

/// Every doc id with any value in `field` (existence).
fn scan_field(db: &Db, field: &str) -> Result<HashSet<String>> {
    let handle = db.cf(CF_PIDX)?;
    let prefix = keys::pidx_field_prefix(field);
    let mut out = HashSet::new();
    for item in db.0.iterator_cf(
        handle,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    ) {
        let (k, _) = item.map_err(rocks_err)?;
        if !k.starts_with(&prefix) {
            break;
        }
        // The doc id is everything after the LAST separator; typed values may
        // themselves contain NUL bytes (numeric encodings), doc ids may not.
        if let Some(pos) = k.iter().rposition(|&b| b == SEP) {
            out.insert(String::from_utf8_lossy(&k[pos + 1..]).into_owned());
        }
    }
    Ok(out)
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
        let json = serde_json::to_string(&f).unwrap();
        let back: Filter = serde_json::from_str(&json).unwrap();
        assert_eq!(back.must.len(), 1);
        assert_eq!(back.should.len(), 1);
        assert_eq!(back.must_not.len(), 1);
    }
}
