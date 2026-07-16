//! Filter execution against the estate's payload secondary indexes.
//!
//! The DSL itself ([`rrf_core::Filter`], [`rrf_core::Condition`]) is pure
//! data in the core contract — clients build filters without a storage
//! dependency. This module is the estate side: resolving a filter to its
//! exact matching id-set from sorted `pidx` scans.
//!
//! Execution is two-strategy, chosen per query:
//! - **filter-first** when every referenced field has a payload secondary
//!   index (`Estate::create_payload_index`): the exact matching id-set is
//!   resolved from sorted index scans, then scored exactly inside it;
//! - **post-filter** otherwise: over-fetch, hydrate payloads, retain matches.

use std::collections::HashSet;

use rrf_core::{Condition, Filter, Result};

use crate::estate::{rocks_err, Db};
use crate::keys::{self, CF_META, CF_PIDX, META_PIDX, SEP};

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
        Condition::DateRange {
            key,
            gt,
            gte,
            lt,
            lte,
        } => {
            let ms =
                |o: &Option<String>| o.as_deref().and_then(rrf_core::time::rfc3339_to_epoch_ms);
            scan_dt_range(db, key, ms(gt), ms(gte), ms(lt), ms(lte))
        }
        Condition::GeoRadius {
            key,
            lat,
            lon,
            radius_m,
        } => {
            let ((lat_min, lon_min), (lat_max, lon_max)) =
                rrf_core::geo::radius_bbox(*lat, *lon, *radius_m);
            scan_geo(db, key, lat_min, lon_min, lat_max, lon_max, c)
        }
        Condition::GeoBox {
            key,
            lat_min,
            lon_min,
            lat_max,
            lon_max,
        } => scan_geo(db, key, *lat_min, *lon_min, *lat_max, *lon_max, c),
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

/// Ordered scan of `field`'s datetime rows between the bounds (epoch ms) —
/// starts at the lower bound, stops at the upper. Same shape as
/// [`scan_range`], over the `PIDX_DT` tag's chronology.
fn scan_dt_range(
    db: &Db,
    field: &str,
    gt: Option<i64>,
    gte: Option<i64>,
    lt: Option<i64>,
    lte: Option<i64>,
) -> Result<HashSet<String>> {
    let handle = db.cf(CF_PIDX)?;
    let dt_prefix = keys::pidx_dt_prefix(field);
    let lower = gte.or(gt).unwrap_or(i64::MIN);
    let mut start = dt_prefix.clone();
    start.extend_from_slice(&keys::encode_i64_sortable(lower));

    let mut out = HashSet::new();
    for item in db.0.iterator_cf(
        handle,
        rocksdb::IteratorMode::From(&start, rocksdb::Direction::Forward),
    ) {
        let (k, _) = item.map_err(rocks_err)?;
        if !k.starts_with(&dt_prefix) {
            break;
        }
        let val_start = dt_prefix.len();
        let Some(bytes) = k.get(val_start..val_start + 8) else {
            continue;
        };
        let raw = u64::from_be_bytes(bytes.try_into().expect("8-byte slice"));
        let x = (raw ^ (1 << 63)) as i64;
        if lt.map(|b| x >= b).unwrap_or(false) || lte.map(|b| x > b).unwrap_or(false) {
            break; // chronological order: past the upper bound means done
        }
        if gt.map(|b| x <= b).unwrap_or(false) || gte.map(|b| x < b).unwrap_or(false) {
            continue;
        }
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

/// Geo scan: one Z-order range `[morton(min corner), morton(max corner)]`
/// covers every point of the box (Morton codes are monotone per axis);
/// Z-jumps outside the box are false positives, culled by re-checking the
/// condition EXACTLY against each candidate's stored metadata — so radius
/// results are true haversine, never quantized.
#[allow(clippy::too_many_arguments)]
fn scan_geo(
    db: &Db,
    field: &str,
    lat_min: f64,
    lon_min: f64,
    lat_max: f64,
    lon_max: f64,
    exact: &Condition,
) -> Result<HashSet<String>> {
    let handle = db.cf(CF_PIDX)?;
    let geo_prefix = keys::pidx_geo_prefix(field);
    let z_lo = rrf_core::geo::morton(lat_min, lon_min);
    let z_hi = rrf_core::geo::morton(lat_max, lon_max);
    let mut start = geo_prefix.clone();
    start.extend_from_slice(&z_lo.to_be_bytes());

    let docs_cf = db.cf(crate::keys::CF_DOCS)?;
    let mut out = HashSet::new();
    for item in db.0.iterator_cf(
        handle,
        rocksdb::IteratorMode::From(&start, rocksdb::Direction::Forward),
    ) {
        let (k, _) = item.map_err(rocks_err)?;
        if !k.starts_with(&geo_prefix) {
            break;
        }
        let val_start = geo_prefix.len();
        let Some(bytes) = k.get(val_start..val_start + 8) else {
            continue;
        };
        let z = u64::from_be_bytes(bytes.try_into().expect("8-byte slice"));
        if z > z_hi {
            break; // sorted by code; past the max corner means done
        }
        let Some(&sep) = k.get(val_start + 8) else {
            continue;
        };
        if sep != SEP {
            continue;
        }
        let doc_id = String::from_utf8_lossy(&k[val_start + 9..]).into_owned();
        // Exact check from the durable document.
        if let Some(bytes) = db.0.get_cf(docs_cf, doc_id.as_bytes()).map_err(rocks_err)? {
            let doc: crate::model::StoredDoc = serde_json::from_slice(&bytes)?;
            if exact.matches(&doc.metadata) {
                out.insert(doc_id);
            }
        }
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
