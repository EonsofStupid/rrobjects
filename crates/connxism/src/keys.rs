//! Column families and key encodings — the estate's physical layout.
//!
//! One estate == one RocksDB. Everything the engine knows lives in a column
//! family with a documented key scheme, so the layout *is* the spec:
//!
//! | CF        | key                  | value                              |
//! |-----------|----------------------|------------------------------------|
//! | `meta`    | fixed strings        | JSON ([`crate::model::EstateInfo`], counters, shape counts) |
//! | `nodes`   | node id              | JSON [`crate::model::NodeInfo`]     |
//! | `conns`   | connector id         | JSON [`crate::model::ConnectorInfo`]|
//! | `docs`    | doc id               | JSON [`crate::model::StoredDoc`]    |
//! | `vecs`    | doc id               | f32-LE bytes (the embedding)        |
//! | `terms`   | `term \x00 doc id`   | JSON posting `{tf, len}`            |
//! | `tags`    | `tag \x00 doc id`    | empty (presence = membership)       |
//! | `trends`  | `metric \x00 ts_be`  | f64-LE bytes                        |
//!
//! Postings are **one row per (term, document)** — writes are blind puts
//! (no read-modify-write), reads are sorted prefix scans. This is the
//! LSM-native inverted-index layout; it is what lets ingestion stay
//! write-amplification-flat as the corpus grows.

/// All column families, in creation order.
pub const COLUMN_FAMILIES: &[&str] = &[
    CF_META, CF_NODES, CF_CONNS, CF_DOCS, CF_VECS, CF_TERMS, CF_TAGS, CF_TRENDS, CF_RELS, CF_FEED,
    CF_PIDX,
];

/// Estate metadata + counters.
pub const CF_META: &str = "meta";
/// Node registry.
pub const CF_NODES: &str = "nodes";
/// Connector registry.
pub const CF_CONNS: &str = "conns";
/// Document payloads.
pub const CF_DOCS: &str = "docs";
/// Dense vectors.
pub const CF_VECS: &str = "vecs";
/// BM25 inverted index (postings).
pub const CF_TERMS: &str = "terms";
/// Tag membership.
pub const CF_TAGS: &str = "tags";
/// Metric time-series.
pub const CF_TRENDS: &str = "trends";
/// Relations (RELATE-style edges), both directions.
pub const CF_RELS: &str = "rels";
/// Durable changefeed: seq (u64 BE) → JSON change record.
pub const CF_FEED: &str = "feed";
/// Payload secondary indexes: `field \x00 typed-value \x00 doc_id` → empty.
pub const CF_PIDX: &str = "pidx";

/// meta: the estate info blob.
pub const META_ESTATE: &[u8] = b"estate";
/// meta: total indexed documents (u64 LE).
pub const META_DOC_COUNT: &[u8] = b"doc_count";
/// meta: sum of all document token lengths (u64 LE), for BM25 avgdl.
pub const META_TOTAL_TOKENS: &[u8] = b"total_tokens";
/// meta: JSON map `shape key → count`.
pub const META_SHAPES: &[u8] = b"shapes";
/// meta: next changefeed sequence number (u64 LE).
pub const META_FEED_SEQ: &[u8] = b"feed_seq";
/// meta: JSON array of payload-indexed metadata field names.
pub const META_PIDX: &[u8] = b"payload_indexes";

/// Separator between compound-key segments (never appears in ids/tags/metrics).
pub const SEP: u8 = 0x00;

/// Encode a tag-membership key: `tag \x00 doc_id`.
pub fn tag_key(tag: &str, doc_id: &str) -> Vec<u8> {
    compound(tag, doc_id)
}

/// Prefix that scans all documents carrying `tag`.
pub fn tag_prefix(tag: &str) -> Vec<u8> {
    prefix(tag)
}

/// Encode a postings row key: `term \x00 doc_id`.
pub fn term_key(term: &str, doc_id: &str) -> Vec<u8> {
    compound(term, doc_id)
}

/// Prefix that scans a term's whole postings list.
pub fn term_prefix(term: &str) -> Vec<u8> {
    prefix(term)
}

fn compound(a: &str, b: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(a.len() + 1 + b.len());
    k.extend_from_slice(a.as_bytes());
    k.push(SEP);
    k.extend_from_slice(b.as_bytes());
    k
}

fn prefix(a: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(a.len() + 1);
    k.extend_from_slice(a.as_bytes());
    k.push(SEP);
    k
}

/// Encode a trend key: `metric \x00 timestamp-be` (big-endian sorts by time).
/// The timestamp is **nanoseconds** so rapid samples never collide.
pub fn trend_key(metric: &str, at_ns: u64) -> Vec<u8> {
    let mut k = Vec::with_capacity(metric.len() + 1 + 8);
    k.extend_from_slice(metric.as_bytes());
    k.push(SEP);
    k.extend_from_slice(&at_ns.to_be_bytes());
    k
}

/// Prefix that scans a metric's whole series.
pub fn trend_prefix(metric: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(metric.len() + 1);
    k.extend_from_slice(metric.as_bytes());
    k.push(SEP);
    k
}

/// Split a compound key at the separator; returns (prefix, suffix).
pub fn split_compound(key: &[u8]) -> Option<(&[u8], &[u8])> {
    let pos = key.iter().position(|&b| b == SEP)?;
    Some((&key[..pos], &key[pos + 1..]))
}

/// Direction marker for outbound relation rows.
pub const REL_OUT: u8 = b'o';
/// Direction marker for inbound relation rows.
pub const REL_IN: u8 = b'i';

/// Encode a relation row: `dir  anchor \x00 verb \x00 other`.
///
/// Every RELATE writes two rows — an `o` row anchored on `from` and an `i`
/// row anchored on `to` — so traversal in either direction is one sorted
/// prefix scan, and every write stays a blind put.
pub fn rel_key(dir: u8, anchor: &str, verb: &str, other: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + anchor.len() + 1 + verb.len() + 1 + other.len());
    k.push(dir);
    k.extend_from_slice(anchor.as_bytes());
    k.push(SEP);
    k.extend_from_slice(verb.as_bytes());
    k.push(SEP);
    k.extend_from_slice(other.as_bytes());
    k
}

/// Prefix scanning every relation of `anchor` in one direction (all verbs).
pub fn rel_prefix(dir: u8, anchor: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + anchor.len() + 1);
    k.push(dir);
    k.extend_from_slice(anchor.as_bytes());
    k.push(SEP);
    k
}

/// Prefix scanning `anchor`'s relations under one verb.
pub fn rel_verb_prefix(dir: u8, anchor: &str, verb: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(1 + anchor.len() + 1 + verb.len() + 1);
    k.push(dir);
    k.extend_from_slice(anchor.as_bytes());
    k.push(SEP);
    k.extend_from_slice(verb.as_bytes());
    k.push(SEP);
    k
}

/// Decode `verb \x00 other` from a relation key's suffix (after the prefix).
pub fn rel_suffix(key: &[u8], prefix_len: usize) -> Option<(String, String)> {
    let rest = key.get(prefix_len..)?;
    let (verb, other) = split_compound(rest)?;
    Some((
        String::from_utf8_lossy(verb).into_owned(),
        String::from_utf8_lossy(other).into_owned(),
    ))
}

// ---- payload secondary index keys ------------------------------------------
//
// Row: `field \x00 typed-value \x00 doc_id` → empty (presence = membership).
// The typed value is a one-byte tag + an order-preserving encoding, so a
// sorted prefix scan under `field \x00 n` walks numbers in numeric order —
// range filters are index scans, not table scans. Document ids must not
// contain NUL (already the rule for tags and terms).

/// Type tag: number (order-preserving 8-byte encoding follows).
pub const PIDX_NUM: u8 = b'n';
/// Type tag: string (raw UTF-8 bytes follow).
pub const PIDX_STR: u8 = b's';
/// Type tag: boolean (one byte, 0/1).
pub const PIDX_BOOL: u8 = b'b';
/// Type tag: anything else (canonical JSON text follows).
pub const PIDX_OTHER: u8 = b'o';

/// Order-preserving byte encoding of an `f64`: negative values invert all
/// bits, non-negative values flip the sign bit — lexicographic byte order
/// equals numeric order.
pub fn encode_f64_sortable(x: f64) -> [u8; 8] {
    let bits = x.to_bits();
    let mapped = if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits | (1 << 63)
    };
    mapped.to_be_bytes()
}

/// Inverse of [`encode_f64_sortable`].
pub fn decode_f64_sortable(b: [u8; 8]) -> f64 {
    let mapped = u64::from_be_bytes(b);
    let bits = if mapped & (1 << 63) != 0 {
        mapped & !(1 << 63)
    } else {
        !mapped
    };
    f64::from_bits(bits)
}

/// Typed, order-preserving encoding of a metadata value (tag + payload).
pub fn encode_pidx_value(value: &serde_json::Value) -> Vec<u8> {
    match value {
        serde_json::Value::Number(n) => {
            let mut out = Vec::with_capacity(9);
            out.push(PIDX_NUM);
            out.extend_from_slice(&encode_f64_sortable(n.as_f64().unwrap_or(0.0)));
            out
        }
        serde_json::Value::String(s) => {
            let mut out = Vec::with_capacity(1 + s.len());
            out.push(PIDX_STR);
            out.extend_from_slice(s.as_bytes());
            out
        }
        serde_json::Value::Bool(b) => vec![PIDX_BOOL, u8::from(*b)],
        other => {
            let text = other.to_string();
            let mut out = Vec::with_capacity(1 + text.len());
            out.push(PIDX_OTHER);
            out.extend_from_slice(text.as_bytes());
            out
        }
    }
}

/// Full payload-index row key: `field \x00 typed-value \x00 doc_id`.
pub fn pidx_key(field: &str, value: &serde_json::Value, doc_id: &str) -> Vec<u8> {
    let enc = encode_pidx_value(value);
    let mut k = Vec::with_capacity(field.len() + 1 + enc.len() + 1 + doc_id.len());
    k.extend_from_slice(field.as_bytes());
    k.push(SEP);
    k.extend_from_slice(&enc);
    k.push(SEP);
    k.extend_from_slice(doc_id.as_bytes());
    k
}

/// Prefix scanning every row of one field (all values, all types).
pub fn pidx_field_prefix(field: &str) -> Vec<u8> {
    prefix(field)
}

/// Prefix scanning every document carrying exactly `value` in `field`.
pub fn pidx_value_prefix(field: &str, value: &serde_json::Value) -> Vec<u8> {
    let enc = encode_pidx_value(value);
    let mut k = Vec::with_capacity(field.len() + 1 + enc.len() + 1);
    k.extend_from_slice(field.as_bytes());
    k.push(SEP);
    k.extend_from_slice(&enc);
    k.push(SEP);
    k
}

/// Prefix under which all *numeric* rows of `field` sort in numeric order.
pub fn pidx_num_prefix(field: &str) -> Vec<u8> {
    let mut k = prefix(field);
    k.push(PIDX_NUM);
    k
}

/// Encode an embedding as little-endian f32 bytes.
pub fn encode_vec(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode little-endian f32 bytes back into a vector.
pub fn decode_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.75, 0.0];
        assert_eq!(decode_vec(&encode_vec(&v)), v);
    }

    #[test]
    fn trend_keys_sort_by_time() {
        let a = trend_key("qps", 1);
        let b = trend_key("qps", 2);
        let c = trend_key("qps", 10);
        assert!(a < b && b < c);
    }

    #[test]
    fn f64_sortable_encoding_orders_numerically() {
        let vals = [
            f64::NEG_INFINITY,
            -1e9,
            -3.5,
            -0.0,
            0.0,
            0.001,
            42.0,
            1e12,
            f64::INFINITY,
        ];
        for w in vals.windows(2) {
            assert!(
                encode_f64_sortable(w[0]) <= encode_f64_sortable(w[1]),
                "{} !<= {}",
                w[0],
                w[1]
            );
        }
        for v in vals {
            assert_eq!(decode_f64_sortable(encode_f64_sortable(v)), v);
        }
    }

    #[test]
    fn pidx_numeric_rows_sort_by_value() {
        let a = pidx_key("priority", &serde_json::json!(-2), "d1");
        let b = pidx_key("priority", &serde_json::json!(0), "d1");
        let c = pidx_key("priority", &serde_json::json!(10), "d1");
        assert!(a < b && b < c);
    }

    #[test]
    fn compound_split() {
        let k = tag_key("alpha", "doc9");
        let (t, d) = split_compound(&k).unwrap();
        assert_eq!(t, b"alpha");
        assert_eq!(d, b"doc9");
    }
}
