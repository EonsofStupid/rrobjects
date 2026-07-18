//! Feasibility spike for the RocksDB → Fjall 3.x port (`docs/FJALL_MIGRATION_PLAN.md`).
//!
//! connxism leans on RocksDB for four things, all funnelled through the `Db`
//! wrapper in `estate.rs`:
//!   1. **column families** — one per logical store (`COLUMN_FAMILIES`),
//!   2. **atomic cross-CF batches** (`WriteBatch`) — a doc and its postings land
//!      together or not at all,
//!   3. **prefix/range iteration** (`IteratorMode::From`) — BM25 postings walks,
//!   4. an **associative merge operator** on `tdf` (blind `±1` document-frequency
//!      deltas) — the one thing Fjall has no equivalent for.
//!
//! This grounds the port in the real Fjall 3.1.7 API before a line of connxism is
//! rewritten, and pins the `tdf` replacement (a transaction-scoped read-modify-
//! write) as a test, not a wiki. Gated behind `kvs-fjall` so the default RocksDB
//! build never pulls Fjall in.
//!
//! Terminology note: Fjall 3.1.7 renamed the classic pair. `Database` is the
//! top-level store (the old "Keyspace"); a `Keyspace` is one partition (the old
//! "Partition" / a RocksDB column family). The migration plan uses the 3.1.7
//! names.

#![cfg(feature = "kvs-fjall")]

use fjall::{Database, KeyspaceCreateOptions, PersistMode};

#[test]
fn keyspaces_are_column_families_and_survive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();

    // One Fjall keyspace per RocksDB column family.
    let nodes = db
        .keyspace("nodes", KeyspaceCreateOptions::default)
        .unwrap();
    let vecs = db.keyspace("vecs", KeyspaceCreateOptions::default).unwrap();

    nodes.insert("n1", "alpha").unwrap();
    vecs.insert("n1", [1u8, 2, 3]).unwrap();
    db.persist(PersistMode::SyncAll).unwrap(); // the fsync-on-write path

    assert_eq!(nodes.get("n1").unwrap().as_deref(), Some(&b"alpha"[..]));
    assert_eq!(vecs.get("n1").unwrap().as_deref(), Some(&[1u8, 2, 3][..]));

    // Reopen: durable across restart (journal recovery), like RocksDB's WAL.
    drop((nodes, vecs, db));
    let db = Database::builder(dir.path()).open().unwrap();
    let nodes = db
        .keyspace("nodes", KeyspaceCreateOptions::default)
        .unwrap();
    assert_eq!(nodes.get("n1").unwrap().as_deref(), Some(&b"alpha"[..]));
}

#[test]
fn cross_keyspace_batch_is_atomic() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();
    let docs = db.keyspace("docs", KeyspaceCreateOptions::default).unwrap();
    let terms = db
        .keyspace("terms", KeyspaceCreateOptions::default)
        .unwrap();

    // The commit-path invariant: a doc and its postings commit together. Fjall's
    // batch spans keyspaces and commits atomically — the WriteBatch replacement.
    let mut batch = db.batch();
    batch.insert(&docs, "d1", "hello world");
    batch.insert(&terms, "hello\x00d1", "");
    batch.insert(&terms, "world\x00d1", "");
    batch.commit().unwrap();

    assert!(docs.get("d1").unwrap().is_some());
    assert!(terms.get("hello\x00d1").unwrap().is_some());
    assert!(terms.get("world\x00d1").unwrap().is_some());
}

#[test]
fn prefix_scan_walks_one_terms_postings_list() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();
    let terms = db
        .keyspace("terms", KeyspaceCreateOptions::default)
        .unwrap();

    // `term \x00 doc_id` rows — the BM25 postings layout. A prefix scan over
    // `term \x00` walks exactly that term's list, same as `IteratorMode::From`
    // plus the manual `starts_with` break (which Fjall's `prefix` removes).
    for doc in ["d1", "d2", "d9"] {
        terms.insert(format!("rust\x00{doc}"), "").unwrap();
    }
    terms.insert("go\x00d1", "").unwrap();

    let hits: Vec<String> = terms
        .prefix(b"rust\x00")
        .map(|guard| String::from_utf8_lossy(&guard.key().unwrap()).into_owned())
        .collect();
    assert_eq!(
        hits.len(),
        3,
        "prefix returns only `rust` postings: {hits:?}"
    );
    assert!(hits.iter().all(|k| k.starts_with("rust\x00")));
}

/// The ONE real gap: RocksDB's associative merge operator (`tdf`'s blind `±1`
/// document-frequency deltas). Fjall has no merge operators — its compaction
/// filters transform/drop entries but do not *compose operands*. The replacement
/// is a read-modify-write, correct because connxism serializes every write through
/// one writer lock, so a term's counter never races itself. On-disk format
/// (i64 LE) is unchanged, so `tdf` needs no data migration.
#[test]
fn df_counter_replacement_is_a_read_modify_write() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::builder(dir.path()).open().unwrap();
    let tdf = db.keyspace("tdf", KeyspaceCreateOptions::default).unwrap();

    let bump = |delta: i64| {
        let cur = tdf
            .get("rust")
            .unwrap()
            .map(|b| i64::from_le_bytes(b.as_ref().try_into().unwrap()))
            .unwrap_or(0);
        tdf.insert("rust", (cur + delta).to_le_bytes()).unwrap();
    };
    bump(1);
    bump(1);
    bump(1);
    bump(-1);
    let df = i64::from_le_bytes(
        tdf.get("rust")
            .unwrap()
            .unwrap()
            .as_ref()
            .try_into()
            .unwrap(),
    );
    assert_eq!(
        df, 2,
        "df nets correctly via RMW — no merge operator needed"
    );
}
