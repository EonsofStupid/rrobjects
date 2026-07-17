//! Apply an RRQL [`Statement`] against a live estate.
//!
//! `rro-ql` is a pure function of text — it parses and lowers, and depends only
//! on `rro-core` (ADR-0003). It cannot execute, because executing needs an
//! estate and that would invert the DAG.
//!
//! This module is the other half: the estate-side executor. It is deliberately
//! thin — every arm is a call to a method `connxism` already has and already
//! tests. RRQL adds a *surface*, not a second implementation of the engine.

use std::sync::Arc;

use rro_core::{Metadata, Recall, Result, RroError};
use rro_ql::{Define, Delete, Remove, Statement, Update};

/// What applying a statement produced.
///
/// One shape per statement kind rather than a stringly "rows affected": a caller
/// that asked `DEFINE` and one that asked `DELETE` want different answers, and
/// flattening them loses the only information each carries.
/// There is no `Query` variant on purpose: [`apply`] refuses `SELECT` (it has no
/// embedder), so carrying candidates here would be a shape that can never be
/// produced — and it would have forced `Candidate: PartialEq` on `rro-core` for
/// a variant nobody constructs.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SqlOutcome {
    /// A `DEFINE` applied.
    Defined {
        /// What was defined, e.g. `index author`.
        what: String,
    },
    /// A `REMOVE` applied.
    Removed {
        /// What was removed.
        what: String,
        /// Members dropped, for `REMOVE COLLECTION`.
        #[serde(skip_serializing_if = "Option::is_none")]
        count: Option<u64>,
    },
    /// An `UPDATE` applied.
    Updated {
        /// The record id.
        id: String,
        /// Whether the payload was replaced (`CONTENT`) or merged (`SET`).
        replaced: bool,
    },
    /// A `DELETE` applied.
    Deleted {
        /// The record id.
        id: String,
        /// Whether only the payload was touched.
        payload_only: bool,
    },
}

/// Apply a **write** statement to `estate`.
///
/// `SELECT` is not handled here: it needs an embedder to fill the query vector,
/// which is the flow's job, not the estate's. Callers route reads through
/// `rro_ql::parse_query` + the flow.
///
/// # Errors
/// [`RroError`] if the estate rejects the operation.
pub async fn apply(estate: &Arc<connxism::Estate>, stmt: Statement) -> Result<SqlOutcome> {
    match stmt {
        Statement::Select(_) => Err(RroError::msg(
            "apply() handles writes; a SELECT needs an embedder for its query vector — \
             route it through rro_ql::parse_query and the flow",
        )),

        Statement::Define(Define::Index { field }) => {
            estate.create_payload_index(&field)?;
            Ok(SqlOutcome::Defined {
                what: format!("index {field}"),
            })
        }
        Statement::Define(Define::Alias { alias, collection }) => {
            estate.create_alias(&alias, &collection)?;
            Ok(SqlOutcome::Defined {
                what: format!("alias {alias} -> {collection}"),
            })
        }

        Statement::Remove(Remove::Alias { alias }) => {
            estate.delete_alias(&alias)?;
            Ok(SqlOutcome::Removed {
                what: format!("alias {alias}"),
                count: None,
            })
        }
        Statement::Remove(Remove::Collection { name }) => {
            let count = estate.drop_collection(&name)?;
            Ok(SqlOutcome::Removed {
                what: format!("collection {name}"),
                count: Some(count),
            })
        }

        Statement::Update(Update { id, set, replace }) => {
            let mut meta = Metadata::new();
            for (k, v) in set {
                meta.insert(k, v.to_json());
            }
            // SET merges (set_payload), CONTENT replaces (overwrite_payload).
            // The estate has both; picking one for both spellings would silently
            // destroy fields the caller never mentioned.
            if replace {
                estate.recall().overwrite_payload(&id, meta).await?;
            } else {
                estate.recall().set_payload(&id, meta).await?;
            }
            Ok(SqlOutcome::Updated {
                id,
                replaced: replace,
            })
        }

        Statement::Delete(Delete {
            id,
            payload_only,
            keys,
        }) => {
            let recall = estate.recall();
            match (payload_only, keys.is_empty()) {
                (false, _) => recall.remove(&rro_core::Id::from(id.clone())).await?,
                (true, true) => recall.clear_payload(&id).await?,
                (true, false) => recall.delete_payload_keys(&id, keys).await?,
            }
            Ok(SqlOutcome::Deleted { id, payload_only })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rro_core::VectorRecord;

    async fn estate_with_a_doc() -> (tempfile::TempDir, Arc<connxism::Estate>) {
        let dir = tempfile::tempdir().unwrap();
        let estate = Arc::new(connxism::Estate::open(dir.path(), "sql").unwrap());
        let mut r = VectorRecord::new("doc1", rro_core::Embedding(vec![1.0, 0.0]), "hello");
        r.metadata.insert("team".into(), serde_json::json!("blue"));
        r.metadata.insert("rank".into(), serde_json::json!(1));
        estate.recall().upsert(vec![r]).await.unwrap();
        (dir, estate)
    }

    async fn run(estate: &Arc<connxism::Estate>, src: &str) -> Result<SqlOutcome> {
        apply(estate, rro_ql::parse(src).expect("parses")).await
    }

    /// SET must MERGE: a field the caller never mentioned must survive.
    #[tokio::test]
    async fn update_set_merges_and_leaves_other_fields_alone() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "UPDATE doc1 SET team = 'red'").await.unwrap();
        let m = estate.recall().doc("doc1").await.unwrap().unwrap().metadata;
        assert_eq!(m.get("team").unwrap(), &serde_json::json!("red"));
        assert_eq!(
            m.get("rank").unwrap(),
            &serde_json::json!(1),
            "SET merged, so `rank` must survive — if this fails, SET is silently \
             destroying fields the caller never named"
        );
    }

    /// CONTENT must REPLACE: unmentioned fields are gone, on purpose.
    #[tokio::test]
    async fn update_content_replaces_the_whole_payload() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "UPDATE doc1 CONTENT SET team = 'red'")
            .await
            .unwrap();
        let m = estate.recall().doc("doc1").await.unwrap().unwrap().metadata;
        assert_eq!(m.get("team").unwrap(), &serde_json::json!("red"));
        assert!(m.get("rank").is_none(), "CONTENT replaces; `rank` must be gone");
    }

    #[tokio::test]
    async fn delete_payload_keys_removes_only_those_keys() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "DELETE PAYLOAD doc1 (rank)").await.unwrap();
        let m = estate.recall().doc("doc1").await.unwrap().unwrap().metadata;
        assert!(m.get("rank").is_none());
        assert!(m.get("team").is_some(), "only `rank` was named");
    }

    #[tokio::test]
    async fn delete_payload_clears_but_keeps_the_record() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "DELETE PAYLOAD doc1").await.unwrap();
        let d = estate.recall().doc("doc1").await.unwrap();
        assert!(d.is_some(), "the RECORD must survive DELETE PAYLOAD");
        assert!(d.unwrap().metadata.is_empty());
    }

    #[tokio::test]
    async fn delete_removes_the_record() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "DELETE doc1").await.unwrap();
        assert!(estate.recall().doc("doc1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn define_index_reaches_the_estate() {
        let (_d, estate) = estate_with_a_doc().await;
        let out = run(&estate, "DEFINE INDEX ON team").await.unwrap();
        assert_eq!(
            out,
            SqlOutcome::Defined {
                what: "index team".into()
            }
        );
        assert!(
            estate.payload_indexes().unwrap().contains(&"team".to_string()),
            "the index must actually exist afterwards, not just be reported"
        );
    }

    #[tokio::test]
    async fn define_and_remove_alias_round_trip() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "DEFINE ALIAS current FOR alpha").await.unwrap();
        assert!(estate.aliases().unwrap().contains_key("current"));
        run(&estate, "REMOVE ALIAS current").await.unwrap();
        assert!(!estate.aliases().unwrap().contains_key("current"));
    }

    #[tokio::test]
    async fn a_select_is_refused_here_and_says_where_to_go() {
        let (_d, estate) = estate_with_a_doc().await;
        let e = run(&estate, "SELECT *").await.unwrap_err();
        assert!(e.to_string().contains("embedder"), "{e}");
    }

    /// The estate is asymmetric here, on purpose, and RRQL inherits it rather
    /// than papering over it:
    ///
    /// - `DELETE <ghost>` is a **no-op** — idempotent, the same as SQL's DELETE
    ///   (0 rows affected is not an error). Re-running a delete must be safe.
    /// - `DELETE PAYLOAD <ghost>` **errors** — you asked to modify a record that
    ///   is not there, and silently succeeding would hide the caller's mistake.
    ///
    /// This test exists because the first version asserted `DELETE ghost` errors,
    /// which was my assumption rather than the engine's behaviour. The engine was
    /// right.
    #[tokio::test]
    async fn deleting_a_ghost_record_is_idempotent_but_patching_one_is_not() {
        let (_d, estate) = estate_with_a_doc().await;
        assert!(
            run(&estate, "DELETE ghost").await.is_ok(),
            "DELETE of a missing record must be a no-op, not an error"
        );
        assert!(
            run(&estate, "DELETE PAYLOAD ghost").await.is_err(),
            "patching a missing record must fail loudly"
        );
        assert!(
            run(&estate, "UPDATE ghost SET a = 1").await.is_err(),
            "updating a missing record must fail loudly"
        );
    }
}
