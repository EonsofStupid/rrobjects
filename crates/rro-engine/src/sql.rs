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
use rro_ql::{Define, Delete, Direction, Relate, Remove, Statement, Update};

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
    /// A `RELATE` applied.
    Related {
        /// `from -verb-> to`.
        edge: String,
    },
    /// A `TRAVERSE` walked.
    Traversed {
        /// Visited ids, in traversal order (breadth-first, nearest hops first).
        ids: Vec<String>,
    },
    /// An `INFO`.
    Catalog {
        /// The estate's live catalog.
        info: serde_json::Value,
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
        Statement::Define(Define::Field {
            field,
            collection,
            ty,
        }) => {
            estate.define_field(&collection, &field, ty.as_str())?;
            Ok(SqlOutcome::Defined {
                what: format!("field {collection}.{field} type {}", ty.as_str()),
            })
        }

        Statement::Remove(Remove::Field { field, collection }) => {
            estate.remove_field(&collection, &field)?;
            Ok(SqlOutcome::Removed {
                what: format!("field {collection}.{field}"),
                count: None,
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

        Statement::Relate(Relate { from, verb, to }) => {
            estate.relate(&from, &verb, &to)?;
            Ok(SqlOutcome::Related {
                edge: format!("{from} -{verb}-> {to}"),
            })
        }

        Statement::Traverse(t) => {
            let d = connxism::TraversalSpec::default();
            let spec = connxism::TraversalSpec {
                verbs: t.verbs,
                // The arrow IS the direction; a default would silently walk a
                // way the caller did not ask for.
                outbound: matches!(t.dir, Direction::Out | Direction::Both),
                inbound: matches!(t.dir, Direction::In | Direction::Both),
                // Bounds are clamped, not trusted: this statement can arrive
                // from a remote peer, and an unbounded walk is a DoS.
                depth: t.depth.unwrap_or(d.depth).min(64),
                limit: t.limit.unwrap_or(d.limit).min(10_000),
            };
            let refs: Vec<&str> = t.start.iter().map(String::as_str).collect();
            Ok(SqlOutcome::Traversed {
                ids: estate.traverse(&refs, &spec)?,
            })
        }

        Statement::Info(_) => Ok(SqlOutcome::Catalog {
            info: serde_json::to_value(estate.info())
                .map_err(|e| RroError::msg(format!("serialize estate info: {e}")))?,
        }),

        // LIVE is a STREAM, not a value. It cannot be a SqlOutcome without
        // pretending a subscription is a reply — the a2a `live` verb serves it
        // (Client::live), streaming change frames from the LIVE cursor.
        Statement::Live(_) => Err(RroError::msg(
            "LIVE opens a push stream, not a one-shot reply — send it to the a2a \
             `live` verb (Client::live), which streams change frames from the \
             LIVE cursor",
        )),
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
        assert!(
            !m.contains_key("rank"),
            "CONTENT replaces; `rank` must be gone"
        );
    }

    #[tokio::test]
    async fn delete_payload_keys_removes_only_those_keys() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "DELETE PAYLOAD doc1 (rank)").await.unwrap();
        let m = estate.recall().doc("doc1").await.unwrap().unwrap().metadata;
        assert!(!m.contains_key("rank"));
        assert!(m.contains_key("team"), "only `rank` was named");
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
            estate
                .payload_indexes()
                .unwrap()
                .contains(&"team".to_string()),
            "the index must actually exist afterwards, not just be reported"
        );
    }

    #[tokio::test]
    async fn define_field_reaches_the_estate_and_binds_the_type() {
        let (_d, estate) = estate_with_a_doc().await;
        let out = run(&estate, "DEFINE FIELD price ON products TYPE float")
            .await
            .unwrap();
        assert_eq!(
            out,
            SqlOutcome::Defined {
                what: "field products.price type float".into()
            }
        );
        // The constraint actually exists in the estate's schema afterwards.
        let schema = estate.schema().unwrap();
        assert_eq!(
            schema
                .get("products")
                .and_then(|f| f.get("price"))
                .map(String::as_str),
            Some("float")
        );

        // REMOVE FIELD via RRQL clears it.
        run(&estate, "REMOVE FIELD price ON products")
            .await
            .unwrap();
        assert!(estate.schema().unwrap().is_empty());
    }

    #[tokio::test]
    async fn define_and_remove_alias_round_trip() {
        let (_d, estate) = estate_with_a_doc().await;
        run(&estate, "DEFINE ALIAS current FOR alpha")
            .await
            .unwrap();
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

    // ---- B3: RELATE / TRAVERSE / LIVE / INFO ----------------------------

    async fn graph() -> (tempfile::TempDir, Arc<connxism::Estate>) {
        let dir = tempfile::tempdir().unwrap();
        let estate = Arc::new(connxism::Estate::open(dir.path(), "g").unwrap());
        let recs: Vec<VectorRecord> = (0..5)
            .map(|i| {
                VectorRecord::new(
                    format!("n{i}"),
                    rro_core::Embedding(vec![i as f32, 1.0]),
                    format!("node {i}"),
                )
            })
            .collect();
        estate.recall().upsert(recs).await.unwrap();
        (dir, estate)
    }

    /// RRQL's walk must equal the direct estate walk. If they diverge, the
    /// language is answering a different question than the engine.
    #[tokio::test]
    async fn relate_then_traverse_equals_the_direct_walk() {
        let (_d, estate) = graph().await;
        run(&estate, "RELATE n0 -> cites -> n1").await.unwrap();
        run(&estate, "RELATE n1 -> cites -> n2").await.unwrap();
        run(&estate, "RELATE n2 -> cites -> n3").await.unwrap();

        let out = run(&estate, "TRAVERSE n0 -> cites -> DEPTH 3")
            .await
            .unwrap();
        let spec = connxism::TraversalSpec {
            verbs: vec!["cites".into()],
            outbound: true,
            inbound: false,
            depth: 3,
            limit: 10_000,
        };
        let direct = estate.traverse(&["n0"], &spec).unwrap();
        assert_eq!(
            out,
            SqlOutcome::Traversed {
                ids: direct.clone()
            }
        );
        assert!(direct.contains(&"n3".to_string()), "3 hops reaches n3");
    }

    /// The arrow IS the direction. `<-` must walk the edge backwards, and a
    /// default direction would silently walk a way nobody asked for.
    #[tokio::test]
    async fn the_arrow_decides_direction() {
        let (_d, estate) = graph().await;
        run(&estate, "RELATE n0 -> cites -> n1").await.unwrap();

        // outbound from n0 reaches n1
        match run(&estate, "TRAVERSE n0 -> cites -> DEPTH 1")
            .await
            .unwrap()
        {
            SqlOutcome::Traversed { ids } => assert!(ids.contains(&"n1".to_string())),
            o => panic!("{o:?}"),
        }
        // inbound from n0 reaches nothing (the edge points away)
        match run(&estate, "TRAVERSE n0 <- cites <- DEPTH 1")
            .await
            .unwrap()
        {
            SqlOutcome::Traversed { ids } => {
                assert!(
                    !ids.contains(&"n1".to_string()),
                    "`<-` must not walk outbound edges"
                );
            }
            o => panic!("{o:?}"),
        }
        // inbound from n1 reaches n0
        match run(&estate, "TRAVERSE n1 <- cites <- DEPTH 1")
            .await
            .unwrap()
        {
            SqlOutcome::Traversed { ids } => assert!(ids.contains(&"n0".to_string())),
            o => panic!("{o:?}"),
        }
    }

    #[tokio::test]
    async fn depth_is_honored_not_ignored() {
        let (_d, estate) = graph().await;
        run(&estate, "RELATE n0 -> cites -> n1").await.unwrap();
        run(&estate, "RELATE n1 -> cites -> n2").await.unwrap();
        run(&estate, "RELATE n2 -> cites -> n3").await.unwrap();
        match run(&estate, "TRAVERSE n0 -> cites -> DEPTH 1")
            .await
            .unwrap()
        {
            SqlOutcome::Traversed { ids } => {
                assert!(
                    !ids.contains(&"n3".to_string()),
                    "DEPTH 1 must not reach a 3-hop node"
                );
            }
            o => panic!("{o:?}"),
        }
    }

    /// A statement can arrive from a remote peer. An unbounded walk is a DoS, so
    /// the bounds are CLAMPED rather than trusted.
    #[tokio::test]
    async fn absurd_bounds_are_clamped_not_trusted() {
        let (_d, estate) = graph().await;
        run(&estate, "RELATE n0 -> cites -> n1").await.unwrap();
        // depth 9999 -> clamped to 64; limit 999999 -> clamped to 10k. The point
        // is that it answers instead of walking forever.
        assert!(
            run(&estate, "TRAVERSE n0 -> cites -> DEPTH 9999 LIMIT 999999")
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn info_returns_the_live_catalog() {
        let (_d, estate) = graph().await;
        match run(&estate, "INFO").await.unwrap() {
            SqlOutcome::Catalog { info } => {
                assert!(info.is_object(), "INFO must return the estate's catalog");
            }
            o => panic!("{o:?}"),
        }
    }

    /// LIVE is a subscription. Returning it as a one-shot value would be
    /// pretending a stream is a reply — the one-shot path refuses and names the
    /// `live` stream verb (which the a2a handler now serves).
    #[tokio::test]
    async fn live_is_refused_here_and_points_at_the_live_stream() {
        let (_d, estate) = graph().await;
        let e = run(&estate, "LIVE").await.unwrap_err();
        assert!(e.to_string().contains("live"), "names the right seam: {e}");
    }
}
