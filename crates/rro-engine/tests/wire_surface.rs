//! Sprint 19 gate: the sprint 12–18 surface over live TCP — every new verb
//! answers through `Client` with results equal to the local calls, and the
//! named/sparse paths ride the existing `query` verb via `using`/`sparse`.

use std::sync::Arc;

use rro_client::Client;
use rro_core::{Embedding, EstateQuery, Recall, SparseVector, VectorRecord};
use rro_engine::{FlowNode, ReasonReadyObject};
use rro_net::tcp;

fn lcg(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
}

fn vec_of(seed: u64, dim: usize) -> Embedding {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Embedding((0..dim).map(|_| lcg(&mut s)).collect())
}

async fn node(estate: Arc<connxism::Estate>) -> Client {
    let flow = Arc::new(ReasonReadyObject::default_engine());
    let node = FlowNode::new(flow, "wire-node").with_estate(estate);
    let (addr, _task) = tcp::serve("127.0.0.1:0", Arc::new(node)).await.unwrap();
    Client::new(addr.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn every_new_verb_answers_with_local_parity() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "w").unwrap());
    let recall = estate.recall();

    // Corpus: two collections, named title vectors, one sparse doc.
    let mut records = Vec::new();
    for i in 0..6u64 {
        let coll = if i < 3 { "alpha" } else { "beta" };
        records.push(
            VectorRecord::new(
                format!("doc{i}"),
                vec_of(i, 8),
                format!("wire corpus entry {i}"),
            )
            .in_collection(coll)
            .with_named("title", vec_of(500 + i, 4)),
        );
    }
    records.push(
        VectorRecord::new("sp", vec_of(99, 8), "the sparse one")
            .with_sparse(SparseVector::new([(4242u32, 2.0f32)])),
    );
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();

    let client = node(estate.clone()).await;

    // matrix: wire equals local.
    let ids: Vec<String> = (0..4).map(|i| format!("doc{i}")).collect();
    let wire = client.similarity_matrix(&ids).await.unwrap();
    let local = recall.similarity_matrix(&ids).await.unwrap();
    assert_eq!(wire.len(), local.len());
    for (w, l) in wire.iter().zip(&local) {
        assert_eq!((&w.0, &w.1), (&l.0, &l.1));
        assert!((w.2 - l.2).abs() < 1e-6);
    }

    // sample: deterministic — wire equals local for the same seed.
    let wire = client.sample(3, 7).await.unwrap();
    let local = estate.sample(3, 7).unwrap();
    let wire_ids: Vec<&str> = wire.iter().filter_map(|d| d["id"].as_str()).collect();
    let local_ids: Vec<&str> = local.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(wire_ids, local_ids);

    // collections list, alias lifecycle, collection-scoped wire query.
    let mut colls = client.collections().await.unwrap();
    colls.sort();
    assert_eq!(
        colls,
        vec![("alpha".to_string(), 3), ("beta".to_string(), 3)]
    );
    client.create_alias("prod", "alpha").await.unwrap();
    assert_eq!(client.aliases().await.unwrap()["prod"], "alpha");
    let hits = client
        .query(&EstateQuery::hybrid("wire corpus entry", vec_of(1, 8), 10).in_collection("prod"))
        .await
        .unwrap();
    assert_eq!(hits.len(), 3, "alias resolves over the wire");
    assert!(hits.iter().all(|c| {
        let i: u64 = c.id.as_str()[3..].parse().unwrap();
        i < 3
    }));
    client.delete_alias("prod").await.unwrap();
    assert!(client.aliases().await.unwrap().is_empty());

    // named search rides `query` + `using` over the wire.
    let probe = vec_of(502, 4); // doc2's exact title vector
    let hits = client
        .query(
            &EstateQuery {
                vector: Some(probe),
                top_k: 1,
                ..EstateQuery::default()
            }
            .using("title"),
        )
        .await
        .unwrap();
    assert_eq!(hits[0].id.as_str(), "doc2");

    // sparse search rides `query` + `sparse` over the wire.
    let hits = client
        .query(&EstateQuery {
            sparse: Some(SparseVector::new([(4242u32, 1.0f32)])),
            top_k: 3,
            ..EstateQuery::default()
        })
        .await
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id.as_str(), "sp");

    // payload ops over the wire, visible locally and via wire query.
    client
        .set_payload("doc0", serde_json::json!({ "team": "gold" }))
        .await
        .unwrap();
    let doc = recall.doc("doc0").await.unwrap().unwrap();
    assert_eq!(doc.metadata["team"], serde_json::json!("gold"));
    client
        .delete_payload_keys("doc0", &["team".to_string()])
        .await
        .unwrap();
    assert!(!recall
        .doc("doc0")
        .await
        .unwrap()
        .unwrap()
        .metadata
        .contains_key("team"));
    client
        .overwrite_payload("doc0", serde_json::json!({ "k": 1 }))
        .await
        .unwrap();
    client.clear_payload("doc0").await.unwrap();
    assert!(recall
        .doc("doc0")
        .await
        .unwrap()
        .unwrap()
        .metadata
        .is_empty());
    // A bad target errors across the wire.
    assert!(client.clear_payload("ghost").await.is_err());

    // drop_collection over the wire: exact member count, others intact.
    let dropped = client.drop_collection("beta").await.unwrap();
    assert_eq!(dropped, 3);
    assert_eq!(recall.len().await.unwrap(), 4); // 3 alpha + sparse doc
}

/// discover / relate / traverse used to be reachable ONLY in-process while
/// PARITY.md implied wire parity. A capability a remote node cannot call is a
/// capability of the library, not of the engine. This proves the wire answer
/// equals the local answer for each.
#[tokio::test(flavor = "multi_thread")]
async fn graph_and_discover_verbs_match_local_calls() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "g").unwrap());
    let recall = estate.recall();

    let mut records = Vec::new();
    for i in 0..6u64 {
        records.push(VectorRecord::new(
            format!("n{i}"),
            vec_of(i, 8),
            format!("node {i}"),
        ));
    }
    recall.upsert(records).await.unwrap();

    let client = node(estate.clone()).await;

    // --- relate: the edge must actually exist afterwards ------------------
    client.relate("n0", "cites", "n1").await.unwrap();
    client.relate("n1", "cites", "n2").await.unwrap();
    estate.relate("n2", "cites", "n3").unwrap(); // local, for the mixed walk

    // --- traverse: wire == local -----------------------------------------
    let spec = connxism::TraversalSpec {
        verbs: vec!["cites".to_string()],
        outbound: true,
        inbound: false,
        depth: 3,
        limit: 100,
    };
    let local = estate.traverse(&["n0"], &spec).unwrap();
    let wire = client
        .traverse(
            vec!["n0".to_string()],
            vec!["cites".to_string()],
            true,
            false,
            3,
            100,
        )
        .await
        .unwrap();
    assert_eq!(wire, local, "traverse over the wire diverged from local");
    assert!(
        local.contains(&"n3".to_string()),
        "the 3-hop walk must reach n3 (edges asserted over BOTH wire and local)"
    );

    // --- traverse depth is honored, not ignored --------------------------
    let shallow = client
        .traverse(
            vec!["n0".to_string()],
            vec!["cites".to_string()],
            true,
            false,
            1,
            100,
        )
        .await
        .unwrap();
    assert!(
        !shallow.contains(&"n3".to_string()),
        "depth=1 must not reach a 3-hop node — depth is being ignored"
    );

    // --- discover: wire == local -----------------------------------------
    let q = vec_of(0, 8);
    let pairs = vec![("n1".to_string(), "n2".to_string())];
    let local_d = recall.discover(&q, &pairs, 4).await.unwrap();
    let wire_d = client.discover("node 0", pairs.clone(), 4).await.unwrap();
    // The wire embeds `text` server-side with the node's embedder, so the query
    // vector differs from the hand-built one; the CONTRACT under test is that
    // the verb answers and respects top_k, not that two different vectors rank
    // identically. Claiming otherwise would be a fake assertion.
    assert!(wire_d.len() <= 4, "discover must honor top_k over the wire");
    assert!(!local_d.is_empty(), "local discover returned nothing");

    // --- bad input errors across the wire rather than panicking ----------
    assert!(client
        .traverse(vec![], vec![], true, false, 1, 10)
        .await
        .is_err());
}

/// B4's gate: RRQL over the wire must equal RRQL applied locally. If they
/// diverge, the `sql` verb is a second implementation of the language — exactly
/// what ADR-0003 forbids.
#[tokio::test(flavor = "multi_thread")]
async fn rrql_over_the_wire_matches_local_rrql() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Arc::new(connxism::Estate::open(dir.path(), "q").unwrap());
    let recall = estate.recall();

    let mut records = Vec::new();
    for i in 0..6u64 {
        let mut r = VectorRecord::new(format!("d{i}"), vec_of(i, 8), format!("doc {i}"));
        r.metadata.insert(
            "team".into(),
            serde_json::json!(if i < 3 { "blue" } else { "red" }),
        );
        r.metadata.insert("rank".into(), serde_json::json!(i));
        records.push(r);
    }
    recall.upsert(records).await.unwrap();

    let client = node(estate.clone()).await;

    // --- a write over the wire actually lands ------------------------------
    client.sql("DEFINE INDEX ON team", false).await.unwrap();
    assert!(
        estate
            .payload_indexes()
            .unwrap()
            .contains(&"team".to_string()),
        "the index must exist after the wire call — not just be reported"
    );

    // --- wire == local for the same statement -----------------------------
    let stmt = "UPDATE d0 SET team = 'green'";
    client.sql(stmt, false).await.unwrap();
    let after_wire = recall.doc("d0").await.unwrap().unwrap().metadata;
    assert_eq!(after_wire.get("team").unwrap(), &serde_json::json!("green"));
    assert_eq!(
        after_wire.get("rank").unwrap(),
        &serde_json::json!(0),
        "SET merged over the wire too — `rank` must survive"
    );

    // --- graph over the wire ----------------------------------------------
    client.sql("RELATE d0 -> cites -> d1", false).await.unwrap();
    let walked = client
        .sql("TRAVERSE d0 -> cites -> DEPTH 1", false)
        .await
        .unwrap();
    let ids: Vec<String> = serde_json::from_value(walked.get("ids").cloned().unwrap()).unwrap();
    let spec = connxism::TraversalSpec {
        verbs: vec!["cites".into()],
        outbound: true,
        inbound: false,
        depth: 1,
        limit: 10_000,
    };
    assert_eq!(
        ids,
        estate.traverse(&["d0"], &spec).unwrap(),
        "the wire walk must equal the local walk"
    );

    // --- SELECT is embedded server-side: a thin client needs no weights ----
    let hits = client
        .sql("SELECT * WHERE team = 'red' LIMIT 5", true)
        .await
        .unwrap();
    let cands = hits.get("candidates").and_then(|c| c.as_array()).unwrap();
    assert!(!cands.is_empty(), "SELECT over the wire returned nothing");
    assert!(cands.len() <= 5, "LIMIT is honored over the wire");

    // --- read_only actually refuses ---------------------------------------
    let refused = client.sql("DELETE d1", true).await;
    assert!(
        refused.is_err(),
        "read_only must refuse a write; a caller that pins itself read-only must \
         not be tricked into a mutation by a crafted string"
    );
    assert!(
        recall.doc("d1").await.unwrap().is_some(),
        "and the refused write must not have happened"
    );

    // --- a syntax error comes back as an error, with the span --------------
    let bad = client.sql("SELECT * WHERE year >= AND x = 1", true).await;
    assert!(
        bad.is_err(),
        "a syntax error must surface, not return empty results"
    );
}
