//! Sprint 19 gate: the sprint 12–18 surface over live TCP — every new verb
//! answers through `Client` with results equal to the local calls, and the
//! named/sparse paths ride the existing `query` verb via `using`/`sparse`.

use std::sync::Arc;

use rrf_client::Client;
use rrf_core::{Embedding, EstateQuery, Recall, SparseVector, VectorRecord};
use rrf_flow::{FlowNode, ReasonReadyFlow};
use rrf_net::tcp;

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
    let flow = Arc::new(ReasonReadyFlow::default_engine());
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
