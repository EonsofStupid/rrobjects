//! Sprint 24 gates: prefetch pipelines (union → exact outer rescore,
//! depth-capped, wire-shaped) and index-first facets equal to the doc
//! scan, with honest fallback on non-reconstructible tags.

use connxism::{Estate, EstateQuery};
use rrf_core::{maxsim, Embedding, Recall, SparseVector, VectorRecord};

fn lcg(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    ((*s as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
}

fn vec_of(seed: u64) -> Embedding {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    Embedding((0..16).map(|_| lcg(&mut s)).collect())
}

async fn seed(estate: &Estate) -> connxism::ConnXRecall {
    let recall = estate.recall();
    let mut records = Vec::new();
    for i in 0..60u64 {
        let mut r = VectorRecord::new(
            format!("doc{i:02}"),
            vec_of(i),
            format!("prefetch corpus entry {i}"),
        )
        .with_multi(vec![vec_of(3000 + i), vec_of(4000 + i)]);
        if i % 5 == 0 {
            r = r.with_sparse(SparseVector::new([((i % 7) as u32 + 100, 1.0f32)]));
        }
        r.metadata
            .insert("team".into(), serde_json::json!(format!("team{}", i % 4)));
        r.metadata
            .insert("priority".into(), serde_json::json!((i % 3) as f64));
        r.metadata
            .insert("flag".into(), serde_json::json!(i % 2 == 0));
        records.push(r);
    }
    recall.upsert(records).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

#[tokio::test(flavor = "multi_thread")]
async fn prefetch_pipeline_equals_hand_built_two_stage() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "pf").unwrap();
    let recall = seed(&estate).await;

    let qv = vec_of(999);
    let qtokens = vec![vec_of(3000 + 17)]; // doc17's first token vector

    // Hand-built: dense top-20 ids, then MaxSim over exactly that set.
    let dense = recall
        .query(EstateQuery {
            vector: Some(qv.clone()),
            top_k: 20,
            ..EstateQuery::default()
        })
        .await
        .unwrap();
    let scope: Vec<String> = dense.iter().map(|c| c.id.as_str().to_string()).collect();
    let hand = recall
        .query(
            EstateQuery {
                vector: Some(qv.clone()),
                top_k: 5,
                ..EstateQuery::default()
            }
            .within(scope.clone())
            .multi_query(qtokens.clone()),
        )
        .await
        .unwrap();

    // Prefetch pipeline: same shape, one query.
    let piped = recall
        .query(
            EstateQuery {
                vector: Some(qv.clone()),
                top_k: 5,
                ..EstateQuery::default()
            }
            .prefetch(
                EstateQuery {
                    vector: Some(qv.clone()),
                    top_k: 20,
                    ..EstateQuery::default()
                },
                20,
            )
            .multi_query(qtokens.clone()),
        )
        .await
        .unwrap();

    assert_eq!(
        piped.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        hand.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        "pipeline equals the hand-built two-stage"
    );
    // The MaxSim winner is ranked by true MaxSim.
    if scope.contains(&"doc17".to_string()) {
        assert_eq!(piped[0].id.as_str(), "doc17");
        let brute = maxsim(&qtokens, &[vec_of(3017), vec_of(4017)]);
        assert!((piped[0].score - brute).abs() < 1e-3);
    }

    // Union of two prefetches: sparse stage + dense stage.
    let sparse_probe = SparseVector::new([(101u32, 1.0f32)]);
    let sparse_ids: Vec<String> = recall
        .sparse_search(&sparse_probe, 10)
        .await
        .unwrap()
        .iter()
        .map(|c| c.id.as_str().to_string())
        .collect();
    assert!(!sparse_ids.is_empty());
    let unioned = recall
        .query(
            EstateQuery {
                vector: Some(qv.clone()),
                top_k: 60,
                ..EstateQuery::default()
            }
            .prefetch(
                EstateQuery {
                    sparse: Some(sparse_probe),
                    top_k: 10,
                    ..EstateQuery::default()
                },
                10,
            )
            .prefetch(
                EstateQuery {
                    vector: Some(qv.clone()),
                    top_k: 5,
                    ..EstateQuery::default()
                },
                5,
            ),
        )
        .await
        .unwrap();
    // Every result comes from one of the two stages; sparse-only docs made
    // it in even if dense-invisible.
    let got: std::collections::HashSet<&str> = unioned.iter().map(|c| c.id.as_str()).collect();
    for id in &sparse_ids {
        assert!(
            got.contains(id.as_str()),
            "sparse stage member {id} in union"
        );
    }

    // Depth cap: 3 levels of nesting errors.
    let level1 = EstateQuery {
        vector: Some(qv.clone()),
        top_k: 5,
        ..EstateQuery::default()
    };
    let level2 = EstateQuery {
        vector: Some(qv.clone()),
        top_k: 5,
        ..EstateQuery::default()
    }
    .prefetch(level1, 5);
    let level3 = EstateQuery {
        vector: Some(qv.clone()),
        top_k: 5,
        ..EstateQuery::default()
    }
    .prefetch(level2, 5);
    let level4 = EstateQuery {
        vector: Some(qv.clone()),
        top_k: 5,
        ..EstateQuery::default()
    }
    .prefetch(level3, 5);
    // Three levels of nesting execute (a pipeline)…
    assert!(recall.query(level4.clone()).await.is_ok());
    // …four do not.
    let level5 = EstateQuery {
        vector: Some(qv.clone()),
        top_k: 5,
        ..EstateQuery::default()
    }
    .prefetch(level4, 5);
    assert!(recall.query(level5).await.is_err(), "depth cap enforced");

    // Wire shape: serde roundtrip + old payloads parse without prefetch.
    let q = EstateQuery::hybrid("t", vec_of(1), 5).prefetch(EstateQuery::text("inner", 3), 3);
    let back: EstateQuery = serde_json::from_str(&serde_json::to_string(&q).unwrap()).unwrap();
    assert_eq!(back.prefetch.len(), 1);
    assert_eq!(back.prefetch[0].limit, 3);
    let old: EstateQuery = serde_json::from_str(r#"{"text":"x","vector":null,"top_k":3}"#).unwrap();
    assert!(old.prefetch.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn facet_index_first_equals_doc_scan() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "fx").unwrap();
    let _ = seed(&estate).await;

    // Truth before any index exists: the doc scan.
    let scan_team = estate.facet("team").unwrap();
    let scan_pri = estate.facet("priority").unwrap();
    let scan_flag = estate.facet("flag").unwrap();
    assert_eq!(scan_team.len(), 4);
    assert_eq!(scan_team["team0"], 15);

    // Index the fields: facet flips to the run-length index path — equal.
    estate.create_payload_index("team").unwrap();
    estate.create_payload_index("priority").unwrap();
    estate.create_payload_index("flag").unwrap();
    assert_eq!(estate.facet("team").unwrap(), scan_team);
    assert_eq!(estate.facet("priority").unwrap(), scan_pri);
    assert_eq!(estate.facet("flag").unwrap(), scan_flag);

    // Distinct listing = facet keys.
    assert_eq!(
        estate.distinct("team").unwrap(),
        vec!["team0", "team1", "team2", "team3"]
    );

    // A datetime field: indexed rows carry the DT tag whose spelling the
    // key can't reconstruct — facet falls back to the doc scan, still exact.
    let recall = estate.recall();
    let mut r = VectorRecord::new("dated", vec_of(777), "dated entry");
    r.metadata
        .insert("created".into(), serde_json::json!("2026-07-16T00:00:00Z"));
    recall.upsert(vec![r]).await.unwrap();
    estate.create_payload_index("created").unwrap();
    let f = estate.facet("created").unwrap();
    assert_eq!(
        f["2026-07-16T00:00:00Z"], 1,
        "fallback keeps exact spelling"
    );
}
