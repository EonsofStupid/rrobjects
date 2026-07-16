//! Sprint 21 gates: geo filters answered index-first — Z-order range scan
//! plus exact doc-level re-check — equal to brute-force haversine/box
//! truth on a seeded city grid, wired through the query plane, retracting
//! on move.

use connxism::{Estate, EstateQuery};
use rrf_core::{Condition, Embedding, Filter, Recall, VectorRecord};

/// A 15×15 grid over greater NYC-ish: lat 40.5..41.2, lon -74.3..-73.6.
fn grid() -> Vec<VectorRecord> {
    let mut out = Vec::new();
    for i in 0..15 {
        for j in 0..15 {
            let lat = 40.5 + i as f64 * 0.05;
            let lon = -74.3 + j as f64 * 0.05;
            let mut r = VectorRecord::new(
                format!("p{i:02}_{j:02}"),
                Embedding(vec![0.3, 0.2, 0.1, 0.4]),
                format!("geo corpus point {i},{j}"),
            );
            r.metadata
                .insert("loc".into(), serde_json::json!({"lat": lat, "lon": lon}));
            out.push(r);
        }
    }
    out
}

async fn seed(estate: &Estate) -> connxism::ConnXRecall {
    estate.create_payload_index("loc").unwrap();
    let recall = estate.recall();
    recall.upsert(grid()).await.unwrap();
    recall.quiesce().await.unwrap();
    recall
}

/// Brute-force truth: every grid doc whose metadata satisfies `c`.
async fn truth(recall: &connxism::ConnXRecall, c: &Condition) -> Vec<String> {
    let mut out = Vec::new();
    for i in 0..15 {
        for j in 0..15 {
            let id = format!("p{i:02}_{j:02}");
            let doc = recall.doc(&id).await.unwrap().unwrap();
            if c.matches(&doc.metadata) {
                out.push(id);
            }
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn geo_box_and_radius_equal_brute_force() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "geo").unwrap();
    let recall = seed(&estate).await;

    // Boxes of several sizes and positions.
    for (lat_min, lon_min, lat_max, lon_max) in [
        (40.6, -74.2, 40.8, -74.0),
        (40.5, -74.3, 41.2, -73.6),     // the whole grid
        (40.71, -74.29, 40.74, -74.26), // tiny: at most one point
        (39.0, -75.0, 39.5, -74.5),     // disjoint: empty
    ] {
        let c = Condition::geo_box("loc", lat_min, lon_min, lat_max, lon_max);
        let filter = Filter::default().must(c.clone());
        let idx = estate.ids_where(&filter).unwrap().expect("index-first");
        let want = truth(&recall, &c).await;
        assert_eq!(
            idx, want,
            "box ({lat_min},{lon_min})..({lat_max},{lon_max})"
        );
    }

    // Radii around several centers — exact haversine truth.
    for (lat, lon, radius_m) in [
        (40.85, -73.95, 10_000.0),
        (40.85, -73.95, 3_000.0),
        (40.5, -74.3, 25_000.0), // corner center
        (40.85, -73.95, 10.0),   // nothing within 10 m of an off-grid point
    ] {
        let c = Condition::geo_radius("loc", lat, lon, radius_m);
        let filter = Filter::default().must(c.clone());
        let idx = estate.ids_where(&filter).unwrap().expect("index-first");
        let want = truth(&recall, &c).await;
        assert_eq!(idx, want, "radius {radius_m}m @ ({lat},{lon})");
        if radius_m > 1_000.0 {
            assert!(!want.is_empty(), "sanity: non-trivial truth set");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn geo_rides_the_query_plane_and_retracts_on_move() {
    let dir = tempfile::tempdir().unwrap();
    let estate = Estate::open(dir.path(), "geoq").unwrap();
    let recall = seed(&estate).await;

    let around = |lat: f64, lon: f64, m: f64| {
        Filter::default().must(Condition::geo_radius("loc", lat, lon, m))
    };

    // Query plane: filtered hybrid returns exactly the truth set.
    let c = Condition::geo_radius("loc", 40.85, -73.95, 5_000.0);
    let want = truth(&recall, &c).await;
    let hits = recall
        .query(
            EstateQuery::hybrid("geo corpus point", Embedding(vec![0.3, 0.2, 0.1, 0.4]), 250)
                .filtered(around(40.85, -73.95, 5_000.0)),
        )
        .await
        .unwrap();
    let mut got: Vec<String> = hits.iter().map(|c| c.id.as_str().to_string()).collect();
    got.sort();
    assert_eq!(got, want);
    assert!(!want.is_empty());

    // Move a matching point far away: it leaves the radius exactly.
    let moved = &want[0];
    let mut r = VectorRecord::new(
        moved.clone(),
        Embedding(vec![0.3, 0.2, 0.1, 0.4]),
        "geo corpus point moved",
    );
    r.metadata.insert(
        "loc".into(),
        serde_json::json!({"lat": -33.87, "lon": 151.21}),
    );
    recall.upsert(vec![r]).await.unwrap();

    let after = estate
        .ids_where(&Filter::default().must(c.clone()))
        .unwrap()
        .unwrap();
    assert!(
        !after.contains(moved),
        "moved point retracted from the index"
    );
    assert_eq!(after.len(), want.len() - 1);

    // …and it is findable at its new home (Sydney-ish).
    let sydney = estate
        .ids_where(&Filter::default().must(Condition::geo_radius("loc", -33.87, 151.21, 1_000.0)))
        .unwrap()
        .unwrap();
    assert_eq!(sydney, vec![moved.clone()]);
}
