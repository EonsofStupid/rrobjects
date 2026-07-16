//! Geospatial primitives: `{lat, lon}` metadata points, great-circle
//! distance, and the Z-order (Morton) encoding the geo payload index keys
//! use. Zero dependencies, authored from the underlying math.
//!
//! **Honest limits (v1):** boxes must not cross the antimeridian (±180°);
//! latitudes clamp to ±90. Radius pre-filters use a lat/lon bounding box,
//! so results are exact only because every index candidate is re-checked
//! with true haversine against stored metadata.

/// Mean Earth radius in meters (IUGG).
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Great-circle distance in meters between two `(lat, lon)` points
/// (degrees), by the haversine formula.
pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = (lat2 - lat1).to_radians();
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().atan2((1.0 - a).sqrt())
}

/// Extract a `{lat, lon}` point from a metadata value.
pub fn point_of(value: &serde_json::Value) -> Option<(f64, f64)> {
    let lat = value.get("lat")?.as_f64()?;
    let lon = value.get("lon")?.as_f64()?;
    if !(-90.0..=90.0).contains(&lat) || !(-180.0..=180.0).contains(&lon) {
        return None;
    }
    Some((lat, lon))
}

/// Bits per axis in the Morton encoding (52-bit codes: quantization steps
/// of ~2.7e-6° ≈ 0.3 m — the exact post-check absorbs the rounding).
pub const GEO_BITS: u32 = 26;

/// Quantize one coordinate into `GEO_BITS` bits over `[lo, hi]`.
fn quantize(x: f64, lo: f64, hi: f64) -> u64 {
    let max = (1u64 << GEO_BITS) - 1;
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    (t * max as f64).round() as u64
}

/// Interleave the low `GEO_BITS` bits of `a` (even positions) and `b`
/// (odd positions) into one Morton code. Monotone in each argument —
/// which is exactly why one `[z(min corner), z(max corner)]` range scan
/// covers every point of a box (plus false positives outside it).
fn interleave(a: u64, b: u64) -> u64 {
    let mut out = 0u64;
    for i in 0..GEO_BITS {
        out |= ((a >> i) & 1) << (2 * i);
        out |= ((b >> i) & 1) << (2 * i + 1);
    }
    out
}

/// The Morton (Z-order) code of a `(lat, lon)` point.
pub fn morton(lat: f64, lon: f64) -> u64 {
    interleave(quantize(lat, -90.0, 90.0), quantize(lon, -180.0, 180.0))
}

/// The lat/lon bounding box of a radius query: `±radius` meters around
/// the center, degrees per meter scaled by latitude for longitude.
/// Longitude clamps to ±180 (no antimeridian wrap in v1).
pub fn radius_bbox(lat: f64, lon: f64, radius_m: f64) -> ((f64, f64), (f64, f64)) {
    const M_PER_DEG_LAT: f64 = 111_320.0;
    let dlat = radius_m / M_PER_DEG_LAT;
    let dlon = radius_m / (M_PER_DEG_LAT * lat.to_radians().cos().max(1e-6));
    (
        ((lat - dlat).max(-90.0), (lon - dlon).max(-180.0)),
        ((lat + dlat).min(90.0), (lon + dlon).min(180.0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haversine_known_pairs() {
        // Paris ↔ London ≈ 343.5 km.
        let d = haversine_m(48.8566, 2.3522, 51.5074, -0.1278);
        assert!((d - 343_500.0).abs() < 2_000.0, "{d}");
        // Same point: zero.
        assert_eq!(haversine_m(40.0, -74.0, 40.0, -74.0), 0.0);
        // One degree of latitude ≈ 111.2 km anywhere.
        let d = haversine_m(10.0, 20.0, 11.0, 20.0);
        assert!((d - 111_195.0).abs() < 200.0, "{d}");
    }

    #[test]
    fn morton_is_monotone_per_axis() {
        // The property the index scan depends on: growing either
        // coordinate never shrinks the code.
        let mut s = 7u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 1600) as f64 / 10.0 - 80.0 // lat-ish range
        };
        for _ in 0..500 {
            let (lat, lon) = (rnd(), rnd() * 2.0);
            let (dlat, dlon) = (rnd().abs() * 0.1, rnd().abs() * 0.1);
            assert!(morton(lat, lon) <= morton((lat + dlat).min(90.0), lon));
            assert!(morton(lat, lon) <= morton(lat, (lon + dlon).min(180.0)));
        }
    }

    #[test]
    fn point_extraction_guards_ranges() {
        assert_eq!(
            point_of(&serde_json::json!({"lat": 40.7, "lon": -74.0})),
            Some((40.7, -74.0))
        );
        assert_eq!(
            point_of(&serde_json::json!({"lat": 91.0, "lon": 0.0})),
            None
        );
        assert_eq!(point_of(&serde_json::json!({"lat": 0.0})), None);
        assert_eq!(point_of(&serde_json::json!("nope")), None);
    }
}
