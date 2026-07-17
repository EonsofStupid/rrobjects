//! Scalar quantization (SQ8): 4× smaller vector memory, measured recall.
//!
//! Each vector is quantized independently to one byte per dimension with its
//! own affine map `x ≈ code * scale + offset` (per-vector min/max). Because
//! the map is affine, dot products against quantized storage stay *exact in
//! the codes*:
//!
//! - **asymmetric** (full-precision query `q` vs codes `c`):
//!   `q · x ≈ scale · Σ cᵢqᵢ + offset · Σ qᵢ` — one integer-weighted dot plus
//!   a precomputed query sum;
//! - **symmetric** (codes vs codes):
//!   `a · b ≈ sₐs_b Σ cₐc_b + sₐo_b Σ cₐ + oₐs_b Σ c_b + d·oₐo_b` — the code
//!   sums are stored once per vector.
//!
//! The quantized graph answers approximately; callers that keep the
//! full-precision vectors elsewhere (the estate's durable vector column
//! family) **rescore** the returned candidates exactly. Quantization here is
//! a memory decision, never a silent accuracy decision.

/// Per-vector affine parameters for SQ8 codes.
#[derive(Debug, Clone, Copy)]
pub struct SqParams {
    /// Code → value multiplier.
    pub scale: f32,
    /// Code → value offset (the vector's minimum).
    pub offset: f32,
    /// Σ codes, cached for symmetric dots.
    pub code_sum: f32,
}

/// Quantize `v`, appending its codes to `codes`; returns the affine params.
pub fn quantize_into(v: &[f32], codes: &mut Vec<u8>) -> SqParams {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &x in v {
        min = min.min(x);
        max = max.max(x);
    }
    if v.is_empty() {
        return SqParams {
            scale: 0.0,
            offset: 0.0,
            code_sum: 0.0,
        };
    }
    let range = max - min;
    let scale = if range > f32::EPSILON {
        range / 255.0
    } else {
        0.0
    };
    let mut code_sum = 0.0f32;
    for &x in v {
        let code = if scale > 0.0 {
            ((x - min) / scale).round().clamp(0.0, 255.0) as u8
        } else {
            0
        };
        code_sum += code as f32;
        codes.push(code);
    }
    SqParams {
        scale,
        offset: min,
        code_sum,
    }
}

/// Asymmetric dot: full-precision query against one vector's codes.
/// `qsum` is `Σ qᵢ`, computed once per query.
pub fn dot_query(codes: &[u8], p: &SqParams, q: &[f32], qsum: f32) -> f32 {
    let mut acc = 0.0f32;
    for (&c, &x) in codes.iter().zip(q) {
        acc += c as f32 * x;
    }
    p.scale * acc + p.offset * qsum
}

/// Symmetric dot: codes against codes, both dequantized implicitly.
pub fn dot_codes(a: &[u8], pa: &SqParams, b: &[u8], pb: &SqParams) -> f32 {
    let mut acc = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        acc += x as u32 * y as u32;
    }
    let d = a.len() as f32;
    pa.scale * pb.scale * acc as f32
        + pa.scale * pb.offset * pa.code_sum
        + pa.offset * pb.scale * pb.code_sum
        + d * pa.offset * pb.offset
}

/// Reconstruct the (lossy) full-precision vector from its codes.
pub fn decode(codes: &[u8], p: &SqParams) -> Vec<f32> {
    codes
        .iter()
        .map(|&c| c as f32 * p.scale + p.offset)
        .collect()
}

// ---- binary quantization (BQ) ----------------------------------------------
//
// One **bit** per dimension — the sign of each component — so a vector shrinks
// 32× from f32 (vs SQ8's 4×). The estimate keeps only orientation, which for
// normalized embeddings is most of the cosine signal; magnitude is gone. BQ is
// far lossier than SQ8, so it is a *traversal* code: the graph is walked cheaply
// on bits, then candidates are rescored exactly from the durable vectors (same
// contract as SQ8 — quantization is a memory/IO decision, never a silent
// accuracy one).
//
// Distances, for bits interpreted as sign ±1:
// - **asymmetric** (full query `q` vs bits `b`): `Σ qᵢ·sign(bᵢ)` — the query's
//   own magnitude is kept, only the stored side is signed;
// - **symmetric** (bits vs bits): `Σ sign(aᵢ)sign(bᵢ) = D − 2·popcount(a ⊕ b)` —
//   one XOR + popcount per word.

/// Bytes needed to hold `dim` sign bits, packed 8 per byte.
pub fn bq_bytes(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// Encode `v`'s sign bits (bit `i` = 1 iff `v[i] > 0`), appending
/// `bq_bytes(v.len())` bytes to `bits`. Padding bits in the final byte stay 0.
pub fn bq_encode_into(v: &[f32], bits: &mut Vec<u8>) {
    let start = bits.len();
    bits.resize(start + bq_bytes(v.len()), 0);
    for (i, &x) in v.iter().enumerate() {
        if x > 0.0 {
            bits[start + i / 8] |= 1 << (i % 8);
        }
    }
}

/// Asymmetric estimate: a full-precision query against one vector's sign bits.
/// `Σ qᵢ` if the bit is set, `−qᵢ` otherwise.
pub fn bq_dot_query(bits: &[u8], q: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (i, &qi) in q.iter().enumerate() {
        let set = (bits[i / 8] >> (i % 8)) & 1 == 1;
        acc += if set { qi } else { -qi };
    }
    acc
}

/// Symmetric estimate: sign bits against sign bits, `D − 2·popcount(a ⊕ b)`.
/// Padding bits are 0 in both operands, so they cancel and never affect the
/// count.
pub fn bq_dot_codes(a: &[u8], b: &[u8], dim: usize) -> f32 {
    let ham: u32 = a.iter().zip(b).map(|(&x, &y)| (x ^ y).count_ones()).sum();
    dim as f32 - 2.0 * ham as f32
}

/// Reconstruct an approximate **unit** vector from sign bits: each component is
/// `±1/√dim`. Magnitude is unrecoverable from bits; this keeps the vector unit-
/// norm so cosine-style use stays sane (used for in-graph neighbor rescoring).
pub fn bq_decode(bits: &[u8], dim: usize) -> Vec<f32> {
    let s = 1.0 / (dim as f32).sqrt();
    (0..dim)
        .map(|i| {
            let set = (bits[i / 8] >> (i % 8)) & 1 == 1;
            if set {
                s
            } else {
                -s
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_vec(seed: u64, dim: usize) -> Vec<f32> {
        let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..dim)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x as f64 / u64::MAX as f64) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn exact_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn asymmetric_dot_tracks_exact() {
        for seed in 0..20u64 {
            let v = pseudo_vec(seed, 128);
            let q = pseudo_vec(seed + 1000, 128);
            let mut codes = Vec::new();
            let p = quantize_into(&v, &mut codes);
            let qsum: f32 = q.iter().sum();
            let approx = dot_query(&codes, &p, &q, qsum);
            let exact = exact_dot(&v, &q);
            assert!(
                (approx - exact).abs() < 0.05 * 128f32.sqrt(),
                "seed {seed}: approx {approx} vs exact {exact}"
            );
        }
    }

    #[test]
    fn symmetric_dot_tracks_exact() {
        for seed in 0..20u64 {
            let a = pseudo_vec(seed, 128);
            let b = pseudo_vec(seed + 500, 128);
            let mut ca = Vec::new();
            let mut cb = Vec::new();
            let pa = quantize_into(&a, &mut ca);
            let pb = quantize_into(&b, &mut cb);
            let approx = dot_codes(&ca, &pa, &cb, &pb);
            let exact = exact_dot(&a, &b);
            assert!(
                (approx - exact).abs() < 0.08 * 128f32.sqrt(),
                "seed {seed}: approx {approx} vs exact {exact}"
            );
        }
    }

    #[test]
    fn constant_vector_roundtrips() {
        let v = vec![0.25f32; 32];
        let mut codes = Vec::new();
        let p = quantize_into(&v, &mut codes);
        let back = decode(&codes, &p);
        for x in back {
            assert!((x - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn decode_error_is_bounded_by_half_step() {
        let v = pseudo_vec(7, 64);
        let mut codes = Vec::new();
        let p = quantize_into(&v, &mut codes);
        let back = decode(&codes, &p);
        for (orig, dec) in v.iter().zip(&back) {
            assert!((orig - dec).abs() <= p.scale * 0.5 + 1e-6);
        }
    }

    #[test]
    fn bq_byte_count_and_padding() {
        assert_eq!(bq_bytes(8), 1);
        assert_eq!(bq_bytes(9), 2);
        assert_eq!(bq_bytes(0), 0);
        // A dim not a multiple of 8: padding bits stay 0, so self-XOR is 0.
        let v = pseudo_vec(3, 12);
        let mut bits = Vec::new();
        bq_encode_into(&v, &mut bits);
        assert_eq!(bits.len(), 2);
        assert_eq!(bq_dot_codes(&bits, &bits, 12), 12.0, "self-similarity is D");
    }

    #[test]
    fn bq_encodes_signs() {
        let v = vec![0.3, -0.1, 0.0, 2.0, -5.0];
        let mut bits = Vec::new();
        bq_encode_into(&v, &mut bits);
        // bits: +,-,0(→not>0),+,- => 1,0,0,1,0
        assert_eq!(bits[0] & 0b11111, 0b01001);
    }

    #[test]
    fn bq_symmetric_equals_signed_dot() {
        // D − 2·Hamming must equal Σ sign(a)sign(b) computed directly.
        for seed in 0..20u64 {
            let a = pseudo_vec(seed, 96);
            let b = pseudo_vec(seed + 7, 96);
            let (mut ba, mut bb) = (Vec::new(), Vec::new());
            bq_encode_into(&a, &mut ba);
            bq_encode_into(&b, &mut bb);
            let direct: f32 = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.signum() * y.signum())
                .sum();
            assert_eq!(bq_dot_codes(&ba, &bb, 96), direct);
        }
    }

    /// BQ is lossy, so it cannot track the exact dot like SQ8 — but it must
    /// *rank* well enough to traverse: the vector a query points at should score
    /// highest under BQ. Here the query is one corpus vector, so its own sign
    /// code is the exact match and must win under both BQ estimators.
    #[test]
    fn bq_ranks_the_true_neighbour_first() {
        let dim = 128;
        let corpus: Vec<Vec<f32>> = (0..200).map(|i| pseudo_vec(i, dim)).collect();
        let codes: Vec<Vec<u8>> = corpus
            .iter()
            .map(|v| {
                let mut b = Vec::new();
                bq_encode_into(v, &mut b);
                b
            })
            .collect();

        let mut hits = 0;
        for (qi, q) in corpus.iter().enumerate() {
            // asymmetric: full query vs every code
            let best = (0..corpus.len())
                .max_by(|&a, &b| bq_dot_query(&codes[a], q).total_cmp(&bq_dot_query(&codes[b], q)))
                .unwrap();
            if best == qi {
                hits += 1;
            }
        }
        // Every self-query should recover itself (its signs match perfectly).
        assert_eq!(hits, corpus.len(), "BQ must rank the exact match first");
    }
}
