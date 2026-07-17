//! AST → `rro-core`'s typed query. The part that must not lie.
//!
//! `Filter` is **flat**: `must` (all hold) / `should` (any holds) / `must_not`
//! (none hold). The AST is a **tree** of AND/OR/NOT. Those are not the same
//! algebra, and the whole integrity of RRQL is in how that gap is handled:
//!
//! - `a AND b`      → both into `must`. Exact.
//! - `a OR b`       → both into `should`. Exact **only at the top level**.
//! - `NOT a`        → `a` into `must_not`. Exact for a single condition.
//! - `a AND (b OR c)` → **REJECTED.** `Filter` has no nesting, so there is no
//!   honest encoding. Approximating it would silently return wrong rows.
//!
//! Rejecting is the point. A query language that accepts a query and quietly
//! answers a *different* one is worse than a language that says "not supported":
//! the first produces confident wrong answers, the second produces a fix.
//! When `Condition` gains nesting, this lowers it; not before.

use rro_core::{Condition, EstateQuery, Filter};

use crate::ast::{CmpOp, Expr, Select, Value};
use crate::error::QlError;

/// Default `top_k` when a query omits `LIMIT`.
pub const DEFAULT_LIMIT: usize = 10;

/// Lower a parsed `SELECT` into an [`EstateQuery`].
///
/// # Errors
/// [`QlError`] if the expression cannot be represented by [`Filter`] exactly.
pub fn lower_select(s: Select) -> Result<EstateQuery, QlError> {
    let mut q = EstateQuery {
        top_k: s.limit.unwrap_or(DEFAULT_LIMIT),
        collection: s.from,
        ..Default::default()
    };
    if let Some(e) = s.where_ {
        let mut f = Filter::default();
        lower_expr(&e, &mut f, Slot::Must)?;
        q.dsl = Some(f);
    }
    Ok(q)
}

/// Which bucket a condition is being lowered into.
#[derive(Clone, Copy, PartialEq)]
enum Slot {
    Must,
    Should,
    MustNot,
}

fn lower_expr(e: &Expr, f: &mut Filter, slot: Slot) -> Result<(), QlError> {
    match e {
        Expr::And(l, r) => {
            // AND inside a `should` bucket would mean "any of (all of ...)",
            // which `Filter` cannot say.
            if slot != Slot::Must {
                return Err(unsupported("AND nested inside OR or NOT"));
            }
            lower_expr(l, f, Slot::Must)?;
            lower_expr(r, f, Slot::Must)
        }
        Expr::Or(l, r) => match slot {
            // OR is exact only when it is the WHOLE filter: `should` is a flat
            // list, so `a AND (b OR c)` has nowhere to live.
            Slot::Must => {
                if !f.must.is_empty() || !f.must_not.is_empty() {
                    return Err(unsupported(
                        "OR combined with other clauses (Filter's `should` is a flat \
                         list with no nesting)",
                    ));
                }
                lower_expr(l, f, Slot::Should)?;
                lower_expr(r, f, Slot::Should)
            }
            // `a OR b OR c` parses as Or(Or(a,b), c), so the inner OR arrives
            // here. A disjunction of disjunctions is still ONE flat disjunction
            // — `should` says it exactly, so flatten rather than refuse.
            Slot::Should => {
                lower_expr(l, f, Slot::Should)?;
                lower_expr(r, f, Slot::Should)
            }
            Slot::MustNot => Err(unsupported("OR inside NOT")),
        },
        Expr::Not(inner) => {
            if slot != Slot::Must {
                return Err(unsupported("NOT nested inside OR"));
            }
            match inner.as_ref() {
                Expr::And(..) | Expr::Or(..) | Expr::Not(..) => {
                    Err(unsupported("NOT applied to a compound expression"))
                }
                leaf => lower_expr(leaf, f, Slot::MustNot),
            }
        }
        leaf => {
            let c = leaf_condition(leaf)?;
            match slot {
                Slot::Must => f.must.push(c),
                Slot::Should => f.should.push(c),
                Slot::MustNot => f.must_not.push(c),
            }
            Ok(())
        }
    }
}

/// One leaf → one `Condition`.
fn leaf_condition(e: &Expr) -> Result<Condition, QlError> {
    Ok(match e {
        Expr::Eq { key, value } => Condition::Eq {
            key: key.clone(),
            value: value.to_json(),
        },
        // `a != v` has no Condition of its own; it is `must_not { Eq }`. The
        // caller cannot express that from a leaf slot, so reject rather than
        // pretend — `NOT a = v` is the supported spelling.
        Expr::Neq { .. } => {
            return Err(unsupported(
                "`!=` (write `NOT key = value`; Filter expresses inequality as must_not)",
            ))
        }
        Expr::In { key, values } => Condition::Any {
            key: key.clone(),
            values: values.iter().map(Value::to_json).collect(),
        },
        Expr::Cmp { key, op, value } => {
            let (gt, gte, lt, lte) = match op {
                CmpOp::Gt => (Some(*value), None, None, None),
                CmpOp::Gte => (None, Some(*value), None, None),
                CmpOp::Lt => (None, None, Some(*value), None),
                CmpOp::Lte => (None, None, None, Some(*value)),
            };
            Condition::Range {
                key: key.clone(),
                gt,
                gte,
                lt,
                lte,
            }
        }
        Expr::Exists { key } => Condition::Exists { key: key.clone() },
        Expr::GeoRadius {
            key,
            lat,
            lon,
            radius_m,
        } => Condition::GeoRadius {
            key: key.clone(),
            lat: *lat,
            lon: *lon,
            radius_m: *radius_m,
        },
        Expr::GeoBox {
            key,
            lat_min,
            lon_min,
            lat_max,
            lon_max,
        } => Condition::GeoBox {
            key: key.clone(),
            lat_min: *lat_min,
            lon_min: *lon_min,
            lat_max: *lat_max,
            lon_max: *lon_max,
        },
        Expr::And(..) | Expr::Or(..) | Expr::Not(..) => {
            return Err(unsupported(
                "a compound expression where a condition was expected",
            ))
        }
    })
}

fn unsupported(what: &str) -> QlError {
    QlError::new(
        format!(
            "{what} cannot be expressed by the typed Filter, so RRQL will not accept it. \
             Rewriting it into something Filter *nearly* means would answer a different \
             query than you asked."
        ),
        (0, 0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_query;

    #[test]
    fn and_lowers_to_must() {
        let q = parse_query("SELECT * WHERE lang = 'en' AND year >= 2020").unwrap();
        let f = q.dsl.unwrap();
        assert_eq!(f.must.len(), 2);
        assert!(f.should.is_empty() && f.must_not.is_empty());
        assert_eq!(q.top_k, DEFAULT_LIMIT, "no LIMIT -> the default");
    }

    #[test]
    fn or_lowers_to_should() {
        let f = parse_query("SELECT * WHERE lang = 'en' OR lang = 'fr'")
            .unwrap()
            .dsl
            .unwrap();
        assert_eq!(f.should.len(), 2);
        assert!(f.must.is_empty());
    }

    #[test]
    fn not_lowers_to_must_not() {
        let f = parse_query("SELECT * WHERE NOT lang = 'en'")
            .unwrap()
            .dsl
            .unwrap();
        assert_eq!(f.must_not.len(), 1);
        assert!(f.must.is_empty());
    }

    #[test]
    fn from_and_limit_reach_the_query() {
        let q = parse_query("SELECT * FROM docs LIMIT 25").unwrap();
        assert_eq!(q.collection.as_deref(), Some("docs"));
        assert_eq!(q.top_k, 25);
    }

    #[test]
    fn comparisons_become_range_bounds() {
        let f = parse_query("SELECT * WHERE year >= 2020 AND year < 2025")
            .unwrap()
            .dsl
            .unwrap();
        assert!(matches!(f.must[0], Condition::Range { gte: Some(y), .. } if y == 2020.0));
        assert!(matches!(f.must[1], Condition::Range { lt: Some(y), .. } if y == 2025.0));
    }

    #[test]
    fn in_becomes_any_and_exists_becomes_exists() {
        let f = parse_query("SELECT * WHERE tag IN ('a','b') AND EXISTS(author)")
            .unwrap()
            .dsl
            .unwrap();
        assert!(matches!(&f.must[0], Condition::Any { values, .. } if values.len() == 2));
        assert!(matches!(&f.must[1], Condition::Exists { .. }));
    }

    #[test]
    fn geo_lowers_exactly() {
        let f = parse_query("SELECT * WHERE loc INSIDE RADIUS(51.5, -0.12, 5000)")
            .unwrap()
            .dsl
            .unwrap();
        assert!(matches!(f.must[0], Condition::GeoRadius { radius_m, .. } if radius_m == 5000.0));
    }

    // --- the refusals: each of these could be "nearly" lowered, and must not be

    #[test]
    fn and_mixed_with_or_is_rejected_not_approximated() {
        let e = parse_query("SELECT * WHERE a = 1 AND (b = 2 OR c = 3)").unwrap_err();
        assert!(
            e.message.contains("cannot be expressed"),
            "must refuse, not approximate: {}",
            e.message
        );
    }

    #[test]
    fn or_combined_with_must_is_rejected() {
        assert!(parse_query("SELECT * WHERE a = 1 AND b = 2 OR c = 3").is_err());
    }

    #[test]
    fn not_of_a_group_is_rejected() {
        assert!(parse_query("SELECT * WHERE NOT (a = 1 AND b = 2)").is_err());
    }

    #[test]
    fn neq_tells_you_the_supported_spelling() {
        let e = parse_query("SELECT * WHERE lang != 'en'").unwrap_err();
        assert!(
            e.message.contains("NOT key = value"),
            "the refusal must name the fix: {}",
            e.message
        );
        // ...and that spelling works.
        assert!(parse_query("SELECT * WHERE NOT lang = 'en'").is_ok());
    }

    #[test]
    fn plain_or_at_the_top_level_is_fine() {
        // The rejection must be narrow: OR alone is exact.
        assert!(parse_query("SELECT * WHERE a = 1 OR b = 2 OR c = 3").is_ok());
    }

    /// `a OR b OR c` parses as Or(Or(a,b), c). A disjunction of disjunctions is
    /// still ONE flat disjunction, and `should` says that exactly — so all three
    /// must land side by side. The first version of this lowering refused the
    /// inner OR and would have rejected a query it can answer perfectly.
    #[test]
    fn a_chain_of_ors_flattens_into_one_should_list() {
        let f = parse_query("SELECT * WHERE a = 1 OR b = 2 OR c = 3")
            .unwrap()
            .dsl
            .unwrap();
        assert_eq!(f.should.len(), 3, "all three arms are one flat disjunction");
        assert!(f.must.is_empty() && f.must_not.is_empty());
    }
}
