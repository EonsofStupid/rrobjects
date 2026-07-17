//! B1's gate: **parsed ≡ hand-built typed query**, over randomly generated ASTs.
//!
//! ADR-0003 states the rule this enforces: *RRQL compiles to the proven
//! machinery; it never re-implements it.* The way that goes wrong is not a
//! crash — it is a query that parses fine and lowers to a `Filter` meaning
//! something subtly different. Example tests cannot find that; they only check
//! the cases someone already thought of.
//!
//! So: generate a random AST, render it to text, parse the text, lower it, and
//! require the result to equal the `Filter` built directly from the same AST.
//! Any divergence between "what the language says" and "what the engine
//! executes" fails here.

use proptest::prelude::*;
use rro_core::{Condition, Filter};
use rro_ql::{parse_query, Expr, Value};

// ---- generators -----------------------------------------------------------

fn key() -> impl Strategy<Value = String> {
    prop::sample::select(vec!["lang", "year", "tag", "author", "score", "loc"])
        .prop_map(str::to_string)
}

fn val() -> impl Strategy<Value = Value> {
    prop_oneof![
        // No quotes/backslashes: escaping is the lexer's job and has its own
        // tests. Mixing concerns here would make a failure ambiguous.
        "[a-z][a-z0-9 ]{0,8}".prop_map(Value::Str),
        (-1000.0f64..1000.0).prop_map(|n| Value::Num((n * 100.0).round() / 100.0)),
        any::<bool>().prop_map(Value::Bool),
        Just(Value::Null),
    ]
}

/// Leaves only — the compound shapes RRQL deliberately refuses are covered by
/// the refusal tests in `lower.rs`, and generating them here would just assert
/// that rejection twice.
fn leaf() -> impl Strategy<Value = Expr> {
    prop_oneof![
        (key(), val()).prop_map(|(key, value)| Expr::Eq { key, value }),
        (key(), prop::collection::vec(val(), 1..4))
            .prop_map(|(key, values)| Expr::In { key, values }),
        (key(), -1000.0f64..1000.0).prop_map(|(key, v)| Expr::Cmp {
            key,
            op: rro_ql_cmp_gte(),
            value: (v * 100.0).round() / 100.0
        }),
        key().prop_map(|key| Expr::Exists { key }),
    ]
}

// `CmpOp` is re-exported through the AST; this keeps the generator readable.
fn rro_ql_cmp_gte() -> rro_ql::CmpOp {
    rro_ql::CmpOp::Gte
}

// ---- rendering: AST -> RRQL text ------------------------------------------

fn render_value(v: &Value) -> String {
    match v {
        Value::Str(s) => format!("'{s}'"),
        Value::Num(n) => format!("{n}"),
        Value::Bool(b) => format!("{b}"),
        Value::Null => "null".to_string(),
    }
}

fn render_leaf(e: &Expr) -> String {
    match e {
        Expr::Eq { key, value } => format!("{key} = {}", render_value(value)),
        Expr::In { key, values } => {
            let vs: Vec<String> = values.iter().map(render_value).collect();
            format!("{key} IN ({})", vs.join(", "))
        }
        Expr::Cmp { key, value, .. } => format!("{key} >= {value}"),
        Expr::Exists { key } => format!("EXISTS({key})"),
        other => unreachable!("the generator only makes leaves, got {other:?}"),
    }
}

// ---- the oracle: AST -> Filter, built directly ----------------------------

fn leaf_condition(e: &Expr) -> Condition {
    match e {
        Expr::Eq { key, value } => Condition::Eq {
            key: key.clone(),
            value: value.to_json(),
        },
        Expr::In { key, values } => Condition::Any {
            key: key.clone(),
            values: values.iter().map(Value::to_json).collect(),
        },
        Expr::Cmp { key, value, .. } => Condition::Range {
            key: key.clone(),
            gt: None,
            gte: Some(*value),
            lt: None,
            lte: None,
        },
        Expr::Exists { key } => Condition::Exists { key: key.clone() },
        other => unreachable!("{other:?}"),
    }
}

proptest! {
    /// A conjunction of N random leaves must lower to exactly those N
    /// conditions in `must`, in order.
    #[test]
    fn and_chain_parses_to_the_hand_built_filter(leaves in prop::collection::vec(leaf(), 1..6)) {
        let text = format!(
            "SELECT * WHERE {}",
            leaves.iter().map(render_leaf).collect::<Vec<_>>().join(" AND ")
        );
        let got = parse_query(&text).expect("generated RRQL must parse").dsl.expect("has a filter");
        let want = Filter {
            must: leaves.iter().map(leaf_condition).collect(),
            ..Default::default()
        };
        prop_assert_eq!(got, want, "text: {}", text);
    }

    /// A disjunction of N random leaves must lower to exactly those N
    /// conditions in `should` — including the left-associative nesting
    /// (`a OR b OR c` = `Or(Or(a,b), c)`), which must flatten.
    #[test]
    fn or_chain_parses_to_the_hand_built_filter(leaves in prop::collection::vec(leaf(), 1..6)) {
        let text = format!(
            "SELECT * WHERE {}",
            leaves.iter().map(render_leaf).collect::<Vec<_>>().join(" OR ")
        );
        let got = parse_query(&text).expect("generated RRQL must parse").dsl.expect("has a filter");
        let want = if leaves.len() == 1 {
            // A single "disjunct" is just a must.
            Filter { must: vec![leaf_condition(&leaves[0])], ..Default::default() }
        } else {
            Filter { should: leaves.iter().map(leaf_condition).collect(), ..Default::default() }
        };
        prop_assert_eq!(got, want, "text: {}", text);
    }

    /// `NOT <leaf>` must land in `must_not`, exactly.
    #[test]
    fn negated_leaf_lands_in_must_not(l in leaf()) {
        let text = format!("SELECT * WHERE NOT {}", render_leaf(&l));
        let got = parse_query(&text).expect("must parse").dsl.expect("has a filter");
        prop_assert_eq!(
            got,
            Filter { must_not: vec![leaf_condition(&l)], ..Default::default() },
            "text: {}", text
        );
    }

    /// LIMIT is top_k, FROM is the collection. No silent defaults sneaking in.
    #[test]
    fn from_and_limit_survive_the_round_trip(n in 1usize..10_000) {
        let q = parse_query(&format!("SELECT * FROM docs LIMIT {n}")).unwrap();
        prop_assert_eq!(q.top_k, n);
        prop_assert_eq!(q.collection.as_deref(), Some("docs"));
    }

    /// Parsing must never panic, whatever it is fed. A query surface that can be
    /// crashed by a string is a denial-of-service on every node exposing it.
    #[test]
    fn arbitrary_input_never_panics(s in ".*") {
        let _ = parse_query(&s);
    }

    /// Nor on RRQL-ish noise, which is likelier to reach a real parser bug than
    /// pure random bytes.
    #[test]
    fn rrql_shaped_noise_never_panics(
        s in prop::collection::vec(
            prop::sample::select(vec![
                "SELECT", "*", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS",
                "EXISTS", "LIMIT", "INSIDE", "RADIUS", "BOX",
                "(", ")", ",", "=", "!=", "<", "<=", ">", ">=",
                "x", "'s'", "1", "true", "null", "`odd name`", "--c",
            ]),
            0..14
        ).prop_map(|v| v.join(" "))
    ) {
        let _ = parse_query(&s);
    }
}
