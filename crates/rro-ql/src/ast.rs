//! The RRQL AST.
//!
//! Deliberately thin. It exists to be **lowered** into `rro-core`'s types, not
//! to be a second model of a query — a parallel model is how a language and its
//! engine drift apart. Anything the AST can express, [`crate::lower_select`]
//! must be able to turn into an `EstateQuery`/`Filter`, and the property test
//! holds that line.

/// A literal value. Mirrors what `Condition` accepts (`serde_json::Value`).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A string.
    Str(String),
    /// A number.
    Num(f64),
    /// A boolean.
    Bool(bool),
    /// `null`.
    Null,
}

impl Value {
    /// As the JSON value `Condition` stores.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Str(s) => serde_json::Value::String(s.clone()),
            Value::Num(n) => serde_json::Number::from_f64(*n)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Null => serde_json::Value::Null,
        }
    }
}

/// A `WHERE` expression.
///
/// `And`/`Or`/`Not` are kept as a tree here even though `Filter` is flat
/// (must/should/must_not). Lowering flattens them, and rejects what `Filter`
/// genuinely cannot represent rather than silently approximating it — see
/// [`crate::lower_select`].
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// `key = value`
    Eq {
        /// Metadata field.
        key: String,
        /// Value.
        value: Value,
    },
    /// `key != value`
    Neq {
        /// Metadata field.
        key: String,
        /// Value.
        value: Value,
    },
    /// `key IN (a, b, c)`
    In {
        /// Metadata field.
        key: String,
        /// Accepted values.
        values: Vec<Value>,
    },
    /// `key > | >= | < | <= n` — numeric.
    Cmp {
        /// Metadata field.
        key: String,
        /// The operator.
        op: CmpOp,
        /// Bound.
        value: f64,
    },
    /// `key IS EXISTS` / `EXISTS(key)`
    Exists {
        /// Metadata field.
        key: String,
    },
    /// `key INSIDE RADIUS(lat, lon, meters)`
    GeoRadius {
        /// Metadata field.
        key: String,
        /// Center latitude.
        lat: f64,
        /// Center longitude.
        lon: f64,
        /// Radius, meters.
        radius_m: f64,
    },
    /// `key INSIDE BOX(lat_min, lon_min, lat_max, lon_max)`
    GeoBox {
        /// Metadata field.
        key: String,
        /// South edge.
        lat_min: f64,
        /// West edge.
        lon_min: f64,
        /// North edge.
        lat_max: f64,
        /// East edge.
        lon_max: f64,
    },
    /// `a AND b`
    And(Box<Expr>, Box<Expr>),
    /// `a OR b`
    Or(Box<Expr>, Box<Expr>),
    /// `NOT a`
    Not(Box<Expr>),
}

/// A numeric comparison operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `>`
    Gt,
    /// `>=`
    Gte,
    /// `<`
    Lt,
    /// `<=`
    Lte,
}

/// `SELECT * [FROM collection] [WHERE expr] [LIMIT n]`
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Select {
    /// `FROM <collection>` — the estate's named collection.
    pub from: Option<String>,
    /// `WHERE <expr>`
    pub where_: Option<Expr>,
    /// `LIMIT <n>` — becomes `top_k`.
    pub limit: Option<usize>,
}

/// A parsed statement. One variant today; B2/B3 add DEFINE/CRUD/RELATE.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// A `SELECT`.
    Select(Select),
}
