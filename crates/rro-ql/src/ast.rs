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

/// `DEFINE INDEX ON <field>` — a payload index.
///
/// Only the subjects the engine actually has. Promising `DEFINE TABLE`/`FIELD`/
/// `EVENT` before schemas exist (Phase 10) would be a language writing cheques
/// the engine cannot cash.
#[derive(Debug, Clone, PartialEq)]
pub enum Define {
    /// `DEFINE INDEX ON <field>` → `create_payload_index`.
    Index {
        /// Metadata field to index.
        field: String,
    },
    /// `DEFINE ALIAS <alias> FOR <collection>` → `create_alias`.
    Alias {
        /// The alias name.
        alias: String,
        /// The collection it points at.
        collection: String,
    },
}

/// `REMOVE INDEX ON <field>` / `REMOVE ALIAS <a>` / `REMOVE COLLECTION <c>`.
#[derive(Debug, Clone, PartialEq)]
pub enum Remove {
    /// `REMOVE ALIAS <alias>` → `delete_alias`.
    Alias {
        /// The alias name.
        alias: String,
    },
    /// `REMOVE COLLECTION <name>` → `drop_collection`.
    Collection {
        /// The collection name.
        name: String,
    },
}

/// `UPDATE <id> SET k = v, ...` / `UPDATE <id> CONTENT {..}` — payload writes.
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// Record id.
    pub id: String,
    /// `(key, value)` pairs.
    pub set: Vec<(String, Value)>,
    /// `CONTENT` replaces the whole payload; `SET` merges into it.
    ///
    /// The distinction is load-bearing: `set_payload` patches, and
    /// `overwrite_payload` replaces. Collapsing them would silently destroy
    /// fields the caller never mentioned.
    pub replace: bool,
}

/// `DELETE <id>` — remove a record, or clear its payload.
#[derive(Debug, Clone, PartialEq)]
pub struct Delete {
    /// Record id.
    pub id: String,
    /// `DELETE PAYLOAD <id>` clears metadata but keeps the record;
    /// `DELETE <id>` removes the record entirely.
    pub payload_only: bool,
    /// `DELETE PAYLOAD <id> (k, k2)` removes only those keys.
    pub keys: Vec<String>,
}

/// `RELATE <from> -> <verb> -> <to>` — assert one graph edge.
#[derive(Debug, Clone, PartialEq)]
pub struct Relate {
    /// Source record id.
    pub from: String,
    /// The edge verb.
    pub verb: String,
    /// Target record id.
    pub to: String,
}

/// Which way a traversal follows edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `->verb->` — follow outbound edges.
    Out,
    /// `<-verb<-` — follow inbound edges.
    In,
    /// `<->verb<->` — follow both.
    Both,
}

/// `TRAVERSE <id> ->verb-> [DEPTH n] [LIMIT n]`
#[derive(Debug, Clone, PartialEq)]
pub struct Traverse {
    /// Start ids.
    pub start: Vec<String>,
    /// Verbs to follow; empty = all.
    pub verbs: Vec<String>,
    /// Direction.
    pub dir: Direction,
    /// Max hops.
    pub depth: Option<usize>,
    /// Hard cap on visited ids.
    pub limit: Option<usize>,
}

/// `LIVE [SINCE n]` — the push changefeed. `LIVE` is the familiar keyword for a
/// live subscription; in RRO the capability is `watch`, which this routes to.
#[derive(Debug, Clone, PartialEq)]
pub struct Live {
    /// Resume from this feed sequence (`SINCE`), or from now.
    pub since: Option<u64>,
}

/// `INFO` — the live catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct Info;

/// A parsed statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// A `SELECT`.
    Select(Select),
    /// A `DEFINE`.
    Define(Define),
    /// A `REMOVE`.
    Remove(Remove),
    /// An `UPDATE`.
    Update(Update),
    /// A `DELETE`.
    Delete(Delete),
    /// A `RELATE`.
    Relate(Relate),
    /// A `TRAVERSE`.
    Traverse(Traverse),
    /// A `LIVE` (the `watch` push feed).
    Live(Live),
    /// An `INFO`.
    Info(Info),
}

impl Statement {
    /// The leading keyword, for errors that name what the caller actually sent.
    pub fn keyword(&self) -> &'static str {
        match self {
            Statement::Select(_) => "SELECT",
            Statement::Define(_) => "DEFINE",
            Statement::Remove(_) => "REMOVE",
            Statement::Update(_) => "UPDATE",
            Statement::Delete(_) => "DELETE",
            Statement::Relate(_) => "RELATE",
            Statement::Traverse(_) => "TRAVERSE",
            Statement::Live(_) => "LIVE",
            Statement::Info(_) => "INFO",
        }
    }

    /// Whether this statement mutates the estate.
    ///
    /// The seam a caller needs to gate writes — an MCP tool or a REST endpoint
    /// exposed read-only can refuse on this without re-deriving the taxonomy.
    pub fn is_write(&self) -> bool {
        // RELATE mutates. TRAVERSE/LIVE/INFO read, like SELECT — a read-only
        // surface must be able to expose them.
        matches!(
            self,
            Statement::Define(_)
                | Statement::Remove(_)
                | Statement::Update(_)
                | Statement::Delete(_)
                | Statement::Relate(_)
        )
    }
}
