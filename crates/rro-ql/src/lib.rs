//! # rro-ql — RRQL, the Reason Ready Query Language
//!
//! Text in, typed query out. **Parsing only** — no execution, no storage, no
//! transport. Every statement lowers to something `rro-core` already defines and
//! `connxism` already executes and tests.
//!
//! ```text
//! SELECT * FROM docs WHERE lang = 'en' AND year >= 2020 LIMIT 10
//!   -> EstateQuery { collection: Some("docs"), top_k: 10, dsl: Some(Filter {
//!        must: [Eq{lang,"en"}, Range{year, gte:2020}], .. }), .. }
//! ```
//!
//! ## The rule
//!
//! **RRQL compiles to the proven machinery; it never re-implements it.** If a
//! construct cannot be expressed as an [`rro_core::EstateQuery`] /
//! [`rro_core::Filter`], it does not go in the language until the typed layer
//! supports it. A query surface that can say things its engine cannot do is how
//! a language starts lying about its engine.
//!
//! The gate is mechanical, and it is a property test, not an example test:
//! **parse(render(ast)) ≡ ast** over randomly generated ASTs.
//!
//! Why a crate and not a module: see `docs/adr/0003-rro-ql.md` (the COSTAR).
//! Short version — `rro-core` is the innermost crate and must not grow the
//! outermost concern; `connxism` would couple the language to RocksDB; and in
//! `rro-engine` neither `rro-client` (the MCP `rro_sql` tool) nor a future
//! `rro-http` could reach it.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod ast;
mod error;
mod lexer;
mod lower;
mod parser;

pub use ast::{
    CmpOp, Define, Delete, Direction, Expr, Info, Live, Relate, Remove, Select, Statement,
    Traverse, Update, Value,
};
pub use error::QlError;
pub use lexer::{lex, LexError, Token, TokenKind};
pub use lower::lower_select;
pub use parser::parse;

/// Parse a **read** — RRQL text → a typed [`rro_core::EstateQuery`].
///
/// Reads are the case with a pure, dependency-free answer: a `SELECT` lowers to
/// a value, and no estate is needed to produce it. Writes (`DEFINE`, `UPDATE`,
/// `DELETE`, `REMOVE`) cannot: they name an *effect*, and this crate does not
/// execute. They parse to a [`Statement`], and the caller — which owns an
/// estate — applies it. That split is why `rro-ql` can stay a pure function of
/// text with `rro-core` as its only dependency (ADR-0003).
///
/// # Errors
/// [`QlError`] with a byte span pointing at the offending text, or a refusal if
/// `src` is a write statement.
pub fn parse_query(src: &str) -> Result<rro_core::EstateQuery, QlError> {
    match parse(src)? {
        Statement::Select(s) => lower_select(s),
        other => Err(QlError::new(
            format!(
                "`{}` is a write, not a query — parse it with `parse()` and apply the \
                 Statement against an estate. parse_query() returns an EstateQuery, and \
                 a write has no query to return.",
                other.keyword()
            ),
            (0, 0),
        )),
    }
}
