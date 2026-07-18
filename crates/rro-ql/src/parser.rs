//! Recursive-descent parser: tokens → [`Statement`].
//!
//! Precedence, loosest to tightest: `OR` < `AND` < `NOT` < primary. Standard SQL
//! shape, so `a = 1 OR b = 2 AND c = 3` parses as `a=1 OR (b=2 AND c=3)` — which
//! is what anyone who has written SQL expects, and getting it backwards is a
//! silent wrong-answer bug rather than a syntax error.

use crate::ast::{
    CmpOp, Define, Delete, Direction, Expr, FieldType, Info, Live, Relate, Remove, Select,
    Statement, Traverse, Update, Value,
};
use crate::error::QlError;
use crate::lexer::{lex, Token, TokenKind};

/// Parse RRQL text into a [`Statement`].
///
/// # Errors
/// [`QlError`] with a span pointing at the offending text.
pub fn parse(src: &str) -> Result<Statement, QlError> {
    let tokens = lex(src)?;
    let mut p = Parser { tokens, pos: 0 };
    let stmt = p.statement()?;
    p.expect_eof()?;
    Ok(stmt)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].kind
    }

    fn span(&self) -> (usize, usize) {
        self.tokens[self.pos.min(self.tokens.len() - 1)].span
    }

    fn bump(&mut self) -> TokenKind {
        let k = self.tokens[self.pos.min(self.tokens.len() - 1)]
            .kind
            .clone();
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        k
    }

    fn eat(&mut self, want: &TokenKind) -> bool {
        if self.peek() == want {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, want: &TokenKind) -> Result<(), QlError> {
        if self.eat(want) {
            Ok(())
        } else {
            Err(QlError::new(
                format!("expected {want}, found {}", self.peek()),
                self.span(),
            ))
        }
    }

    fn expect_eof(&self) -> Result<(), QlError> {
        if matches!(self.peek(), TokenKind::Eof) {
            Ok(())
        } else {
            Err(QlError::new(
                format!("unexpected trailing {}", self.peek()),
                self.span(),
            ))
        }
    }

    fn ident(&mut self) -> Result<String, QlError> {
        let span = self.span();
        match self.bump() {
            TokenKind::Ident(s) => Ok(s),
            other => Err(QlError::new(
                format!("expected an identifier, found {other}"),
                span,
            )),
        }
    }

    fn number(&mut self) -> Result<f64, QlError> {
        let span = self.span();
        match self.bump() {
            TokenKind::Num(n) => Ok(n),
            other => Err(QlError::new(
                format!("expected a number, found {other}"),
                span,
            )),
        }
    }

    fn value(&mut self) -> Result<Value, QlError> {
        let span = self.span();
        match self.bump() {
            TokenKind::Str(s) => Ok(Value::Str(s)),
            TokenKind::Num(n) => Ok(Value::Num(n)),
            TokenKind::Bool(b) => Ok(Value::Bool(b)),
            TokenKind::Null => Ok(Value::Null),
            other => Err(QlError::new(
                format!("expected a value, found {other}"),
                span,
            )),
        }
    }

    fn statement(&mut self) -> Result<Statement, QlError> {
        if self.eat(&TokenKind::Select) {
            return Ok(Statement::Select(self.select()?));
        }
        if self.eat(&TokenKind::Define) {
            return Ok(Statement::Define(self.define()?));
        }
        if self.eat(&TokenKind::Remove) {
            return Ok(Statement::Remove(self.remove()?));
        }
        if self.eat(&TokenKind::Update) {
            return Ok(Statement::Update(self.update()?));
        }
        if self.eat(&TokenKind::Delete) {
            return Ok(Statement::Delete(self.delete()?));
        }
        if self.eat(&TokenKind::Relate) {
            return Ok(Statement::Relate(self.relate()?));
        }
        if self.eat(&TokenKind::Traverse) {
            return Ok(Statement::Traverse(self.traverse()?));
        }
        if self.eat(&TokenKind::Live) {
            let since = if self.eat(&TokenKind::Since) {
                Some(self.whole_number()? as u64)
            } else {
                None
            };
            return Ok(Statement::Live(Live { since }));
        }
        if self.eat(&TokenKind::Info) {
            return Ok(Statement::Info(Info));
        }
        Err(QlError::new(
            format!(
                "expected SELECT, DEFINE, REMOVE, UPDATE, DELETE, RELATE, TRAVERSE, \
                 LIVE or INFO, found {}",
                self.peek()
            ),
            self.span(),
        ))
    }

    /// `RELATE <from> -> <verb> -> <to>`
    ///
    /// The `->` arrow is the conventional graph-edge spelling — it reads as
    /// "assert an edge" — and RRO's `relate(from, verb, to)` over the connectome
    /// is exactly that shape.
    fn relate(&mut self) -> Result<Relate, QlError> {
        let from = self.record_id()?;
        self.expect(&TokenKind::ArrowOut)?;
        let verb = self.ident()?;
        self.expect(&TokenKind::ArrowOut)?;
        let to = self.record_id()?;
        Ok(Relate { from, verb, to })
    }

    /// `TRAVERSE <id>[, <id>] ->verb-> [DEPTH n] [LIMIT n]`
    ///
    /// Direction comes from the arrow: `->` out, `<-` in, `<->` both. A bare
    /// arrow with no verb means "every verb".
    fn traverse(&mut self) -> Result<Traverse, QlError> {
        let mut start = vec![self.record_id()?];
        while self.eat(&TokenKind::Comma) {
            start.push(self.record_id()?);
        }

        let span = self.span();
        let dir = match self.bump() {
            TokenKind::ArrowOut => Direction::Out,
            TokenKind::ArrowIn => Direction::In,
            TokenKind::ArrowBoth => Direction::Both,
            other => {
                return Err(QlError::new(
                    format!("expected `->`, `<-` or `<->` after the start ids, found {other}"),
                    span,
                ))
            }
        };

        // `-> verb ->` names a verb; `->` alone means every verb.
        let mut verbs = Vec::new();
        if let TokenKind::Ident(_) = self.peek() {
            verbs.push(self.ident()?);
            // The closing arrow is optional: `TRAVERSE a ->cites->` and
            // `TRAVERSE a ->cites` mean the same walk, and rejecting the shorter
            // form would be ceremony.
            let _ = self.eat(&TokenKind::ArrowOut)
                || self.eat(&TokenKind::ArrowIn)
                || self.eat(&TokenKind::ArrowBoth);
        }

        let mut depth = None;
        let mut limit = None;
        if self.eat(&TokenKind::Depth) {
            depth = Some(self.whole_number()?);
        }
        if self.eat(&TokenKind::Limit) {
            limit = Some(self.whole_number()?);
        }
        Ok(Traverse {
            start,
            verbs,
            dir,
            depth,
            limit,
        })
    }

    /// A non-negative whole number — for LIMIT / DEPTH / SINCE.
    fn whole_number(&mut self) -> Result<usize, QlError> {
        let span = self.span();
        let n = self.number()?;
        if n < 0.0 || n.fract() != 0.0 {
            return Err(QlError::new(
                format!("expected a non-negative whole number, found {n}"),
                span,
            ));
        }
        Ok(n as usize)
    }

    /// `DEFINE INDEX ON <field>` | `DEFINE ALIAS <a> FOR <collection>`
    /// | `DEFINE FIELD <field> ON <collection> TYPE <type>`
    fn define(&mut self) -> Result<Define, QlError> {
        if self.eat(&TokenKind::Index) {
            self.expect(&TokenKind::On)?;
            return Ok(Define::Index {
                field: self.ident()?,
            });
        }
        if self.eat(&TokenKind::Alias) {
            let alias = self.ident()?;
            self.expect(&TokenKind::For)?;
            return Ok(Define::Alias {
                alias,
                collection: self.ident()?,
            });
        }
        if self.eat(&TokenKind::Field) {
            let field = self.ident()?;
            self.expect(&TokenKind::On)?;
            let collection = self.ident()?;
            self.expect(&TokenKind::Type)?;
            let span = self.span();
            let ty_name = self.ident()?;
            let ty = FieldType::parse(&ty_name).ok_or_else(|| {
                QlError::new(
                    format!(
                        "unknown field type `{ty_name}` — expected string, int, \
                         float, bool, datetime or uuid"
                    ),
                    span,
                )
            })?;
            return Ok(Define::Field {
                field,
                collection,
                ty,
            });
        }
        // Deliberately narrow: RRQL defines exactly what the engine has — payload
        // indexes, aliases, and schemafull field types — and nothing else.
        Err(QlError::new(
            format!(
                "DEFINE supports INDEX, ALIAS and FIELD, found {}. (TABLE/EVENT \
                 are not implemented.)",
                self.peek()
            ),
            self.span(),
        ))
    }

    /// `REMOVE ALIAS <a>` | `REMOVE COLLECTION <c>` | `REMOVE FIELD <f> ON <c>`
    fn remove(&mut self) -> Result<Remove, QlError> {
        if self.eat(&TokenKind::Alias) {
            return Ok(Remove::Alias {
                alias: self.ident()?,
            });
        }
        if self.eat(&TokenKind::Collection) {
            return Ok(Remove::Collection {
                name: self.ident()?,
            });
        }
        if self.eat(&TokenKind::Field) {
            let field = self.ident()?;
            self.expect(&TokenKind::On)?;
            return Ok(Remove::Field {
                field,
                collection: self.ident()?,
            });
        }
        Err(QlError::new(
            format!(
                "REMOVE supports ALIAS, COLLECTION and FIELD, found {}",
                self.peek()
            ),
            self.span(),
        ))
    }

    /// `UPDATE <id> SET k = v, ...` | `UPDATE <id> CONTENT SET k = v, ...`
    fn update(&mut self) -> Result<Update, QlError> {
        let id = self.record_id()?;
        // CONTENT replaces, SET merges. Both spell their pairs the same way; the
        // keyword picks set_payload vs overwrite_payload, and conflating them
        // would silently drop fields the caller never mentioned.
        let replace = if self.eat(&TokenKind::Content) {
            true
        } else {
            self.expect(&TokenKind::Set)?;
            false
        };
        if replace {
            self.expect(&TokenKind::Set)?;
        }
        let mut set = Vec::new();
        loop {
            let key = self.ident()?;
            self.expect(&TokenKind::Eq)?;
            let value = self.value()?;
            set.push((key, value));
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        Ok(Update { id, set, replace })
    }

    /// `DELETE <id>` | `DELETE PAYLOAD <id>` | `DELETE PAYLOAD <id> (k, k2)`
    fn delete(&mut self) -> Result<Delete, QlError> {
        let payload_only = self.eat(&TokenKind::Payload);
        let id = self.record_id()?;
        let mut keys = Vec::new();
        if payload_only && self.eat(&TokenKind::LParen) {
            loop {
                keys.push(self.ident()?);
                if self.eat(&TokenKind::Comma) {
                    continue;
                }
                self.expect(&TokenKind::RParen)?;
                break;
            }
        }
        Ok(Delete {
            id,
            payload_only,
            keys,
        })
    }

    /// A record id: a bare/backticked identifier, or a quoted string.
    ///
    /// Ids in this estate are arbitrary strings (`MED-10`, a UUID, a path), so
    /// they must be quotable — an id-shaped grammar would reject real ids.
    fn record_id(&mut self) -> Result<String, QlError> {
        let span = self.span();
        match self.bump() {
            TokenKind::Ident(s) => Ok(s),
            TokenKind::Str(s) => Ok(s),
            other => Err(QlError::new(
                format!("expected a record id, found {other}"),
                span,
            )),
        }
    }

    fn select(&mut self) -> Result<Select, QlError> {
        // `*` is the only projection today: the estate returns whole candidates,
        // so a column list would be a promise the engine cannot keep.
        self.expect(&TokenKind::Star)?;

        let mut s = Select::default();
        if self.eat(&TokenKind::From) {
            s.from = Some(self.ident()?);
        }
        if self.eat(&TokenKind::Where) {
            s.where_ = Some(self.expr()?);
        }
        if self.eat(&TokenKind::Limit) {
            s.limit = Some(self.whole_number()?);
        }
        Ok(s)
    }

    // OR — loosest
    fn expr(&mut self) -> Result<Expr, QlError> {
        let mut lhs = self.and_expr()?;
        while self.eat(&TokenKind::Or) {
            let rhs = self.and_expr()?;
            lhs = Expr::Or(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn and_expr(&mut self) -> Result<Expr, QlError> {
        let mut lhs = self.unary()?;
        while self.eat(&TokenKind::And) {
            let rhs = self.unary()?;
            lhs = Expr::And(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> Result<Expr, QlError> {
        if self.eat(&TokenKind::Not) {
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<Expr, QlError> {
        if self.eat(&TokenKind::LParen) {
            let e = self.expr()?;
            self.expect(&TokenKind::RParen)?;
            return Ok(e);
        }

        // EXISTS(key)
        if self.eat(&TokenKind::Exists) {
            self.expect(&TokenKind::LParen)?;
            let key = self.ident()?;
            self.expect(&TokenKind::RParen)?;
            return Ok(Expr::Exists { key });
        }

        let key = self.ident()?;

        // key IS EXISTS
        if self.eat(&TokenKind::Is) {
            self.expect(&TokenKind::Exists)?;
            return Ok(Expr::Exists { key });
        }

        // key IN (a, b, ...)
        if self.eat(&TokenKind::In) {
            self.expect(&TokenKind::LParen)?;
            let mut values = Vec::new();
            if !self.eat(&TokenKind::RParen) {
                loop {
                    values.push(self.value()?);
                    if self.eat(&TokenKind::Comma) {
                        continue;
                    }
                    self.expect(&TokenKind::RParen)?;
                    break;
                }
            }
            return Ok(Expr::In { key, values });
        }

        // key INSIDE RADIUS(lat, lon, m) | BOX(lat_min, lon_min, lat_max, lon_max)
        if self.eat(&TokenKind::Inside) {
            if self.eat(&TokenKind::Radius) {
                self.expect(&TokenKind::LParen)?;
                let lat = self.number()?;
                self.expect(&TokenKind::Comma)?;
                let lon = self.number()?;
                self.expect(&TokenKind::Comma)?;
                let radius_m = self.number()?;
                self.expect(&TokenKind::RParen)?;
                return Ok(Expr::GeoRadius {
                    key,
                    lat,
                    lon,
                    radius_m,
                });
            }
            if self.eat(&TokenKind::BoxKw) {
                self.expect(&TokenKind::LParen)?;
                let lat_min = self.number()?;
                self.expect(&TokenKind::Comma)?;
                let lon_min = self.number()?;
                self.expect(&TokenKind::Comma)?;
                let lat_max = self.number()?;
                self.expect(&TokenKind::Comma)?;
                let lon_max = self.number()?;
                self.expect(&TokenKind::RParen)?;
                return Ok(Expr::GeoBox {
                    key,
                    lat_min,
                    lon_min,
                    lat_max,
                    lon_max,
                });
            }
            return Err(QlError::new(
                format!("expected RADIUS or BOX after INSIDE, found {}", self.peek()),
                self.span(),
            ));
        }

        // comparisons
        let span = self.span();
        match self.bump() {
            TokenKind::Eq => Ok(Expr::Eq {
                key,
                value: self.value()?,
            }),
            TokenKind::Neq => Ok(Expr::Neq {
                key,
                value: self.value()?,
            }),
            TokenKind::Gt => Ok(Expr::Cmp {
                key,
                op: CmpOp::Gt,
                value: self.number()?,
            }),
            TokenKind::Gte => Ok(Expr::Cmp {
                key,
                op: CmpOp::Gte,
                value: self.number()?,
            }),
            TokenKind::Lt => Ok(Expr::Cmp {
                key,
                op: CmpOp::Lt,
                value: self.number()?,
            }),
            TokenKind::Lte => Ok(Expr::Cmp {
                key,
                op: CmpOp::Lte,
                value: self.number()?,
            }),
            other => Err(QlError::new(
                format!("expected a comparison after `{key}`, found {other}"),
                span,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel(src: &str) -> Select {
        match parse(src).unwrap() {
            Statement::Select(s) => s,
            other => panic!("expected a SELECT, got {}", other.keyword()),
        }
    }

    // ---- B2: DEFINE / REMOVE / UPDATE / DELETE ---------------------------

    #[test]
    fn define_index_and_alias() {
        assert_eq!(
            parse("DEFINE INDEX ON author").unwrap(),
            Statement::Define(Define::Index {
                field: "author".into()
            })
        );
        assert_eq!(
            parse("DEFINE ALIAS current FOR docs_v2").unwrap(),
            Statement::Define(Define::Alias {
                alias: "current".into(),
                collection: "docs_v2".into()
            })
        );
    }

    /// RRO has payload indexes and aliases, so those are the only `DEFINE`
    /// subjects RRQL accepts. Accepting `DEFINE TABLE` and ignoring it would be
    /// the language lying about the engine — so it must be refused, and the
    /// refusal must say
    /// why.
    #[test]
    fn define_table_is_refused_and_says_why() {
        // FIELD is now implemented (schemafull types); TABLE still is not.
        let e = parse("DEFINE TABLE docs").unwrap_err();
        assert!(
            e.message.contains("INDEX, ALIAS and FIELD"),
            "{}",
            e.message
        );
        assert!(e.message.contains("TABLE"), "names TABLE: {}", e.message);
    }

    #[test]
    fn define_and_remove_field_parse() {
        assert_eq!(
            parse("DEFINE FIELD price ON products TYPE float").unwrap(),
            Statement::Define(Define::Field {
                field: "price".into(),
                collection: "products".into(),
                ty: FieldType::Float,
            })
        );
        assert_eq!(
            parse("REMOVE FIELD price ON products").unwrap(),
            Statement::Remove(Remove::Field {
                field: "price".into(),
                collection: "products".into(),
            })
        );
        // An unknown type is a clear parse error.
        let e = parse("DEFINE FIELD x ON y TYPE bogus").unwrap_err();
        assert!(e.message.contains("unknown field type"), "{}", e.message);
    }

    #[test]
    fn remove_alias_and_collection() {
        assert_eq!(
            parse("REMOVE ALIAS current").unwrap(),
            Statement::Remove(Remove::Alias {
                alias: "current".into()
            })
        );
        assert_eq!(
            parse("REMOVE COLLECTION beta").unwrap(),
            Statement::Remove(Remove::Collection {
                name: "beta".into()
            })
        );
    }

    /// SET merges, CONTENT replaces. Conflating them silently destroys fields
    /// the caller never mentioned, so the AST keeps them distinct.
    #[test]
    fn update_set_merges_and_content_replaces() {
        match parse("UPDATE doc1 SET team = 'blue', rank = 3").unwrap() {
            Statement::Update(u) => {
                assert_eq!(u.id, "doc1");
                assert!(!u.replace, "SET must MERGE");
                assert_eq!(u.set.len(), 2);
                assert_eq!(u.set[1], ("rank".into(), Value::Num(3.0)));
            }
            other => panic!("{}", other.keyword()),
        }
        match parse("UPDATE doc1 CONTENT SET team = 'blue'").unwrap() {
            Statement::Update(u) => assert!(u.replace, "CONTENT must REPLACE"),
            other => panic!("{}", other.keyword()),
        }
    }

    /// Record ids are arbitrary strings in this estate (`MED-10`, a UUID, a
    /// path). An id-shaped grammar would reject real ids, so they are quotable.
    #[test]
    fn record_ids_can_be_quoted_or_bare() {
        for src in ["DELETE doc1", "DELETE 'MED-10'", "DELETE `odd id`"] {
            assert!(parse(src).is_ok(), "{src} should parse");
        }
    }

    #[test]
    fn delete_record_vs_payload_vs_keys() {
        match parse("DELETE doc1").unwrap() {
            Statement::Delete(d) => {
                assert!(!d.payload_only, "DELETE <id> removes the RECORD");
                assert!(d.keys.is_empty());
            }
            other => panic!("{}", other.keyword()),
        }
        match parse("DELETE PAYLOAD doc1").unwrap() {
            Statement::Delete(d) => {
                assert!(d.payload_only, "DELETE PAYLOAD keeps the record");
                assert!(d.keys.is_empty(), "no keys = clear the whole payload");
            }
            other => panic!("{}", other.keyword()),
        }
        match parse("DELETE PAYLOAD doc1 (team, rank)").unwrap() {
            Statement::Delete(d) => {
                assert!(d.payload_only);
                assert_eq!(d.keys, vec!["team".to_string(), "rank".to_string()]);
            }
            other => panic!("{}", other.keyword()),
        }
    }

    #[test]
    fn writes_are_flagged_as_writes() {
        // The seam a read-only MCP tool or REST endpoint gates on.
        assert!(!parse("SELECT *").unwrap().is_write());
        for src in [
            "DEFINE INDEX ON a",
            "REMOVE ALIAS x",
            "UPDATE d SET a = 1",
            "DELETE d",
        ] {
            assert!(parse(src).unwrap().is_write(), "{src} mutates the estate");
        }
    }

    // ---- B3: RELATE / TRAVERSE / LIVE / INFO ----------------------------

    #[test]
    fn relate_asserts_an_edge() {
        assert_eq!(
            parse("RELATE doc1 -> cites -> doc2").unwrap(),
            Statement::Relate(Relate {
                from: "doc1".into(),
                verb: "cites".into(),
                to: "doc2".into()
            })
        );
        // quoted ids, because real ids are arbitrary strings
        assert!(parse("RELATE 'MED-10' -> cites -> 'MED-20'").is_ok());
    }

    /// The arrow decides direction. `<->` must not lex as `<-` then `>` — same
    /// longest-match discipline as `<=` vs `<`, one length up.
    #[test]
    fn traverse_direction_comes_from_the_arrow() {
        let dir = |src: &str| match parse(src).unwrap() {
            Statement::Traverse(t) => t.dir,
            other => panic!("{}", other.keyword()),
        };
        assert_eq!(dir("TRAVERSE a -> cites ->"), Direction::Out);
        assert_eq!(dir("TRAVERSE a <- cites <-"), Direction::In);
        assert_eq!(dir("TRAVERSE a <-> cites <->"), Direction::Both);
    }

    #[test]
    fn traverse_bare_arrow_means_every_verb() {
        match parse("TRAVERSE a ->").unwrap() {
            Statement::Traverse(t) => {
                assert!(t.verbs.is_empty(), "no verb named = follow every verb");
                assert_eq!(t.dir, Direction::Out);
            }
            other => panic!("{}", other.keyword()),
        }
    }

    #[test]
    fn traverse_closing_arrow_is_optional() {
        // `->cites->` and `->cites` are the same walk; demanding the closing
        // arrow would be ceremony.
        let a = parse("TRAVERSE a -> cites ->").unwrap();
        let b = parse("TRAVERSE a -> cites").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn traverse_takes_multiple_starts_depth_and_limit() {
        match parse("TRAVERSE a, b, c -> cites -> DEPTH 3 LIMIT 50").unwrap() {
            Statement::Traverse(t) => {
                assert_eq!(t.start, vec!["a".to_string(), "b".into(), "c".into()]);
                assert_eq!(t.depth, Some(3));
                assert_eq!(t.limit, Some(50));
            }
            other => panic!("{}", other.keyword()),
        }
    }

    #[test]
    fn traverse_without_an_arrow_is_rejected() {
        let e = parse("TRAVERSE a cites").unwrap_err();
        assert!(e.message.contains("`->`"), "{}", e.message);
    }

    #[test]
    fn live_and_info() {
        assert_eq!(
            parse("LIVE").unwrap(),
            Statement::Live(Live { since: None })
        );
        assert_eq!(
            parse("LIVE SINCE 42").unwrap(),
            Statement::Live(Live { since: Some(42) })
        );
        assert_eq!(parse("INFO").unwrap(), Statement::Info(Info));
    }

    /// RELATE mutates; TRAVERSE/LIVE/INFO read. A read-only surface must be able
    /// to expose the reads — getting this wrong either blocks safe verbs or
    /// exposes a write.
    #[test]
    fn only_relate_counts_as_a_write_among_the_graph_verbs() {
        assert!(parse("RELATE a -> v -> b").unwrap().is_write());
        for src in ["TRAVERSE a ->", "LIVE", "LIVE SINCE 1", "INFO"] {
            assert!(!parse(src).unwrap().is_write(), "{src} only reads");
        }
    }

    #[test]
    fn a_write_sent_to_parse_query_is_refused_with_the_fix() {
        let e = crate::parse_query("DELETE doc1").unwrap_err();
        assert!(e.message.contains("is a write"), "{}", e.message);
        assert!(
            e.message.contains("parse()"),
            "names the right call: {}",
            e.message
        );
    }

    #[test]
    fn minimal_select() {
        let s = sel("SELECT *");
        assert_eq!(s, Select::default());
    }

    #[test]
    fn from_where_limit() {
        let s = sel("SELECT * FROM docs WHERE lang = 'en' LIMIT 10");
        assert_eq!(s.from.as_deref(), Some("docs"));
        assert_eq!(s.limit, Some(10));
        assert_eq!(
            s.where_,
            Some(Expr::Eq {
                key: "lang".into(),
                value: Value::Str("en".into())
            })
        );
    }

    /// AND binds tighter than OR. Getting this backwards is a silent
    /// wrong-answer bug, not a syntax error — so it gets its own test.
    #[test]
    fn and_binds_tighter_than_or() {
        let s = sel("SELECT * WHERE a = 1 OR b = 2 AND c = 3");
        match s.where_.unwrap() {
            Expr::Or(l, r) => {
                assert!(matches!(*l, Expr::Eq { .. }), "lhs of OR is `a = 1`");
                assert!(matches!(*r, Expr::And(..)), "rhs of OR is the AND group");
            }
            other => panic!("expected OR at the root, got {other:?}"),
        }
    }

    #[test]
    fn parens_override_precedence() {
        let s = sel("SELECT * WHERE (a = 1 OR b = 2) AND c = 3");
        assert!(
            matches!(s.where_.unwrap(), Expr::And(..)),
            "AND at the root"
        );
    }

    #[test]
    fn not_applies_to_the_next_primary_only() {
        let s = sel("SELECT * WHERE NOT a = 1 AND b = 2");
        match s.where_.unwrap() {
            Expr::And(l, _) => assert!(matches!(*l, Expr::Not(..)), "NOT binds to `a = 1`"),
            other => panic!("expected AND at the root, got {other:?}"),
        }
    }

    #[test]
    fn in_list_and_empty_in_list() {
        let s = sel("SELECT * WHERE tag IN ('a', 'b', 'c')");
        match s.where_.unwrap() {
            Expr::In { key, values } => {
                assert_eq!(key, "tag");
                assert_eq!(values.len(), 3);
            }
            other => panic!("{other:?}"),
        }
        // An empty list is legal and means "matches nothing" at lowering.
        assert!(matches!(
            sel("SELECT * WHERE tag IN ()").where_.unwrap(),
            Expr::In { .. }
        ));
    }

    #[test]
    fn exists_both_spellings() {
        let a = sel("SELECT * WHERE EXISTS(author)").where_.unwrap();
        let b = sel("SELECT * WHERE author IS EXISTS").where_.unwrap();
        assert_eq!(a, b, "EXISTS(k) and `k IS EXISTS` must mean the same thing");
    }

    #[test]
    fn geo_radius_and_box() {
        assert!(matches!(
            sel("SELECT * WHERE loc INSIDE RADIUS(51.5, -0.12, 5000)")
                .where_
                .unwrap(),
            Expr::GeoRadius { .. }
        ));
        assert!(matches!(
            sel("SELECT * WHERE loc INSIDE BOX(51.0, -1.0, 52.0, 0.5)")
                .where_
                .unwrap(),
            Expr::GeoBox { .. }
        ));
    }

    #[test]
    fn comparisons_require_numbers_not_strings() {
        // `year >= 'soon'` is nonsense; Range is numeric. Catch it at parse.
        let e = parse("SELECT * WHERE year >= 'soon'").unwrap_err();
        assert!(e.message.contains("expected a number"), "{}", e.message);
    }

    #[test]
    fn errors_carry_a_useful_span() {
        let src = "SELECT * WHERE year >= AND lang = 'en'";
        let e = parse(src).unwrap_err();
        let (s, en) = e.span;
        assert_eq!(
            &src[s..en],
            "AND",
            "the span must cover the offending token"
        );
    }

    #[test]
    fn trailing_junk_is_rejected() {
        assert!(parse("SELECT * LIMIT 5 nonsense").is_err());
    }

    #[test]
    fn a_column_list_is_rejected_rather_than_silently_ignored() {
        // The estate returns whole candidates; a projection would be a promise
        // the engine cannot keep, so it must not parse.
        assert!(parse("SELECT title, year FROM docs").is_err());
    }

    #[test]
    fn negative_and_fractional_limit_are_rejected() {
        assert!(parse("SELECT * LIMIT -1").is_err());
        assert!(parse("SELECT * LIMIT 2.5").is_err());
    }
}
