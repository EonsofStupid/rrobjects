//! Recursive-descent parser: tokens → [`Statement`].
//!
//! Precedence, loosest to tightest: `OR` < `AND` < `NOT` < primary. Standard SQL
//! shape, so `a = 1 OR b = 2 AND c = 3` parses as `a=1 OR (b=2 AND c=3)` — which
//! is what anyone who has written SQL expects, and getting it backwards is a
//! silent wrong-answer bug rather than a syntax error.

use crate::ast::{CmpOp, Expr, Select, Statement, Value};
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
        let k = self.tokens[self.pos.min(self.tokens.len() - 1)].kind.clone();
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
            other => Err(QlError::new(format!("expected a number, found {other}"), span)),
        }
    }

    fn value(&mut self) -> Result<Value, QlError> {
        let span = self.span();
        match self.bump() {
            TokenKind::Str(s) => Ok(Value::Str(s)),
            TokenKind::Num(n) => Ok(Value::Num(n)),
            TokenKind::Bool(b) => Ok(Value::Bool(b)),
            TokenKind::Null => Ok(Value::Null),
            other => Err(QlError::new(format!("expected a value, found {other}"), span)),
        }
    }

    fn statement(&mut self) -> Result<Statement, QlError> {
        if self.eat(&TokenKind::Select) {
            return Ok(Statement::Select(self.select()?));
        }
        Err(QlError::new(
            format!("expected SELECT, found {}", self.peek()),
            self.span(),
        ))
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
            let span = self.span();
            let n = self.number()?;
            if n < 0.0 || n.fract() != 0.0 {
                return Err(QlError::new(
                    format!("LIMIT must be a non-negative whole number, found {n}"),
                    span,
                ));
            }
            s.limit = Some(n as usize);
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
        }
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
        assert!(matches!(s.where_.unwrap(), Expr::And(..)), "AND at the root");
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
        assert_eq!(&src[s..en], "AND", "the span must cover the offending token");
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
