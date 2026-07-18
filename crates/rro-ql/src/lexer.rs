//! The RRQL lexer: text → tokens.
//!
//! Hand-rolled, zero dependencies. RRO has no nom/pest/logos anywhere, and a
//! query language is not a reason to start — the same call the repo already made
//! twice for HTTP (`ops.rs` and both model clients are hand-rolled on tokio).
//!
//! Every token carries its byte span. That is not decoration: a query language
//! whose errors say "syntax error" is a query language people stop using, and
//! the span is what lets [`crate::QlError`] point at the offending text.

use std::fmt;

/// A lexical token with its source span.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// What it is.
    pub kind: TokenKind,
    /// Byte range in the source.
    pub span: (usize, usize),
}

/// The token vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // literals
    /// Bare or quoted identifier (`title`, `` `odd name` ``).
    Ident(String),
    /// A quoted string (`'x'` or `"x"`).
    Str(String),
    /// A number (integers and floats are one kind; JSON does not distinguish).
    Num(f64),
    /// `true` / `false`.
    Bool(bool),
    /// `null`.
    Null,

    // keywords
    /// `SELECT`
    Select,
    /// `FROM`
    From,
    /// `WHERE`
    Where,
    /// `LIMIT`
    Limit,
    /// `AND`
    And,
    /// `OR`
    Or,
    /// `NOT`
    Not,
    /// `IN`
    In,
    /// `IS`
    Is,
    /// `EXISTS`
    Exists,
    /// `INSIDE` — geo containment.
    Inside,
    /// `RADIUS` — geo radius.
    Radius,
    /// `BOX` — geo bounding box.
    BoxKw,
    /// `DEFINE`
    Define,
    /// `REMOVE`
    Remove,
    /// `INDEX`
    Index,
    /// `ALIAS`
    Alias,
    /// `COLLECTION`
    Collection,
    /// `FIELD`
    Field,
    /// `TYPE`
    Type,
    /// `ON`
    On,
    /// `FOR`
    For,
    /// `UPDATE`
    Update,
    /// `DELETE`
    Delete,
    /// `SET`
    Set,
    /// `CONTENT`
    Content,
    /// `PAYLOAD`
    Payload,
    /// `RELATE`
    Relate,
    /// `TRAVERSE`
    Traverse,
    /// `DEPTH`
    Depth,
    /// `LIVE`
    Live,
    /// `SINCE`
    Since,
    /// `INFO`
    Info,
    /// `->`
    ArrowOut,
    /// `<-`
    ArrowIn,
    /// `<->`
    ArrowBoth,

    // punctuation / operators
    /// `,`
    Comma,
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `*`
    Star,
    /// `=`
    Eq,
    /// `!=`
    Neq,
    /// `<`
    Lt,
    /// `<=`
    Lte,
    /// `>`
    Gt,
    /// `>=`
    Gte,

    /// End of input.
    Eof,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TokenKind::Ident(i) => return write!(f, "identifier `{i}`"),
            TokenKind::Str(s) => return write!(f, "string {s:?}"),
            TokenKind::Num(n) => return write!(f, "number {n}"),
            TokenKind::Bool(b) => return write!(f, "{b}"),
            TokenKind::Null => "null",
            TokenKind::Select => "SELECT",
            TokenKind::From => "FROM",
            TokenKind::Where => "WHERE",
            TokenKind::Limit => "LIMIT",
            TokenKind::And => "AND",
            TokenKind::Or => "OR",
            TokenKind::Not => "NOT",
            TokenKind::In => "IN",
            TokenKind::Is => "IS",
            TokenKind::Exists => "EXISTS",
            TokenKind::Inside => "INSIDE",
            TokenKind::Radius => "RADIUS",
            TokenKind::BoxKw => "BOX",
            TokenKind::Define => "DEFINE",
            TokenKind::Remove => "REMOVE",
            TokenKind::Index => "INDEX",
            TokenKind::Alias => "ALIAS",
            TokenKind::Collection => "COLLECTION",
            TokenKind::Field => "FIELD",
            TokenKind::Type => "TYPE",
            TokenKind::On => "ON",
            TokenKind::For => "FOR",
            TokenKind::Update => "UPDATE",
            TokenKind::Delete => "DELETE",
            TokenKind::Set => "SET",
            TokenKind::Content => "CONTENT",
            TokenKind::Payload => "PAYLOAD",
            TokenKind::Relate => "RELATE",
            TokenKind::Traverse => "TRAVERSE",
            TokenKind::Depth => "DEPTH",
            TokenKind::Live => "LIVE",
            TokenKind::Since => "SINCE",
            TokenKind::Info => "INFO",
            TokenKind::ArrowOut => "`->`",
            TokenKind::ArrowIn => "`<-`",
            TokenKind::ArrowBoth => "`<->`",
            TokenKind::Comma => "`,`",
            TokenKind::LParen => "`(`",
            TokenKind::RParen => "`)`",
            TokenKind::Star => "`*`",
            TokenKind::Eq => "`=`",
            TokenKind::Neq => "`!=`",
            TokenKind::Lt => "`<`",
            TokenKind::Lte => "`<=`",
            TokenKind::Gt => "`>`",
            TokenKind::Gte => "`>=`",
            TokenKind::Eof => "end of input",
        };
        write!(f, "{s}")
    }
}

/// A lexing failure, with the span that caused it.
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    /// What went wrong.
    pub message: String,
    /// Byte range in the source.
    pub span: (usize, usize),
}

/// Tokenize `src`.
///
/// Keywords are case-insensitive (`select` == `SELECT`), identifiers are not —
/// SQL convention, and the estate's metadata keys are case-sensitive JSON keys.
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let b = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < b.len() {
        let c = b[i] as char;

        // whitespace
        if c.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // line comment: -- ...
        if c == '-' && i + 1 < b.len() && b[i + 1] == b'-' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        let start = i;

        // Arrows before operators, LONGEST first: `<->` must not lex as `<-`
        // then `>`, and `<-` must not lex as `<` then `-`. Same discipline as
        // `<=` vs `<`, one length up.
        if let Some(three) = src.get(i..i + 3) {
            if three == "<->" {
                out.push(Token {
                    kind: TokenKind::ArrowBoth,
                    span: (start, i + 3),
                });
                i += 3;
                continue;
            }
        }
        if let Some(two) = src.get(i..i + 2) {
            let arrow = match two {
                "->" => Some(TokenKind::ArrowOut),
                "<-" => Some(TokenKind::ArrowIn),
                _ => None,
            };
            if let Some(k) = arrow {
                out.push(Token {
                    kind: k,
                    span: (start, i + 2),
                });
                i += 2;
                continue;
            }
        }

        // Two-char operators first, so `<=` never lexes as `<` then `=`.
        //
        // `src.get(i..i+2)` and NOT `&src[i..i+2]`: the latter panics when i+2
        // lands inside a multibyte char, so a single `Ѩ` anywhere outside a
        // string would crash the process. On a query surface that is a remote
        // denial-of-service — found by the `arbitrary_input_never_panics`
        // property, not by any example test (the unicode example only covered
        // unicode INSIDE strings, which takes the char-safe path).
        if let Some(two) = src.get(i..i + 2) {
            let kind = match two {
                "!=" => Some(TokenKind::Neq),
                "<=" => Some(TokenKind::Lte),
                ">=" => Some(TokenKind::Gte),
                _ => None,
            };
            if let Some(k) = kind {
                out.push(Token {
                    kind: k,
                    span: (start, i + 2),
                });
                i += 2;
                continue;
            }
        }

        // single-char punctuation
        let single = match c {
            ',' => Some(TokenKind::Comma),
            '(' => Some(TokenKind::LParen),
            ')' => Some(TokenKind::RParen),
            '*' => Some(TokenKind::Star),
            '=' => Some(TokenKind::Eq),
            '<' => Some(TokenKind::Lt),
            '>' => Some(TokenKind::Gt),
            _ => None,
        };
        if let Some(k) = single {
            out.push(Token {
                kind: k,
                span: (start, i + 1),
            });
            i += 1;
            continue;
        }

        // quoted string: '...' or "..." with \\ and \' escapes
        if c == '\'' || c == '"' {
            let quote = b[i];
            i += 1;
            let mut s = String::new();
            loop {
                if i >= b.len() {
                    return Err(LexError {
                        message: "unterminated string".to_string(),
                        span: (start, b.len()),
                    });
                }
                if b[i] == b'\\' && i + 1 < b.len() {
                    // Keep escapes minimal and predictable rather than inventing
                    // a dialect: only the quote and the backslash itself.
                    let esc = b[i + 1];
                    if esc == quote || esc == b'\\' {
                        s.push(esc as char);
                        i += 2;
                        continue;
                    }
                }
                if b[i] == quote {
                    i += 1;
                    break;
                }
                let ch_start = i;
                let ch = src[i..].chars().next().unwrap();
                i = ch_start + ch.len_utf8();
                s.push(ch);
            }
            out.push(Token {
                kind: TokenKind::Str(s),
                span: (start, i),
            });
            continue;
        }

        // backtick identifier: `odd name`
        if c == '`' {
            i += 1;
            let from = i;
            while i < b.len() && b[i] != b'`' {
                i += 1;
            }
            if i >= b.len() {
                return Err(LexError {
                    message: "unterminated `identifier`".to_string(),
                    span: (start, b.len()),
                });
            }
            let name = src[from..i].to_string();
            i += 1;
            out.push(Token {
                kind: TokenKind::Ident(name),
                span: (start, i),
            });
            continue;
        }

        // number: -?digits(.digits)?(e[+-]?digits)?
        if c.is_ascii_digit()
            || (c == '-' && i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit())
        {
            i += 1;
            while i < b.len() && ((b[i] as char).is_ascii_digit() || b[i] == b'.') {
                i += 1;
            }
            if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
                i += 1;
                if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                    i += 1;
                }
                while i < b.len() && (b[i] as char).is_ascii_digit() {
                    i += 1;
                }
            }
            let text = &src[start..i];
            let n: f64 = text.parse().map_err(|_| LexError {
                message: format!("`{text}` is not a number"),
                span: (start, i),
            })?;
            out.push(Token {
                kind: TokenKind::Num(n),
                span: (start, i),
            });
            continue;
        }

        // bare identifier / keyword
        if c.is_ascii_alphabetic() || c == '_' {
            while i < b.len()
                && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.')
            {
                i += 1;
            }
            let word = &src[start..i];
            let kind = match word.to_ascii_uppercase().as_str() {
                "SELECT" => TokenKind::Select,
                "FROM" => TokenKind::From,
                "WHERE" => TokenKind::Where,
                "LIMIT" => TokenKind::Limit,
                "AND" => TokenKind::And,
                "OR" => TokenKind::Or,
                "NOT" => TokenKind::Not,
                "IN" => TokenKind::In,
                "IS" => TokenKind::Is,
                "EXISTS" => TokenKind::Exists,
                "INSIDE" => TokenKind::Inside,
                "RADIUS" => TokenKind::Radius,
                "BOX" => TokenKind::BoxKw,
                "DEFINE" => TokenKind::Define,
                "REMOVE" => TokenKind::Remove,
                "INDEX" => TokenKind::Index,
                "ALIAS" => TokenKind::Alias,
                "COLLECTION" => TokenKind::Collection,
                "FIELD" => TokenKind::Field,
                "TYPE" => TokenKind::Type,
                "ON" => TokenKind::On,
                "FOR" => TokenKind::For,
                "UPDATE" => TokenKind::Update,
                "DELETE" => TokenKind::Delete,
                "SET" => TokenKind::Set,
                "CONTENT" => TokenKind::Content,
                "PAYLOAD" => TokenKind::Payload,
                "RELATE" => TokenKind::Relate,
                "TRAVERSE" => TokenKind::Traverse,
                "DEPTH" => TokenKind::Depth,
                "LIVE" => TokenKind::Live,
                "SINCE" => TokenKind::Since,
                "INFO" => TokenKind::Info,
                "TRUE" => TokenKind::Bool(true),
                "FALSE" => TokenKind::Bool(false),
                "NULL" => TokenKind::Null,
                _ => TokenKind::Ident(word.to_string()),
            };
            out.push(Token {
                kind,
                span: (start, i),
            });
            continue;
        }

        return Err(LexError {
            message: format!("unexpected character `{c}`"),
            span: (start, start + c.len_utf8()),
        });
    }

    out.push(Token {
        kind: TokenKind::Eof,
        span: (b.len(), b.len()),
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn keywords_are_case_insensitive_identifiers_are_not() {
        assert_eq!(kinds("select")[0], TokenKind::Select);
        assert_eq!(kinds("SeLeCt")[0], TokenKind::Select);
        // an identifier keeps its case — metadata keys are case-sensitive JSON
        assert_eq!(kinds("Title")[0], TokenKind::Ident("Title".into()));
    }

    #[test]
    fn two_char_operators_beat_one_char_ones() {
        // The classic lexer bug: `<=` lexing as `<` then `=`.
        assert_eq!(kinds("<="), vec![TokenKind::Lte, TokenKind::Eof]);
        assert_eq!(kinds(">="), vec![TokenKind::Gte, TokenKind::Eof]);
        assert_eq!(kinds("!="), vec![TokenKind::Neq, TokenKind::Eof]);
        assert_eq!(
            kinds("< ="),
            vec![TokenKind::Lt, TokenKind::Eq, TokenKind::Eof]
        );
    }

    #[test]
    fn strings_handle_both_quotes_and_escapes() {
        assert_eq!(kinds("'hi'")[0], TokenKind::Str("hi".into()));
        assert_eq!(kinds("\"hi\"")[0], TokenKind::Str("hi".into()));
        assert_eq!(kinds(r"'it\'s'")[0], TokenKind::Str("it's".into()));
        assert_eq!(kinds(r"'a\\b'")[0], TokenKind::Str(r"a\b".into()));
        // a quote of the OTHER kind is just a character
        assert_eq!(
            kinds("'say \"hi\"'")[0],
            TokenKind::Str("say \"hi\"".into())
        );
    }

    #[test]
    fn unterminated_string_points_at_the_opening_quote() {
        let e = lex("'oops").unwrap_err();
        assert_eq!(
            e.span.0, 0,
            "the span must start at the quote that opened it"
        );
        assert!(e.message.contains("unterminated"));
    }

    #[test]
    fn numbers_cover_int_float_negative_and_exponent() {
        assert_eq!(kinds("42")[0], TokenKind::Num(42.0));
        assert_eq!(kinds("3.5")[0], TokenKind::Num(3.5));
        assert_eq!(kinds("-7")[0], TokenKind::Num(-7.0));
        assert_eq!(kinds("1e3")[0], TokenKind::Num(1000.0));
        assert_eq!(kinds("-2.5e-2")[0], TokenKind::Num(-0.025));
    }

    #[test]
    fn minus_before_a_non_digit_is_not_a_number() {
        // `--` is a comment, and a bare `-` is not part of the v1 grammar; the
        // point is that neither silently lexes as a number.
        assert_eq!(
            kinds("-- a comment\n42"),
            vec![TokenKind::Num(42.0), TokenKind::Eof]
        );
    }

    #[test]
    fn dotted_and_backticked_identifiers() {
        assert_eq!(
            kinds("meta.author")[0],
            TokenKind::Ident("meta.author".into())
        );
        assert_eq!(kinds("`odd name`")[0], TokenKind::Ident("odd name".into()));
    }

    #[test]
    fn unicode_in_strings_does_not_split_a_char() {
        // Byte-indexed lexers love to panic here.
        assert_eq!(
            kinds("'héllo → 世界'")[0],
            TokenKind::Str("héllo → 世界".into())
        );
    }

    #[test]
    fn spans_point_at_the_real_text() {
        let toks = lex("SELECT * WHERE a = 1").unwrap();
        let (s, e) = toks[3].span; // `a`
        assert_eq!(&"SELECT * WHERE a = 1"[s..e], "a");
    }

    /// REGRESSION: a bare multibyte char outside a string used to PANIC —
    /// `&src[i..i+2]` in the two-char-operator check split it. On a query
    /// surface that is a remote DoS: one `Ѩ` kills the node. Found by the
    /// `arbitrary_input_never_panics` property, not by an example.
    #[test]
    fn a_bare_multibyte_char_errors_instead_of_panicking() {
        for src in ["Ѩ", "a Ѩ b", "SELECT * WHERE Ѩ = 1", "→", "世界"] {
            let r = lex(src);
            assert!(
                r.is_err(),
                "{src:?} should be a lex error, not a panic or a token"
            );
        }
    }

    #[test]
    fn multibyte_right_before_a_real_two_char_operator_still_lexes() {
        // The fix must not break the operator it guards.
        let toks = lex("'é' <= 3").unwrap();
        assert_eq!(toks[1].kind, TokenKind::Lte);
    }

    #[test]
    fn unexpected_character_is_an_error_not_a_silent_skip() {
        let e = lex("a # b").unwrap_err();
        assert!(
            e.message.contains('#'),
            "the error must name the character: {}",
            e.message
        );
    }
}
