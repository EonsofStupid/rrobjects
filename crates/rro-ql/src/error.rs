//! RRQL errors — every one carries the span that caused it.
//!
//! A query language whose errors say "syntax error" is a query language people
//! stop using. `caret(src)` renders the offending text so a CLI, the MCP tool
//! and a REST body can all show the same thing.

use crate::lexer::LexError;

/// A parse or lowering failure.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message} (at {}..{})", span.0, span.1)]
pub struct QlError {
    /// What went wrong, in the caller's terms.
    pub message: String,
    /// Byte range in the source.
    pub span: (usize, usize),
}

impl QlError {
    /// An error at `span`.
    pub fn new(message: impl Into<String>, span: (usize, usize)) -> Self {
        QlError {
            message: message.into(),
            span,
        }
    }

    /// Render the failure against its source, with a caret under the span:
    ///
    /// ```text
    /// expected a value (at 24..25)
    ///   SELECT * WHERE year >= AND lang = 'en'
    ///                          ^^^
    /// ```
    pub fn caret(&self, src: &str) -> String {
        let (s, e) = self.span;
        let s = s.min(src.len());
        let e = e.clamp(s, src.len());
        // Locate the line containing the span so long queries stay readable.
        let line_start = src[..s].rfind('\n').map_or(0, |i| i + 1);
        let line_end = src[e..].find('\n').map_or(src.len(), |i| e + i);
        let line = &src[line_start..line_end];
        let pad = " ".repeat(s - line_start);
        let width = (e - s).max(1);
        format!("{self}\n  {line}\n  {pad}{}", "^".repeat(width))
    }
}

impl From<LexError> for QlError {
    fn from(e: LexError) -> Self {
        QlError {
            message: e.message,
            span: e.span,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caret_points_at_the_offending_text() {
        let src = "SELECT * WHERE year >= AND lang = 'en'";
        let e = QlError::new("expected a value", (23, 26));
        let rendered = e.caret(src);
        // The caret line must sit under `AND`.
        let caret_line = rendered.lines().nth(2).unwrap();
        assert_eq!(caret_line.trim_start_matches(' '), "^^^");
        assert_eq!(
            caret_line.len() - 2,
            23 + 3,
            "caret is under the right column"
        );
        assert!(rendered.contains("expected a value"));
    }

    #[test]
    fn caret_survives_an_out_of_range_span() {
        // A malformed span must not panic the engine mid-query.
        let e = QlError::new("boom", (500, 900));
        let _ = e.caret("short");
    }

    #[test]
    fn caret_shows_only_the_offending_line() {
        let src = "SELECT *\nWHERE a = ?\nLIMIT 5";
        let e = QlError::new("bad value", (19, 20));
        let out = e.caret(src);
        assert!(out.contains("WHERE a = ?"), "shows the failing line");
        assert!(!out.contains("LIMIT 5"), "does not dump the whole query");
    }
}
