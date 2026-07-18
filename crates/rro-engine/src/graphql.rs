//! A hand-rolled GraphQL query surface over the estate.
//!
//! Zero dependencies, matching the rest of this tree (`rro-ql` is a hand-rolled
//! query language). No `async-graphql`, no `juniper`.
//!
//! ## GraphQL is a query language, not a transport
//!
//! "GraphQL needs HTTP" conflates the language with a convention. The spec is
//! transport-agnostic — it is *usually* served over HTTP, but it does not require
//! it. So [`execute`] is deliberately pure: `(query, estate, flow) -> JSON`. It
//! takes a query string and returns the response envelope, and it does not know
//! or care what carried the bytes.
//!
//! That means GraphQL rides RRO's existing **a2a NDJSON/TCP transport** — the
//! `graphql` verb in `handler.rs` calls `execute` — reusing the tokio-signal
//! spine, `watch` streaming, and capability-token auth rather than standing up a
//! parallel HTTP server. If a browser ever needs GraphQL-over-HTTP, that is a
//! thin adapter that reads a POST body and calls the same `execute`; the
//! capability lives here, once.
//!
//! ## What "GraphQL" means here, honestly
//!
//! This implements the **query and mutation** halves of GraphQL's execution
//! model: a real recursive-descent parser for selection sets with arguments, and
//! an executor that resolves each requested field and projects exactly the
//! sub-fields asked for — the property that distinguishes GraphQL from REST (the
//! client chooses the response shape). It is **not** the whole spec: no
//! subscriptions, fragments, variables or introspection yet. Those are additive
//! and named in the roadmap. Mutations write through the same estate/flow paths
//! the `index`/`tx` verbs use, and a reader-role token's mutation is refused
//! exactly like its `sql` write.
//!
//! The schema:
//!
//! ```graphql
//! type Query {
//!   health: Health
//!   collections: [Collection!]!
//!   document(id: String!): Document
//!   search(query: String!, topK: Int, mode: String): [Hit!]!
//! }
//! type Mutation {
//!   upsert(id: String!, text: String!): UpsertResult
//!   delete(id: String!): DeleteResult
//! }
//! ```
//!
//! Example — the client picks the shape:
//!
//! ```graphql
//! { search(query: "vector recall", topK: 3) { id score } }
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use rro_core::Recall;

use crate::flow::ReasonReadyObject;

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    LBrace,
    RBrace,
    LParen,
    RParen,
    Colon,
    Name(String),
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Eof,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let b = src.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\n' | b'\r' | b',' => i += 1,
            b'#' => {
                // Comment to end of line.
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            b'{' => {
                out.push(Tok::LBrace);
                i += 1;
            }
            b'}' => {
                out.push(Tok::RBrace);
                i += 1;
            }
            b'(' => {
                out.push(Tok::LParen);
                i += 1;
            }
            b')' => {
                out.push(Tok::RParen);
                i += 1;
            }
            b':' => {
                out.push(Tok::Colon);
                i += 1;
            }
            b'"' => {
                // String literal. No escapes beyond \" and \\ — enough for a
                // query surface, and a malformed string errors rather than
                // reading off the end.
                let mut s = String::new();
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' && i + 1 < b.len() {
                        i += 1;
                        s.push(b[i] as char);
                    } else {
                        s.push(b[i] as char);
                    }
                    i += 1;
                }
                if i >= b.len() {
                    return Err("unterminated string".into());
                }
                i += 1; // closing quote
                out.push(Tok::Str(s));
            }
            c if c == b'-' || c.is_ascii_digit() => {
                let start = i;
                i += 1;
                let mut is_float = false;
                while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'.') {
                    if b[i] == b'.' {
                        is_float = true;
                    }
                    i += 1;
                }
                let text = &src[start..i];
                if is_float {
                    out.push(Tok::Float(
                        text.parse().map_err(|_| format!("bad number: {text}"))?,
                    ));
                } else {
                    out.push(Tok::Int(
                        text.parse().map_err(|_| format!("bad number: {text}"))?,
                    ));
                }
            }
            c if c == b'_' || c.is_ascii_alphabetic() => {
                let start = i;
                while i < b.len() && (b[i] == b'_' || b[i].is_ascii_alphanumeric()) {
                    i += 1;
                }
                let word = &src[start..i];
                out.push(match word {
                    "true" => Tok::Bool(true),
                    "false" => Tok::Bool(false),
                    _ => Tok::Name(word.to_string()),
                });
            }
            _ => return Err(format!("unexpected character '{}'", c as char)),
        }
    }
    out.push(Tok::Eof);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Which GraphQL operation the document is: a read or a write.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Operation {
    Query,
    Mutation,
}

/// A JSON-ish argument value.
#[derive(Debug, Clone, PartialEq)]
enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// One requested field: name, arguments, and a (possibly empty) sub-selection.
#[derive(Debug, Clone)]
struct Field {
    name: String,
    args: BTreeMap<String, Value>,
    selection: Vec<Field>,
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::Eof)
    }
    fn next(&mut self) -> Tok {
        let t = self.toks.get(self.pos).cloned().unwrap_or(Tok::Eof);
        self.pos += 1;
        t
    }
    fn expect(&mut self, t: &Tok) -> Result<(), String> {
        if self.peek() == t {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {t:?}, found {:?}", self.peek()))
        }
    }

    /// A document is a selection set with an optional leading operation keyword:
    /// `query` (the default) or `mutation`. Anonymous operations only — named
    /// operations, variables and fragments are follow-ons.
    fn document(&mut self) -> Result<(Operation, Vec<Field>), String> {
        let op = match self.peek() {
            Tok::Name(n) if n == "query" => {
                self.next();
                Operation::Query
            }
            Tok::Name(n) if n == "mutation" => {
                self.next();
                Operation::Mutation
            }
            _ => Operation::Query,
        };
        let sel = self.selection_set()?;
        if self.peek() != &Tok::Eof {
            return Err(format!(
                "trailing tokens after operation: {:?}",
                self.peek()
            ));
        }
        Ok((op, sel))
    }

    fn selection_set(&mut self) -> Result<Vec<Field>, String> {
        self.expect(&Tok::LBrace)?;
        let mut fields = Vec::new();
        while self.peek() != &Tok::RBrace {
            if self.peek() == &Tok::Eof {
                return Err("unclosed selection set".into());
            }
            fields.push(self.field()?);
        }
        self.expect(&Tok::RBrace)?;
        Ok(fields)
    }

    fn field(&mut self) -> Result<Field, String> {
        let name = match self.next() {
            Tok::Name(n) => n,
            other => return Err(format!("expected a field name, found {other:?}")),
        };
        let args = if self.peek() == &Tok::LParen {
            self.arguments()?
        } else {
            BTreeMap::new()
        };
        let selection = if self.peek() == &Tok::LBrace {
            self.selection_set()?
        } else {
            Vec::new()
        };
        Ok(Field {
            name,
            args,
            selection,
        })
    }

    fn arguments(&mut self) -> Result<BTreeMap<String, Value>, String> {
        self.expect(&Tok::LParen)?;
        let mut args = BTreeMap::new();
        while self.peek() != &Tok::RParen {
            let key = match self.next() {
                Tok::Name(n) => n,
                other => return Err(format!("expected an argument name, found {other:?}")),
            };
            self.expect(&Tok::Colon)?;
            let val = match self.next() {
                Tok::Str(s) => Value::Str(s),
                Tok::Int(i) => Value::Int(i),
                Tok::Float(f) => Value::Float(f),
                Tok::Bool(b) => Value::Bool(b),
                other => return Err(format!("expected an argument value, found {other:?}")),
            };
            args.insert(key, val);
        }
        self.expect(&Tok::RParen)?;
        Ok(args)
    }
}

fn parse(src: &str) -> Result<(Operation, Vec<Field>), String> {
    let toks = lex(src)?;
    let mut p = Parser { toks, pos: 0 };
    p.document()
}

/// Whether `src` is a GraphQL **mutation** (a write). The caller gates it: the
/// `graphql` verb is admitted at reader level, but a reader's mutation is refused
/// exactly like a reader's `sql` write — the write surface is covered whichever
/// language carries it. A malformed document is not a mutation (it will fail in
/// `execute` with a parse error).
pub fn is_mutation(src: &str) -> bool {
    matches!(parse(src), Ok((Operation::Mutation, _)))
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Execute a GraphQL query string against the estate + flow, returning the
/// GraphQL response envelope `{ "data": {...} }` or `{ "errors": [...] }`.
///
/// The whole request succeeds or reports an error per the GraphQL convention;
/// resolver-level failures land under `errors` while other fields still resolve.
pub async fn execute(
    query: &str,
    estate: &connxism::Estate,
    flow: &Arc<ReasonReadyObject>,
) -> serde_json::Value {
    let (operation, roots) = match parse(query) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::json!({ "errors": [{ "message": format!("parse error: {e}") }] });
        }
    };

    let mut data = serde_json::Map::new();
    let mut errors: Vec<serde_json::Value> = Vec::new();

    for field in &roots {
        // The response key is the field name (aliases are a follow-on). Query
        // roots resolve reads; mutation roots resolve writes.
        let resolved = match operation {
            Operation::Query => resolve_root(field, estate, flow).await,
            Operation::Mutation => resolve_mutation(field, estate, flow).await,
        };
        match resolved {
            Ok(v) => {
                data.insert(field.name.clone(), v);
            }
            Err(e) => {
                data.insert(field.name.clone(), serde_json::Value::Null);
                errors.push(serde_json::json!({
                    "message": e,
                    "path": [field.name.clone()],
                }));
            }
        }
    }

    let mut out = serde_json::Map::new();
    out.insert("data".into(), serde_json::Value::Object(data));
    if !errors.is_empty() {
        out.insert("errors".into(), serde_json::Value::Array(errors));
    }
    serde_json::Value::Object(out)
}

async fn resolve_root(
    field: &Field,
    estate: &connxism::Estate,
    flow: &Arc<ReasonReadyObject>,
) -> Result<serde_json::Value, String> {
    match field.name.as_str() {
        "health" => {
            let h = estate.health().map_err(|e| e.to_string())?;
            let v = serde_json::to_value(&h).map_err(|e| e.to_string())?;
            Ok(project(&field.selection, &v))
        }
        "collections" => {
            let cols = estate.collections().map_err(|e| e.to_string())?;
            let arr: Vec<serde_json::Value> = cols
                .into_iter()
                .map(|(name, count)| {
                    project(
                        &field.selection,
                        &serde_json::json!({ "name": name, "count": count }),
                    )
                })
                .collect();
            Ok(serde_json::Value::Array(arr))
        }
        "document" => {
            let id = match field.args.get("id") {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err("document(id: String!) requires a string `id`".into()),
            };
            match estate.recall().doc(&id).await.map_err(|e| e.to_string())? {
                Some(doc) => {
                    let obj = serde_json::json!({
                        "id": doc.id, "text": doc.text, "metadata": doc.metadata,
                    });
                    Ok(project(&field.selection, &obj))
                }
                None => Ok(serde_json::Value::Null),
            }
        }
        "search" => {
            let query = match field.args.get("query") {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err("search(query: String!) requires a string `query`".into()),
            };
            let top_k = match field.args.get("topK") {
                Some(Value::Int(i)) => (*i).max(0) as usize,
                None => 10,
                _ => return Err("topK must be an integer".into()),
            };
            let mode = match field.args.get("mode") {
                Some(Value::Str(s)) if s.eq_ignore_ascii_case("dbsf") => rro_core::FusionMode::Dbsf,
                _ => rro_core::FusionMode::Rrf,
            };
            let vector = flow.embed_query(&query).await.map_err(|e| e.to_string())?;
            let hits = estate
                .recall()
                .query(rro_core::EstateQuery {
                    text: Some(query),
                    vector: Some(vector),
                    top_k,
                    fusion_mode: mode,
                    ..Default::default()
                })
                .await
                .map_err(|e| e.to_string())?;
            let arr: Vec<serde_json::Value> = hits
                .into_iter()
                .map(|c| {
                    let obj = serde_json::json!({
                        "id": c.id.as_str(), "text": c.text, "score": c.score, "metadata": c.metadata,
                    });
                    project(&field.selection, &obj)
                })
                .collect();
            Ok(serde_json::Value::Array(arr))
        }
        other => Err(format!("unknown field `{other}` on Query")),
    }
}

/// Resolve one mutation root field. Writes go through the same estate/flow paths
/// the a2a `index`/`tx` verbs use — GraphQL adds a shape-selecting surface, not a
/// second write implementation.
///
/// ```graphql
/// type Mutation {
///   upsert(id: String!, text: String!): UpsertResult
///   delete(id: String!): DeleteResult
/// }
/// ```
async fn resolve_mutation(
    field: &Field,
    estate: &connxism::Estate,
    flow: &Arc<ReasonReadyObject>,
) -> Result<serde_json::Value, String> {
    let str_arg = |name: &str| match field.args.get(name) {
        Some(Value::Str(s)) => Ok(s.clone()),
        _ => Err(format!("{}(...) requires a string `{name}`", field.name)),
    };
    match field.name.as_str() {
        "upsert" => {
            let id = str_arg("id")?;
            let text = str_arg("text")?;
            // Embed with the flow's embedder, then write to the SAME estate the
            // read resolvers query — so an upsert is immediately visible to a
            // subsequent `search`/`document`, not written to a parallel store.
            let doc = rro_core::Document {
                id: rro_core::Id::from(id.clone()),
                text,
                metadata: rro_core::Metadata::new(),
            };
            let records = flow
                .embed_documents(vec![doc])
                .await
                .map_err(|e| e.to_string())?;
            estate
                .recall()
                .upsert(records)
                .await
                .map_err(|e| e.to_string())?;
            Ok(project(
                &field.selection,
                &serde_json::json!({ "id": id, "indexed": 1 }),
            ))
        }
        "delete" => {
            let id = str_arg("id")?;
            estate
                .recall()
                .remove(&rro_core::Id::from(id.clone()))
                .await
                .map_err(|e| e.to_string())?;
            Ok(project(
                &field.selection,
                &serde_json::json!({ "id": id, "deleted": true }),
            ))
        }
        other => Err(format!("unknown field `{other}` on Mutation")),
    }
}

/// Project a resolved object down to exactly the requested sub-fields — the
/// core of GraphQL: the client chose the shape, so the server returns that shape
/// and no more. An empty selection returns the whole object (scalar leaf).
fn project(selection: &[Field], obj: &serde_json::Value) -> serde_json::Value {
    if selection.is_empty() {
        return obj.clone();
    }
    let mut out = serde_json::Map::new();
    for f in selection {
        let v = obj.get(&f.name).cloned().unwrap_or(serde_json::Value::Null);
        // Recurse for nested selections (e.g. metadata subfields).
        let projected = if f.selection.is_empty() {
            v
        } else {
            project(&f.selection, &v)
        };
        out.insert(f.name.clone(), projected);
    }
    serde_json::Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_selection_with_arguments() {
        let q = r#"{ search(query: "hello world", topK: 3, mode: "dbsf") { id score metadata { kind } } }"#;
        let (op, roots) = parse(q).unwrap();
        assert_eq!(op, Operation::Query);
        assert_eq!(roots.len(), 1);
        let s = &roots[0];
        assert_eq!(s.name, "search");
        assert_eq!(s.args.get("query"), Some(&Value::Str("hello world".into())));
        assert_eq!(s.args.get("topK"), Some(&Value::Int(3)));
        // Nested selection is parsed.
        let meta = s.selection.iter().find(|f| f.name == "metadata").unwrap();
        assert_eq!(meta.selection[0].name, "kind");
    }

    #[test]
    fn optional_query_keyword_is_accepted() {
        assert!(parse("query { health }").is_ok());
        assert!(parse("{ health }").is_ok());
    }

    #[test]
    fn mutation_keyword_is_recognized() {
        let (op, roots) =
            parse(r#"mutation { upsert(id: "d1", text: "hi") { id indexed } }"#).unwrap();
        assert_eq!(op, Operation::Mutation);
        assert_eq!(roots[0].name, "upsert");
        assert_eq!(roots[0].args.get("id"), Some(&Value::Str("d1".into())));

        // is_mutation distinguishes writes from reads and malformed input.
        assert!(is_mutation(r#"mutation { delete(id: "d1") { id } }"#));
        assert!(!is_mutation("{ health }"));
        assert!(!is_mutation("query { search(query: \"x\") { id } }"));
        assert!(!is_mutation("mutation { "), "malformed is not a mutation");
    }

    #[test]
    fn a_malformed_query_is_an_error_not_a_panic() {
        assert!(parse("{ search(query: }").is_err());
        assert!(parse("{ unclosed ").is_err());
        assert!(parse(r#"{ x(a: "unterminated }"#).is_err());
    }

    #[test]
    fn projection_returns_only_requested_fields() {
        let obj = serde_json::json!({ "id": "a", "text": "hello", "score": 0.9 });
        let (_op, sel) = parse("{ id score }").unwrap();
        let out = project(&sel, &obj);
        assert_eq!(out.get("id").unwrap(), "a");
        assert_eq!(out.get("score").unwrap(), 0.9);
        assert!(
            out.get("text").is_none(),
            "unrequested field must be absent"
        );
    }

    #[test]
    fn projection_recurses_into_nested_objects() {
        let obj = serde_json::json!({
            "id": "a",
            "metadata": { "kind": "doc", "team": "eng" }
        });
        let (_op, sel) = parse("{ id metadata { kind } }").unwrap();
        let out = project(&sel, &obj);
        let meta = out.get("metadata").unwrap();
        assert_eq!(meta.get("kind").unwrap(), "doc");
        assert!(
            meta.get("team").is_none(),
            "nested projection drops unrequested"
        );
    }
}
