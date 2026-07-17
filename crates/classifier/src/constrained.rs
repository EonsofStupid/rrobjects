//! The constrained-decode readiness classifier — a real model judging whether
//! the retrieved context is enough to reason on.
//!
//! "Judges whether what it found is enough to reason on" is RRO's headline
//! claim, and until now the only implementation was a heuristic
//! ([`crate::HeuristicClassifier`]). The heuristic is kept: it is the weightless
//! default, it needs no server, and it is the floor this must beat.
//!
//! Same shape as the embedder/reranker HTTP backends: hand-rolled HTTP/1.1 on
//! tokio, zero new deps, talking OpenAI `/v1/chat/completions` to whatever brain
//! engine owns the port — vLLM, llama.cpp, or candle/mistral.rs all serve it.
//!
//! Two things make this trustworthy rather than a vibe:
//!
//! 1. **The schema is BUILT FROM the typed labels** ([`ReadyLabel::ALL`]), so the
//!    grammar and the code cannot drift. A label added in Rust appears in the
//!    grammar automatically; there is no second list to forget.
//! 2. **Confidence comes from logprobs**, not from asking the model to rate
//!    itself. A model's self-reported confidence is a token like any other; the
//!    logprob is the model's actual distribution.
//!
//! `response_format: {type: json_schema, strict: true}` + `temperature: 0` means
//! the server enforces the grammar. If it doesn't, parsing fails loudly rather
//! than guessing — a classifier that silently invents a label is worse than none.

use std::time::Duration;

use async_trait::async_trait;
use rro_core::{Candidate, Classifier, Readiness, Result, RroError};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// The verdict vocabulary.
///
/// The JSON schema handed to the constrained decoder is **generated from this
/// enum**, so the grammar the model is held to and the type the code matches on
/// cannot drift apart. Adding a variant here adds it to the grammar; there is no
/// second list to forget to update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadyLabel {
    /// The context answers the query.
    Ready,
    /// Nothing retrieved bears on the query.
    Insufficient,
    /// Related material, but it does not settle the question.
    Ambiguous,
    /// Retrieved context contradicts itself.
    Conflicting,
}

impl ReadyLabel {
    /// Every label, in one place. The grammar is derived from this array.
    pub const ALL: [ReadyLabel; 4] = [
        ReadyLabel::Ready,
        ReadyLabel::Insufficient,
        ReadyLabel::Ambiguous,
        ReadyLabel::Conflicting,
    ];

    /// The wire tag.
    pub fn tag(self) -> &'static str {
        match self {
            ReadyLabel::Ready => "ready",
            ReadyLabel::Insufficient => "insufficient",
            ReadyLabel::Ambiguous => "ambiguous",
            ReadyLabel::Conflicting => "conflicting",
        }
    }

    /// Only `Ready` means the reasoner may proceed unqualified.
    pub fn is_ready(self) -> bool {
        matches!(self, ReadyLabel::Ready)
    }
}

impl std::str::FromStr for ReadyLabel {
    type Err = RroError;

    fn from_str(s: &str) -> Result<Self> {
        ReadyLabel::ALL
            .into_iter()
            .find(|l| l.tag() == s.trim())
            .ok_or_else(|| {
                RroError::Classify(format!(
                    "label `{s}` is not in the schema — the server did not enforce \
                     strict json_schema, so its output cannot be trusted"
                ))
            })
    }
}

/// The JSON schema, generated from [`ReadyLabel::ALL`].
///
/// Generated, not hand-written: a hand-written copy is a second source of truth
/// that silently rots the first time a label is added.
pub fn schema() -> String {
    let labels = ReadyLabel::ALL
        .iter()
        .map(|l| format!("\"{}\"", l.tag()))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"type\":\"object\",\"properties\":{{\
         \"label\":{{\"type\":\"string\",\"enum\":[{labels}]}},\
         \"rationale\":{{\"type\":\"string\"}}}},\
         \"required\":[\"label\",\"rationale\"],\"additionalProperties\":false}}"
    )
}

/// Config for the constrained-decode classifier.
#[derive(Debug, Clone)]
pub struct ConstrainedConfig {
    /// Full URL, e.g. `http://127.0.0.1:8091/v1/chat/completions`.
    pub endpoint: String,
    /// `model` field. `None` = discover from `/v1/models`.
    pub model: Option<String>,
    /// How many candidates to show the judge.
    pub context_k: usize,
    /// Bytes of each candidate's text to show.
    pub snippet_bytes: usize,
    /// Per-request timeout.
    pub timeout: Duration,
}

impl ConstrainedConfig {
    /// Config pointed at `endpoint`.
    pub fn new(endpoint: impl Into<String>) -> Self {
        ConstrainedConfig {
            endpoint: endpoint.into(),
            model: None,
            context_k: 5,
            snippet_bytes: 500,
            timeout: Duration::from_secs(60),
        }
    }
}

/// Readiness judged by a real model under a strict grammar.
#[derive(Debug)]
pub struct ConstrainedClassifier {
    cfg: ConstrainedConfig,
    host: String,
    port: u16,
    path: String,
    model: String,
    name: String,
}

impl ConstrainedClassifier {
    /// Resolve the model and verify the brain answers.
    pub async fn connect(cfg: ConstrainedConfig) -> Result<Self> {
        let (host, port, path) = parse_url(&cfg.endpoint)?;
        let model = match cfg.model.clone() {
            Some(m) => m,
            None => discover_model(&host, port, cfg.timeout).await?,
        };
        let name = format!("constrained-{}", model.rsplit('/').next().unwrap_or(&model));
        let me = ConstrainedClassifier {
            cfg,
            host,
            port,
            path,
            model,
            name,
        };
        // Fail at startup, not at the first query.
        me.judge("probe", &[]).await?;
        Ok(me)
    }

    async fn judge(&self, query: &str, context: &[Candidate]) -> Result<Readiness> {
        let mut ctx = String::new();
        for (i, c) in context.iter().take(self.cfg.context_k).enumerate() {
            let t = &c.text;
            let cut = t
                .char_indices()
                .nth(self.cfg.snippet_bytes)
                .map_or(t.len(), |(i, _)| i);
            ctx.push_str(&format!("[{}] {}\n", i + 1, &t[..cut]));
        }
        if ctx.is_empty() {
            ctx.push_str("(nothing was retrieved)\n");
        }

        let sys = "You judge whether retrieved context is sufficient to answer a query. \
                   `ready`: the context answers it. `insufficient`: nothing retrieved bears on it. \
                   `ambiguous`: related but does not settle it. `conflicting`: the context \
                   contradicts itself. Judge only what is shown — do not use outside knowledge. \
                   Answer ONLY with the schema.";
        let user = format!("QUERY:\n{query}\n\nRETRIEVED CONTEXT:\n{ctx}");

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "logprobs": true,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "readiness",
                    "strict": true,
                    "schema": serde_json::from_str::<serde_json::Value>(&schema())
                        .map_err(|e| err(format!("own schema is not valid JSON: {e}")))?
                }
            }
        })
        .to_string();

        let req = format!(
            "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{}",
            self.path,
            self.host,
            self.port,
            body.len(),
            body
        );
        let raw = roundtrip(&self.host, self.port, &req, self.cfg.timeout).await?;
        let text = String::from_utf8_lossy(&raw);
        let (head, json) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| err("malformed HTTP response"))?;
        let status = head
            .lines()
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .unwrap_or("?");
        if status != "200" {
            return Err(err(format!(
                "{} returned HTTP {status}: {}",
                self.cfg.endpoint,
                json.chars().take(300).collect::<String>()
            )));
        }
        parse_reply(json, &self.cfg.endpoint)
    }
}

/// Pull the verdict + confidence out of an OpenAI chat reply.
fn parse_reply(json: &str, endpoint: &str) -> Result<Readiness> {
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| err(format!("parse reply: {e}")))?;
    let choice = v
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| err(format!("{endpoint}: reply has no choices")))?;
    let content = choice
        .pointer("/message/content")
        .and_then(|c| c.as_str())
        .ok_or_else(|| err(format!("{endpoint}: reply has no message content")))?;

    let obj: serde_json::Value = serde_json::from_str(content).map_err(|e| {
        err(format!(
            "{endpoint}: verdict is not JSON ({e}) — strict json_schema was not \
             enforced: {content}"
        ))
    })?;
    let label: ReadyLabel = obj
        .get("label")
        .and_then(|l| l.as_str())
        .unwrap_or_default()
        .parse()?;
    let rationale = obj
        .get("rationale")
        .and_then(|r| r.as_str())
        .unwrap_or("")
        .to_string();

    Ok(Readiness {
        ready: label.is_ready(),
        confidence: confidence_from_logprobs(choice),
        label: label.tag().to_string(),
        rationale,
    })
}

/// `exp(mean(logprob))` over the reply's tokens, clamped to `[0,1]`.
///
/// Honest caveat, same as clyffy's: this averages over ALL tokens, including the
/// structural ones the grammar forced (`{"label":"`). Those are near-certain by
/// construction and inflate the mean. Masking to the value tokens is the
/// refinement. No logprobs (server didn't return them) ⇒ 0.0 — an unknown
/// confidence must read as no confidence, never as certainty.
fn confidence_from_logprobs(choice: &serde_json::Value) -> f32 {
    let Some(toks) = choice
        .pointer("/logprobs/content")
        .and_then(|c| c.as_array())
        .filter(|a| !a.is_empty())
    else {
        return 0.0;
    };
    let sum: f64 = toks
        .iter()
        .filter_map(|t| t.get("logprob").and_then(|l| l.as_f64()))
        .sum();
    let n = toks.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    (sum / n).exp().clamp(0.0, 1.0) as f32
}

async fn discover_model(host: &str, port: u16, timeout: Duration) -> Result<String> {
    let req =
        format!("GET /v1/models HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n");
    let raw = roundtrip(host, port, &req, timeout).await?;
    let text = String::from_utf8_lossy(&raw);
    let json = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b)
        .ok_or_else(|| err("malformed /v1/models response"))?;
    let v: serde_json::Value =
        serde_json::from_str(json).map_err(|e| err(format!("parse /v1/models: {e}")))?;
    v.get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|m| m.get("id"))
        .and_then(|s| s.as_str())
        .or_else(|| {
            v.get("models")
                .and_then(|d| d.as_array())
                .and_then(|a| a.first())
                .and_then(|m| m.get("name").or_else(|| m.get("id")))
                .and_then(|s| s.as_str())
        })
        .map(str::to_string)
        .ok_or_else(|| {
            err(format!(
                "could not discover a model from {host}:{port}/v1/models"
            ))
        })
}

async fn roundtrip(host: &str, port: u16, req: &str, timeout: Duration) -> Result<Vec<u8>> {
    tokio::time::timeout(timeout, async {
        let mut stream = TcpStream::connect((host, port))
            .await
            .map_err(|e| err(format!("connect {host}:{port}: {e}")))?;
        stream
            .write_all(req.as_bytes())
            .await
            .map_err(|e| err(format!("write: {e}")))?;
        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .await
            .map_err(|e| err(format!("read: {e}")))?;
        Ok::<_, RroError>(buf)
    })
    .await
    .map_err(|_| err(format!("{host}:{port} timed out after {timeout:?}")))?
}

fn parse_url(url: &str) -> Result<(String, u16, String)> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| err(format!("endpoint must start with http:// — got `{url}`")))?;
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/v1/chat/completions"),
    };
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| err(format!("bad port in `{url}`")))?,
        ),
        None => (authority.to_string(), 80),
    };
    if host.is_empty() {
        return Err(err(format!("no host in `{url}`")));
    }
    Ok((host, port, path.to_string()))
}

fn err(msg: impl Into<String>) -> RroError {
    RroError::Classify(msg.into())
}

#[async_trait]
impl Classifier for ConstrainedClassifier {
    async fn classify(&self, query: &str, context: &[Candidate]) -> Result<Readiness> {
        self.judge(query, context).await
    }

    fn model_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_generated_from_the_labels_and_cannot_drift() {
        let s = schema();
        for l in ReadyLabel::ALL {
            assert!(
                s.contains(&format!("\"{}\"", l.tag())),
                "label `{}` exists in Rust but not in the grammar — the schema is \
                 no longer generated from ReadyLabel::ALL",
                l.tag()
            );
        }
        assert!(s.contains("\"additionalProperties\":false"));
        assert!(serde_json::from_str::<serde_json::Value>(&s).is_ok());
    }

    #[test]
    fn only_ready_means_ready() {
        assert!(ReadyLabel::Ready.is_ready());
        for l in [
            ReadyLabel::Insufficient,
            ReadyLabel::Ambiguous,
            ReadyLabel::Conflicting,
        ] {
            assert!(!l.is_ready(), "{} must not read as ready", l.tag());
        }
    }

    #[test]
    fn an_out_of_schema_label_is_a_loud_error() {
        // If the server ignored strict json_schema, we must not guess.
        let e = "definitely-ready".parse::<ReadyLabel>().unwrap_err();
        assert!(e.to_string().contains("not in the schema"));
    }

    #[test]
    fn missing_logprobs_read_as_zero_confidence_not_certainty() {
        let choice = serde_json::json!({"message": {"content": "{}"}});
        assert_eq!(confidence_from_logprobs(&choice), 0.0);
    }

    #[test]
    fn confidence_is_exp_mean_logprob() {
        let choice = serde_json::json!({
            "logprobs": {"content": [
                {"logprob": -0.10536}, // ~0.90
                {"logprob": -0.10536},
            ]}
        });
        let c = confidence_from_logprobs(&choice);
        assert!((c - 0.90).abs() < 0.01, "expected ~0.90, got {c}");
    }

    #[test]
    fn a_confident_reply_parses_into_a_verdict() {
        let json = serde_json::json!({
            "choices": [{
                "message": {"content": "{\"label\":\"insufficient\",\"rationale\":\"nothing on topic\"}"},
                "logprobs": {"content": [{"logprob": -0.2}]}
            }]
        })
        .to_string();
        let r = parse_reply(&json, "t").unwrap();
        assert!(!r.ready);
        assert_eq!(r.label, "insufficient");
        assert_eq!(r.rationale, "nothing on topic");
        assert!(r.confidence > 0.7);
    }

    #[test]
    fn non_json_content_fails_loudly() {
        let json = serde_json::json!({
            "choices": [{"message": {"content": "sure, it looks ready to me!"}}]
        })
        .to_string();
        let e = parse_reply(&json, "t").unwrap_err().to_string();
        assert!(
            e.contains("strict json_schema was not enforced"),
            "the error must name the cause, got: {e}"
        );
    }
}
