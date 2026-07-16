//! # rrf-client — the typed handle to a Reason Ready node
//!
//! What a host (Clyffy) imports to treat any rrf node as local: ask, index,
//! page the changefeed, ping — over the a2a layer-2 protocol, token-aware.
//!
//! ```no_run
//! # async fn run() -> rrf_core::Result<()> {
//! use rrf_client::Client;
//!
//! let node = Client::new("127.0.0.1:7878").with_token("s3cret");
//! node.index(vec![rrf_core::Document::new("estate rollout notes")]).await?;
//! let answer = node.ask("rollout notes").await?;
//! println!("ready={} intent={:?}", answer.readiness.ready, answer.intent);
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::Deserialize;

use rrf_core::{Document, RecallResult, Result, RrfError};
use rrf_net::{tcp, Message};

/// One page of a node's changefeed.
#[derive(Debug, Clone, Deserialize)]
pub struct ChangesPage {
    /// The change records (estate `Change` objects).
    pub changes: Vec<serde_json::Value>,
    /// Cursor for the next page.
    pub next_seq: u64,
}

/// A typed client for one rrf node.
#[derive(Debug, Clone)]
pub struct Client {
    addr: String,
    token: Option<String>,
    from: String,
}

impl Client {
    /// A client for the node at `addr` (e.g. `127.0.0.1:7878`).
    pub fn new(addr: impl Into<String>) -> Self {
        Client {
            addr: addr.into(),
            token: None,
            from: "rrf-client".to_string(),
        }
    }

    /// Bear a capability token on every message.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Identify this client on the wire.
    pub fn with_identity(mut self, from: impl Into<String>) -> Self {
        self.from = from.into();
        self
    }

    async fn call(&self, verb: &str, body: serde_json::Value) -> Result<serde_json::Value> {
        let mut msg = Message::request(self.from.as_str(), "rrf", verb, body);
        if let Some(t) = &self.token {
            msg = msg.with_token(t.clone());
        }
        let reply = tcp::request(self.addr.as_str(), &msg).await?;
        if let Some(err) = reply.body.get("error").and_then(|e| e.as_str()) {
            return Err(RrfError::Net(format!("node refused `{verb}`: {err}")));
        }
        Ok(reply.body)
    }

    /// Liveness probe.
    pub async fn ping(&self) -> Result<bool> {
        let body = self.call("ping", serde_json::json!({})).await?;
        Ok(body.get("pong").and_then(|p| p.as_bool()).unwrap_or(false))
    }

    /// Run the node's full pipeline for `query` (RRD gate → embed → hybrid
    /// recall → rerank → readiness); returns candidates, verdict, intent.
    pub async fn ask(&self, query: &str) -> Result<RecallResult> {
        let body = self
            .call("ask", serde_json::json!({ "query": query }))
            .await?;
        Ok(serde_json::from_value(body)?)
    }

    /// Ingest a batch of documents; returns the node's total document count.
    pub async fn index(&self, docs: Vec<Document>) -> Result<usize> {
        let body = self
            .call("index", serde_json::json!({ "docs": docs }))
            .await?;
        body.get("total")
            .and_then(|t| t.as_u64())
            .map(|t| t as usize)
            .ok_or_else(|| RrfError::Net("index reply missing total".into()))
    }

    /// Page the node's durable changefeed from `since_seq`.
    pub async fn changes(&self, since_seq: u64, limit: usize) -> Result<ChangesPage> {
        let body = self
            .call(
                "changes",
                serde_json::json!({ "since_seq": since_seq, "limit": limit }),
            )
            .await?;
        Ok(serde_json::from_value(body)?)
    }

    /// The connectome map for a query (JSON graph the UI renders).
    pub async fn map(&self, query: &str) -> Result<serde_json::Value> {
        self.call("map", serde_json::json!({ "query": query }))
            .await
    }
}
