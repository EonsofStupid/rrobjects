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

use rrf_core::{Candidate, Document, EstateQuery, RecallResult, Result, RrfError};
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

    /// Push-stream subscription: hold one long-lived connection and invoke
    /// `on_change` for every change frame the node pushes (event-driven on
    /// the node side — no polling anywhere). Return `false` from the
    /// callback to stop watching; the cursor to resume from is returned.
    /// The same `since_seq` cursor works across `watch` and [`Client::changes`].
    pub async fn watch<F>(&self, since_seq: u64, mut on_change: F) -> Result<u64>
    where
        F: FnMut(serde_json::Value) -> bool + Send,
    {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let stream = tokio::net::TcpStream::connect(self.addr.as_str())
            .await
            .map_err(|e| RrfError::Net(format!("connect: {e}")))?;
        let (read_half, mut write_half) = stream.into_split();

        let mut msg = Message::request(
            self.from.as_str(),
            "rrf",
            "watch",
            serde_json::json!({ "since_seq": since_seq }),
        );
        if let Some(t) = &self.token {
            msg = msg.with_token(t.clone());
        }
        let mut buf = serde_json::to_string(&msg)?;
        buf.push('\n');
        write_half
            .write_all(buf.as_bytes())
            .await
            .map_err(|e| RrfError::Net(format!("write: {e}")))?;

        let mut cursor = since_seq;
        let mut lines = BufReader::new(read_half).lines();
        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| RrfError::Net(format!("read: {e}")))?
        {
            if line.trim().is_empty() {
                continue;
            }
            let frame: Message = serde_json::from_str(&line)?;
            if let Some(err) = frame.body.get("error").and_then(|e| e.as_str()) {
                return Err(RrfError::Net(format!("node refused `watch`: {err}")));
            }
            if let Some(next) = frame.body.get("next_seq").and_then(|v| v.as_u64()) {
                cursor = next;
            }
            let change = frame
                .body
                .get("change")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            if !on_change(change) {
                break; // dropping the connection cancels the stream server-side
            }
        }
        Ok(cursor)
    }

    /// Pairwise cosine similarity among stored vectors (upper triangle).
    pub async fn similarity_matrix(&self, ids: &[String]) -> Result<Vec<(String, String, f32)>> {
        let body = self
            .call("matrix", serde_json::json!({ "ids": ids }))
            .await?;
        Ok(serde_json::from_value(
            body.get("pairs").cloned().unwrap_or_default(),
        )?)
    }

    /// Deterministic random sample of up to `n` stored documents.
    pub async fn sample(&self, n: usize, seed: u64) -> Result<Vec<serde_json::Value>> {
        let body = self
            .call("sample", serde_json::json!({ "n": n, "seed": seed }))
            .await?;
        Ok(serde_json::from_value(
            body.get("docs").cloned().unwrap_or_default(),
        )?)
    }

    /// The node's named collections with exact member counts.
    pub async fn collections(&self) -> Result<Vec<(String, u64)>> {
        let body = self.call("collections", serde_json::json!({})).await?;
        Ok(serde_json::from_value(
            body.get("collections").cloned().unwrap_or_default(),
        )?)
    }

    /// Drop a collection; returns how many members were removed.
    pub async fn drop_collection(&self, name: &str) -> Result<u64> {
        let body = self
            .call("drop_collection", serde_json::json!({ "name": name }))
            .await?;
        Ok(body.get("dropped").and_then(|v| v.as_u64()).unwrap_or(0))
    }

    /// Create (or atomically repoint) a collection alias.
    pub async fn create_alias(&self, alias: &str, collection: &str) -> Result<()> {
        self.call(
            "create_alias",
            serde_json::json!({ "alias": alias, "collection": collection }),
        )
        .await
        .map(|_| ())
    }

    /// The node's alias map.
    pub async fn aliases(&self) -> Result<std::collections::BTreeMap<String, String>> {
        let body = self.call("aliases", serde_json::json!({})).await?;
        Ok(serde_json::from_value(
            body.get("aliases").cloned().unwrap_or_default(),
        )?)
    }

    /// Delete an alias.
    pub async fn delete_alias(&self, alias: &str) -> Result<()> {
        self.call("delete_alias", serde_json::json!({ "alias": alias }))
            .await
            .map(|_| ())
    }

    /// Merge keys into a document's payload.
    pub async fn set_payload(&self, id: &str, metadata: serde_json::Value) -> Result<()> {
        self.call(
            "set_payload",
            serde_json::json!({ "id": id, "metadata": metadata }),
        )
        .await
        .map(|_| ())
    }

    /// Replace a document's payload entirely.
    pub async fn overwrite_payload(&self, id: &str, metadata: serde_json::Value) -> Result<()> {
        self.call(
            "overwrite_payload",
            serde_json::json!({ "id": id, "metadata": metadata }),
        )
        .await
        .map(|_| ())
    }

    /// Remove keys from a document's payload.
    pub async fn delete_payload_keys(&self, id: &str, keys: &[String]) -> Result<()> {
        self.call(
            "delete_payload_keys",
            serde_json::json!({ "id": id, "keys": keys }),
        )
        .await
        .map(|_| ())
    }

    /// Clear a document's payload.
    pub async fn clear_payload(&self, id: &str) -> Result<()> {
        self.call("clear_payload", serde_json::json!({ "id": id }))
            .await
            .map(|_| ())
    }

    /// The connectome map for a query (JSON graph the UI renders).
    pub async fn map(&self, query: &str) -> Result<serde_json::Value> {
        self.call("map", serde_json::json!({ "query": query }))
            .await
    }

    /// Run a typed query against the node's estate: filters (DSL + equality),
    /// score threshold, scope, lean payload — the full query plane. Text-only
    /// queries are embedded by the node, so this client stays weightless.
    pub async fn query(&self, q: &EstateQuery) -> Result<Vec<Candidate>> {
        let body = self.call("query", serde_json::to_value(q)?).await?;
        let candidates = body
            .get("candidates")
            .cloned()
            .ok_or_else(|| RrfError::Net("query reply missing candidates".into()))?;
        Ok(serde_json::from_value(candidates)?)
    }

    /// Recommend by example ids: steer toward `positive`, away from
    /// `negative`; the examples never appear in the results.
    pub async fn recommend(
        &self,
        positive: Vec<String>,
        negative: Vec<String>,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        let body = self
            .call(
                "recommend",
                serde_json::json!({
                    "positive": positive,
                    "negative": negative,
                    "top_k": top_k,
                }),
            )
            .await?;
        let candidates = body
            .get("candidates")
            .cloned()
            .ok_or_else(|| RrfError::Net("recommend reply missing candidates".into()))?;
        Ok(serde_json::from_value(candidates)?)
    }
}
