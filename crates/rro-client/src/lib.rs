//! # rro-client — the typed handle to a Reason Ready node
//!
//! What a host (Clyffy) imports to treat any rro node as local: ask, index,
//! page the changefeed, ping — over the a2a layer-2 protocol, token-aware.
//!
//! ```no_run
//! # async fn run() -> rro_core::Result<()> {
//! use rro_client::Client;
//!
//! let node = Client::new("127.0.0.1:7878").with_token("s3cret");
//! node.index(vec![rro_core::Document::new("estate rollout notes")]).await?;
//! let answer = node.ask("rollout notes").await?;
//! println!("ready={} intent={:?}", answer.readiness.ready, answer.intent);
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use serde::Deserialize;

use rro_core::{Candidate, Document, EstateQuery, RecallResult, Result, RroError};
use rro_net::{tcp, Message};

/// One page of a node's changefeed.
#[derive(Debug, Clone, Deserialize)]
pub struct ChangesPage {
    /// The change records (estate `Change` objects).
    pub changes: Vec<serde_json::Value>,
    /// Cursor for the next page.
    pub next_seq: u64,
}

/// A typed client for one rro node.
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
            from: "rro-client".to_string(),
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
        let mut msg = Message::request(self.from.as_str(), "rro", verb, body);
        if let Some(t) = &self.token {
            msg = msg.with_token(t.clone());
        }
        let reply = tcp::request(self.addr.as_str(), &msg).await?;
        if let Some(err) = reply.body.get("error").and_then(|e| e.as_str()) {
            return Err(RroError::Net(format!("node refused `{verb}`: {err}")));
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
            .ok_or_else(|| RroError::Net("index reply missing total".into()))
    }

    /// Apply a sequence of writes as one atomic transaction, over the wire.
    ///
    /// `ops` is a JSON array whose elements are `{"upsert": [<doc>, …]}` or
    /// `{"remove": "<id>"}`. All of them commit or none do — a failure anywhere
    /// (a bad op, a dimension mismatch) rolls the whole batch back and nothing
    /// durable lands. Returns the number of ops committed.
    pub async fn transaction(&self, ops: serde_json::Value) -> Result<usize> {
        let body = self.call("tx", serde_json::json!({ "ops": ops })).await?;
        if let Some(err) = body.get("error").and_then(|e| e.as_str()) {
            return Err(RroError::Net(format!("tx refused: {err}")));
        }
        body.get("committed")
            .and_then(|t| t.as_u64())
            .map(|t| t as usize)
            .ok_or_else(|| RroError::Net("tx reply missing `committed`".into()))
    }

    /// Run a GraphQL query against the node — over the a2a transport, not HTTP.
    ///
    /// Returns the GraphQL response envelope (`{data}` or `{data, errors}`).
    /// GraphQL is a query language, not a transport, so this rides the same
    /// connection as every other verb.
    pub async fn graphql(&self, query: &str) -> Result<serde_json::Value> {
        self.call("graphql", serde_json::json!({ "query": query }))
            .await
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
    pub async fn watch<F>(&self, since_seq: u64, on_change: F) -> Result<u64>
    where
        F: FnMut(serde_json::Value) -> bool + Send,
    {
        self.stream_verb(
            "watch",
            serde_json::json!({ "since_seq": since_seq }),
            since_seq,
            on_change,
        )
        .await
    }

    /// The RRQL `LIVE` subscription: open a push stream from an RRQL statement.
    /// `LIVE` streams changes from now; `LIVE SINCE n` resumes from seq `n`.
    /// Same frames and cursor semantics as [`Client::watch`].
    pub async fn live<F>(&self, rql: &str, on_change: F) -> Result<u64>
    where
        F: FnMut(serde_json::Value) -> bool + Send,
    {
        self.stream_verb("live", serde_json::json!({ "sql": rql }), 0, on_change)
            .await
    }

    /// Shared push-stream driver for the streaming verbs (`watch`, `live`): one
    /// long-lived connection, invoke `on_change` per frame, return the resume
    /// cursor. `start_cursor` seeds the returned cursor before any frame arrives.
    async fn stream_verb<F>(
        &self,
        verb: &str,
        body: serde_json::Value,
        start_cursor: u64,
        mut on_change: F,
    ) -> Result<u64>
    where
        F: FnMut(serde_json::Value) -> bool + Send,
    {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let stream = tokio::net::TcpStream::connect(self.addr.as_str())
            .await
            .map_err(|e| RroError::Net(format!("connect: {e}")))?;
        let (read_half, mut write_half) = stream.into_split();

        let mut msg = Message::request(self.from.as_str(), "rro", verb, body);
        if let Some(t) = &self.token {
            msg = msg.with_token(t.clone());
        }
        let mut buf = serde_json::to_string(&msg)?;
        buf.push('\n');
        write_half
            .write_all(buf.as_bytes())
            .await
            .map_err(|e| RroError::Net(format!("write: {e}")))?;

        let mut cursor = start_cursor;
        let mut lines = BufReader::new(read_half).lines();
        while let Some(line) = lines
            .next_line()
            .await
            .map_err(|e| RroError::Net(format!("read: {e}")))?
        {
            if line.trim().is_empty() {
                continue;
            }
            let frame: Message = serde_json::from_str(&line)?;
            if let Some(err) = frame.body.get("error").and_then(|e| e.as_str()) {
                return Err(RroError::Net(format!("node refused `{verb}`: {err}")));
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

    /// The node's health snapshot: uptime, estate counters, self-reported
    /// issues.
    pub async fn health(&self) -> Result<serde_json::Value> {
        self.call("health", serde_json::json!({})).await
    }

    /// Full estate introspection: identity, analyzer, dims, payload
    /// indexes, collections, aliases, quotas, feed stats.
    pub async fn info(&self) -> Result<serde_json::Value> {
        self.call("info", serde_json::json!({})).await
    }

    /// Flush the node's estate: memtables + WAL sync — the explicit
    /// durability ack point.
    pub async fn flush(&self) -> Result<()> {
        self.call("flush", serde_json::json!({})).await.map(|_| ())
    }

    /// Force a full compaction pass; returns per-CF live SST bytes.
    pub async fn compact(&self) -> Result<Vec<(String, u64)>> {
        let body = self.call("compact", serde_json::json!({})).await?;
        Ok(serde_json::from_value(
            body.get("cf_bytes").cloned().unwrap_or_default(),
        )?)
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
            .ok_or_else(|| RroError::Net("query reply missing candidates".into()))?;
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
            .ok_or_else(|| RroError::Net("recommend reply missing candidates".into()))?;
        Ok(serde_json::from_value(candidates)?)
    }

    /// Run one RRQL statement.
    ///
    /// `SELECT` is embedded server-side, so a thin client needs no model
    /// weights. `read_only` refuses writes at the node — a caller that pins
    /// itself read-only cannot be tricked into a mutation by a crafted string.
    ///
    /// Returns the node's reply verbatim: the shape depends on the statement
    /// (`{candidates}` for SELECT, `{kind: "defined"|"deleted"|…}` for the
    /// rest), because flattening them into one shape would lose the only
    /// information each carries.
    pub async fn sql(&self, statement: &str, read_only: bool) -> Result<serde_json::Value> {
        self.call(
            "sql",
            serde_json::json!({ "sql": statement, "read_only": read_only }),
        )
        .await
    }

    /// Discover: steer exploration by context pairs, ranking by how much each
    /// candidate agrees with the pairs. `text` is embedded server-side.
    pub async fn discover(
        &self,
        text: &str,
        pairs: Vec<(String, String)>,
        top_k: usize,
    ) -> Result<Vec<Candidate>> {
        let body = self
            .call(
                "discover",
                serde_json::json!({ "text": text, "pairs": pairs, "top_k": top_k }),
            )
            .await?;
        let candidates = body
            .get("candidates")
            .cloned()
            .ok_or_else(|| RroError::Net("discover reply missing candidates".into()))?;
        Ok(serde_json::from_value(candidates)?)
    }

    /// Assert one graph edge: `from -verb-> to`.
    pub async fn relate(&self, from: &str, verb: &str, to: &str) -> Result<()> {
        self.call(
            "relate",
            serde_json::json!({ "from": from, "verb": verb, "to": to }),
        )
        .await?;
        Ok(())
    }

    /// Walk the graph from `start`, breadth-first, nearest hops first.
    ///
    /// `verbs` empty = follow every verb. Returns visited ids in traversal order.
    #[allow(clippy::too_many_arguments)]
    pub async fn traverse(
        &self,
        start: Vec<String>,
        verbs: Vec<String>,
        outbound: bool,
        inbound: bool,
        depth: usize,
        limit: usize,
    ) -> Result<Vec<String>> {
        let body = self
            .call(
                "traverse",
                serde_json::json!({
                    "start": start,
                    "verbs": verbs,
                    "outbound": outbound,
                    "inbound": inbound,
                    "depth": depth,
                    "limit": limit,
                }),
            )
            .await?;
        let ids = body
            .get("ids")
            .cloned()
            .ok_or_else(|| RroError::Net("traverse reply missing ids".into()))?;
        Ok(serde_json::from_value(ids)?)
    }
}
