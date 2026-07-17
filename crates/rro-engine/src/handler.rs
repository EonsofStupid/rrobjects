//! Expose the flow to peers over a2a.
//!
//! A [`FlowNode`] wraps a shared flow and answers a2a messages, so a remote (or
//! co-located) agent can `ask` the engine without owning it. Same [`Handler`]
//! contract for local and TCP transports.

use std::sync::Arc;

use async_trait::async_trait;
use rro_core::{Document, Result};
use rro_net::{Handler, Message, NodeId};

use crate::flow::ReasonReadyObject;

/// A network-facing node backed by a [`ReasonReadyObject`].
pub struct FlowNode {
    flow: Arc<ReasonReadyObject>,
    estate: Option<Arc<connxism::Estate>>,
    token: Option<String>,
    me: NodeId,
    started: std::time::Instant,
}

impl FlowNode {
    /// Wrap a flow as an addressable node.
    pub fn new(flow: Arc<ReasonReadyObject>, me: impl Into<NodeId>) -> Self {
        FlowNode {
            flow,
            estate: None,
            token: None,
            me: me.into(),
            started: std::time::Instant::now(),
        }
    }

    /// Attach the estate so subscribers can page the changefeed (`changes`).
    pub fn with_estate(mut self, estate: Arc<connxism::Estate>) -> Self {
        self.estate = Some(estate);
        self
    }

    /// Require a capability token: every message (except `ping`, the
    /// liveness probe) must bear it or is refused with `unauthorized`.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}

#[async_trait]
impl Handler for FlowNode {
    async fn handle(&self, msg: Message) -> Result<Option<Message>> {
        // L3 in its first form: fresh authorization at every action. Ping
        // stays open as the liveness probe.
        if let Some(required) = &self.token {
            if msg.verb != "ping" && msg.token.as_deref() != Some(required.as_str()) {
                rro_core::events::emit(
                    "a2a.unauthorized",
                    serde_json::json!({ "verb": msg.verb, "from": msg.from.as_str() }),
                );
                return Ok(Some(msg.reply(serde_json::json!({
                    "error": "unauthorized"
                }))));
            }
        }
        match msg.verb.as_str() {
            "ping" => Ok(Some(msg.reply(serde_json::json!({
                "pong": true,
                "node": self.me.as_str(),
            })))),

            // `ask` / `recall`: run the flow for `body.query`.
            "ask" | "recall" => {
                let query = msg.body.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let result = self.flow.ask(query).await?;
                Ok(Some(msg.reply(serde_json::to_value(&result)?)))
            }

            // `map`: run the flow and return the connectome graph.
            "map" => {
                let query = msg.body.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let (_result, graph) = self.flow.ask_with_map(query).await?;
                Ok(Some(msg.reply(serde_json::to_value(&graph)?)))
            }

            // `changes`: page the durable changefeed — the poll-based
            // subscription. Body: {"since_seq": 0, "limit": 256}; the reply's
            // last seq + 1 is the next cursor. Requires an attached estate.
            "changes" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let since = msg
                    .body
                    .get("since_seq")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let limit = msg
                    .body
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256)
                    .min(4096) as usize;
                let changes = estate.changes(since, limit)?;
                let next = changes.last().map(|c| c.seq + 1).unwrap_or(since);
                Ok(Some(msg.reply(serde_json::json!({
                    "changes": changes,
                    "next_seq": next,
                }))))
            }

            // `query`: the full typed query plane over the wire. The body IS
            // an `EstateQuery` (filters, threshold, scope, lean payload).
            // Text-only queries are embedded server-side by the flow's
            // embedder, so thin clients never need model weights.
            "query" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let mut q: rro_core::EstateQuery = match serde_json::from_value(msg.body.clone()) {
                    Ok(q) => q,
                    Err(e) => {
                        return Ok(Some(msg.reply(serde_json::json!({
                            "error": format!("malformed query: {e}")
                        }))));
                    }
                };
                if q.vector.is_none() {
                    if let Some(text) = q.text.clone() {
                        q.vector = Some(self.flow.embed_query(&text).await?);
                    }
                }
                match estate.recall().query(q).await {
                    Ok(candidates) => Ok(Some(
                        msg.reply(serde_json::json!({ "candidates": candidates })),
                    )),
                    // Refusals (quotas, depth caps) reply cleanly instead
                    // of dropping the connection.
                    Err(e) => Ok(Some(
                        msg.reply(serde_json::json!({ "error": e.to_string() })),
                    )),
                }
            }

            // `recommend`: steer by example ids, over the wire.
            // Body: {"positive": ["id", ...], "negative": [...], "top_k": 10}
            "recommend" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let ids = |key: &str| -> Vec<String> {
                    msg.body
                        .get(key)
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default()
                };
                let positive = ids("positive");
                let negative = ids("negative");
                let top_k = msg
                    .body
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10)
                    .min(1024) as usize;
                match estate.recall().recommend(&positive, &negative, top_k).await {
                    Ok(candidates) => Ok(Some(
                        msg.reply(serde_json::json!({ "candidates": candidates })),
                    )),
                    Err(e) => Ok(Some(
                        msg.reply(serde_json::json!({ "error": e.to_string() })),
                    )),
                }
            }

            // `sql`: one RRQL statement. Body: {"sql": "...", "read_only": bool}
            //
            // This is the verb that makes RRO reachable by anything that can
            // send text — an MCP tool, a CLI, a REST body — instead of only by
            // something that can link the crate and hand-build an EstateQuery.
            "sql" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let Some(src) = msg.body.get("sql").and_then(|v| v.as_str()) else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "sql needs a `sql` string"
                    }))));
                };

                let stmt = match rro_ql::parse(src) {
                    Ok(s) => s,
                    // The caret renders the offending line with a marker under
                    // it. A remote caller that gets "syntax error" has to guess;
                    // one that gets the span can fix it.
                    Err(e) => {
                        return Ok(Some(msg.reply(serde_json::json!({
                            "error": e.to_string(),
                            "detail": e.caret(src),
                        }))))
                    }
                };

                // A peer may pin itself read-only. Refusing a write here rather
                // than at the estate means an exposed node can be safely shared
                // without trusting the caller's intent.
                let read_only = msg
                    .body
                    .get("read_only")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if read_only && stmt.is_write() {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": format!(
                            "{} is a write and this request set read_only",
                            stmt.keyword()
                        )
                    }))));
                }

                // SELECT needs the flow (it must embed the query text); every
                // other statement is an estate op.
                if let rro_ql::Statement::Select(_) = stmt {
                    let q = match rro_ql::parse_query(src) {
                        Ok(q) => q,
                        Err(e) => {
                            return Ok(Some(
                                msg.reply(serde_json::json!({ "error": e.to_string() })),
                            ))
                        }
                    };
                    let text = q.text.clone().unwrap_or_default();
                    let vector = match self.flow.embed_query(&text).await {
                        Ok(v) => v,
                        Err(e) => {
                            return Ok(Some(
                                msg.reply(serde_json::json!({ "error": e.to_string() })),
                            ))
                        }
                    };
                    let mut q = q;
                    q.vector = Some(vector);
                    return match estate.recall().query(q).await {
                        Ok(candidates) => Ok(Some(msg.reply(
                            serde_json::json!({ "kind": "query", "candidates": candidates }),
                        ))),
                        Err(e) => Ok(Some(
                            msg.reply(serde_json::json!({ "error": e.to_string() })),
                        )),
                    };
                }

                match crate::sql::apply(estate, stmt).await {
                    Ok(outcome) => Ok(Some(msg.reply(serde_json::json!(outcome)))),
                    Err(e) => Ok(Some(
                        msg.reply(serde_json::json!({ "error": e.to_string() })),
                    )),
                }
            }

            // `discover`: context-pair steered exploration.
            // Body: {"text": "...", "pairs": [["a","b"], ...], "top_k": 10}
            //
            // discover/relate/traverse were reachable ONLY in-process while
            // PARITY.md implied wire parity. A capability a remote node cannot
            // call is not a capability of the engine, it is a capability of the
            // library.
            "discover" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let Some(text) = msg.body.get("text").and_then(|v| v.as_str()) else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "discover needs `text`"
                    }))));
                };
                let pairs: Vec<(String, String)> = msg
                    .body
                    .get("pairs")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|p| {
                                let p = p.as_array()?;
                                Some((
                                    p.first()?.as_str()?.to_string(),
                                    p.get(1)?.as_str()?.to_string(),
                                ))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let top_k = msg
                    .body
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10)
                    .min(1024) as usize;
                let q = match self.flow.embed_query(text).await {
                    Ok(q) => q,
                    Err(e) => {
                        return Ok(Some(
                            msg.reply(serde_json::json!({ "error": e.to_string() })),
                        ))
                    }
                };
                match estate.recall().discover(&q, &pairs, top_k).await {
                    Ok(candidates) => Ok(Some(
                        msg.reply(serde_json::json!({ "candidates": candidates })),
                    )),
                    Err(e) => Ok(Some(
                        msg.reply(serde_json::json!({ "error": e.to_string() })),
                    )),
                }
            }

            // `relate`: assert one graph edge. Body: {"from","verb","to"}
            "relate" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let s = |k: &str| msg.body.get(k).and_then(|v| v.as_str());
                let (Some(from), Some(verb), Some(to)) = (s("from"), s("verb"), s("to")) else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "relate needs `from`, `verb`, `to`"
                    }))));
                };
                match estate.relate(from, verb, to) {
                    Ok(()) => Ok(Some(msg.reply(serde_json::json!({ "related": true })))),
                    Err(e) => Ok(Some(
                        msg.reply(serde_json::json!({ "error": e.to_string() })),
                    )),
                }
            }

            // `traverse`: walk the graph from a start set.
            // Body: {"start": ["id"], "verbs": [..], "outbound": true,
            //        "inbound": false, "depth": 2, "limit": 100}
            "traverse" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let start: Vec<String> = msg
                    .body
                    .get("start")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                if start.is_empty() {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "traverse needs a non-empty `start`"
                    }))));
                }
                let default = connxism::TraversalSpec::default();
                let spec = connxism::TraversalSpec {
                    verbs: msg
                        .body
                        .get("verbs")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default(),
                    outbound: msg
                        .body
                        .get("outbound")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(default.outbound),
                    inbound: msg
                        .body
                        .get("inbound")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(default.inbound),
                    depth: msg
                        .body
                        .get("depth")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(default.depth as u64)
                        .min(64) as usize,
                    limit: msg
                        .body
                        .get("limit")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(default.limit as u64)
                        .min(10_000) as usize,
                };
                let refs: Vec<&str> = start.iter().map(String::as_str).collect();
                match estate.traverse(&refs, &spec) {
                    Ok(ids) => Ok(Some(msg.reply(serde_json::json!({ "ids": ids })))),
                    Err(e) => Ok(Some(
                        msg.reply(serde_json::json!({ "error": e.to_string() })),
                    )),
                }
            }

            // `index`: ingest a batch of documents over a2a.
            // Body: {"docs": [{"id": "...", "text": "..."}, ...]}
            "index" => {
                let docs: Vec<Document> = msg
                    .body
                    .get("docs")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()?
                    .unwrap_or_default();
                let total = self.flow.index(docs).await?;
                Ok(Some(msg.reply(serde_json::json!({ "total": total }))))
            }

            // `health`: uptime + a live estate snapshot + self-reported
            // issues. Body: {"backlog_threshold": 10000} (optional).
            "health" => {
                let uptime = self.started.elapsed().as_secs();
                let mut body = serde_json::json!({
                    "node": self.me.as_str(),
                    "uptime_secs": uptime,
                });
                if let Some(estate) = &self.estate {
                    let threshold = msg
                        .body
                        .get("backlog_threshold")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(10_000) as usize;
                    body["estate"] = serde_json::to_value(estate.health()?)?;
                    body["issues"] = serde_json::to_value(estate.issues(threshold)?)?;
                }
                Ok(Some(msg.reply(body)))
            }

            // Estate admin + analytics verbs, sprint 12–18 surface. Every
            // arm needs the estate; the macro-ish helper keeps them terse.
            "info"
            | "flush"
            | "compact"
            | "matrix"
            | "sample"
            | "collections"
            | "drop_collection"
            | "create_alias"
            | "aliases"
            | "delete_alias"
            | "set_payload"
            | "overwrite_payload"
            | "delete_payload_keys"
            | "clear_payload" => {
                let Some(estate) = &self.estate else {
                    return Ok(Some(msg.reply(serde_json::json!({
                        "error": "no estate attached to this node"
                    }))));
                };
                let b = &msg.body;
                let str_of = |k: &str| b.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
                let reply = match msg.verb.as_str() {
                    "matrix" => {
                        let ids: Vec<String> = b
                            .get("ids")
                            .and_then(|v| v.as_array())
                            .map(|a| {
                                a.iter()
                                    .filter_map(|x| x.as_str().map(str::to_string))
                                    .collect()
                            })
                            .unwrap_or_default();
                        let pairs = estate.recall().similarity_matrix(&ids).await?;
                        serde_json::json!({ "pairs": pairs })
                    }
                    "sample" => {
                        let n = b.get("n").and_then(|v| v.as_u64()).unwrap_or(10).min(4096);
                        let seed = b.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
                        let docs = estate.sample(n as usize, seed)?;
                        serde_json::json!({ "docs": docs })
                    }
                    "info" => {
                        serde_json::json!({
                            "estate": estate.info(),
                            "health": estate.health()?,
                            "payload_indexes": estate.payload_indexes()?,
                            "collections": estate.collections()?,
                            "aliases": estate.aliases()?,
                            "feed": estate.feed_stats()?,
                        })
                    }
                    "flush" => {
                        estate.flush()?;
                        serde_json::json!({ "ok": true })
                    }
                    "compact" => {
                        estate.compact()?;
                        serde_json::json!({ "ok": true, "cf_bytes": estate.cf_sizes()? })
                    }
                    "collections" => {
                        serde_json::json!({ "collections": estate.collections()? })
                    }
                    "drop_collection" => {
                        let dropped = estate.drop_collection(&str_of("name"))?;
                        serde_json::json!({ "dropped": dropped })
                    }
                    "create_alias" => {
                        estate.create_alias(&str_of("alias"), &str_of("collection"))?;
                        serde_json::json!({ "ok": true })
                    }
                    "aliases" => serde_json::json!({ "aliases": estate.aliases()? }),
                    "delete_alias" => {
                        estate.delete_alias(&str_of("alias"))?;
                        serde_json::json!({ "ok": true })
                    }
                    op @ ("set_payload"
                    | "overwrite_payload"
                    | "delete_payload_keys"
                    | "clear_payload") => {
                        let id = str_of("id");
                        let recall = estate.recall();
                        let done = match op {
                            "set_payload" | "overwrite_payload" => {
                                let meta: rro_core::Metadata = b
                                    .get("metadata")
                                    .cloned()
                                    .map(serde_json::from_value)
                                    .transpose()?
                                    .unwrap_or_default();
                                if op == "set_payload" {
                                    recall.set_payload(&id, meta).await
                                } else {
                                    recall.overwrite_payload(&id, meta).await
                                }
                            }
                            "delete_payload_keys" => {
                                let keys: Vec<String> = b
                                    .get("keys")
                                    .and_then(|v| v.as_array())
                                    .map(|a| {
                                        a.iter()
                                            .filter_map(|x| x.as_str().map(str::to_string))
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                recall.delete_payload_keys(&id, keys).await
                            }
                            _ => recall.clear_payload(&id).await,
                        };
                        match done {
                            Ok(()) => serde_json::json!({ "ok": true }),
                            Err(e) => serde_json::json!({ "error": e.to_string() }),
                        }
                    }
                    _ => unreachable!("outer match guards the verb list"),
                };
                Ok(Some(msg.reply(reply)))
            }

            // Unknown verbs REPLY with an error instead of silence — a
            // request/reply client waiting on a dropped verb would hang
            // forever (found the hard way when `flush` briefly wasn't
            // routed here).
            other => Ok(Some(msg.reply(serde_json::json!({
                "error": format!("unknown verb: {other}")
            })))),
        }
    }

    /// `watch`: the push-stream subscription. Body: {"since_seq": 0}. The
    /// node drains the durable changefeed from the cursor into frames
    /// (`{"change": .., "next_seq": ..}`), then awaits the estate's feed
    /// signal — event-driven, no polling interval anywhere. Resume is the
    /// same seq cursor the poll-based `changes` verb uses. The stream ends
    /// when the peer hangs up (failed send tears the task down).
    async fn handle_stream(
        &self,
        msg: Message,
        tx: tokio::sync::mpsc::Sender<Message>,
    ) -> Result<bool> {
        if msg.verb != "watch" {
            return Ok(false);
        }
        if let Some(required) = &self.token {
            if msg.token.as_deref() != Some(required.as_str()) {
                rro_core::events::emit(
                    "a2a.unauthorized",
                    serde_json::json!({ "verb": "watch", "from": msg.from.as_str() }),
                );
                let _ = tx
                    .send(msg.reply(serde_json::json!({ "error": "unauthorized" })))
                    .await;
                return Ok(true);
            }
        }
        let Some(estate) = self.estate.clone() else {
            let _ = tx
                .send(msg.reply(serde_json::json!({ "error": "no estate attached to this node" })))
                .await;
            return Ok(true);
        };

        let mut since = msg
            .body
            .get("since_seq")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let signal = estate.feed_signal();

        tokio::spawn(async move {
            loop {
                // Arm the signal BEFORE draining: a write landing between
                // the drain and the await still wakes us (no lost updates).
                let notified = signal.notified();
                let page = {
                    let estate = estate.clone();
                    match tokio::task::spawn_blocking(move || estate.changes(since, 256)).await {
                        Ok(Ok(page)) => page,
                        _ => break,
                    }
                };
                if page.is_empty() {
                    notified.await;
                    continue;
                }
                for change in page {
                    since = change.seq + 1;
                    let frame = msg.reply(serde_json::json!({
                        "change": change,
                        "next_seq": since,
                    }));
                    if tx.send(frame).await.is_err() {
                        return; // peer hung up
                    }
                }
            }
        });
        Ok(true)
    }
}
