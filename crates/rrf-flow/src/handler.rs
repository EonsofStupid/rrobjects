//! Expose the flow to peers over a2a.
//!
//! A [`FlowNode`] wraps a shared flow and answers a2a messages, so a remote (or
//! co-located) agent can `ask` the engine without owning it. Same [`Handler`]
//! contract for local and TCP transports.

use std::sync::Arc;

use async_trait::async_trait;
use rrf_core::{Document, Result};
use rrf_net::{Handler, Message, NodeId};

use crate::flow::ReasonReadyFlow;

/// A network-facing node backed by a [`ReasonReadyFlow`].
pub struct FlowNode {
    flow: Arc<ReasonReadyFlow>,
    estate: Option<Arc<connxism::Estate>>,
    me: NodeId,
}

impl FlowNode {
    /// Wrap a flow as an addressable node.
    pub fn new(flow: Arc<ReasonReadyFlow>, me: impl Into<NodeId>) -> Self {
        FlowNode {
            flow,
            estate: None,
            me: me.into(),
        }
    }

    /// Attach the estate so subscribers can page the changefeed (`changes`).
    pub fn with_estate(mut self, estate: Arc<connxism::Estate>) -> Self {
        self.estate = Some(estate);
        self
    }
}

#[async_trait]
impl Handler for FlowNode {
    async fn handle(&self, msg: Message) -> Result<Option<Message>> {
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

            _ => Ok(None),
        }
    }
}
