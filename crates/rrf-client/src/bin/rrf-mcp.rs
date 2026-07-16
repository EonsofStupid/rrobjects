//! `rrf-mcp` — the MCP binding: mount a Reason Ready node in any MCP client.
//!
//! Speaks the Model Context Protocol stdio transport (newline-delimited
//! JSON-RPC 2.0) and bridges to the node at `RRF_ADDR` (token: `RRF_TOKEN`)
//! over the a2a layer-2 protocol. Tools exposed:
//!
//! - `rrf_ask`     — full pipeline: RRD gate → hybrid recall → rerank →
//!   readiness; returns candidates + verdict + intent.
//! - `rrf_index`   — ingest documents.
//! - `rrf_changes` — page the durable changefeed.
//!
//! MCP client config (e.g. Claude-family desktop/CLI):
//! ```json
//! { "mcpServers": { "rrf": { "command": "rrf-mcp",
//!     "env": { "RRF_ADDR": "127.0.0.1:7878" } } } }
//! ```

use rrf_client::Client;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn tool_list() -> serde_json::Value {
    serde_json::json!([
        {
            "name": "rrf_ask",
            "description": "Ask the Reason Ready engine: hybrid retrieval + rerank + readiness verdict + intent.",
            "inputSchema": {
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }
        },
        {
            "name": "rrf_index",
            "description": "Ingest documents into the engine's estate.",
            "inputSchema": {
                "type": "object",
                "properties": { "docs": { "type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "text": { "type": "string" },
                        "metadata": { "type": "object" }
                    },
                    "required": ["id", "text"]
                } } },
                "required": ["docs"]
            }
        },
        {
            "name": "rrf_changes",
            "description": "Page the estate's durable changefeed (resume with next_seq).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "since_seq": { "type": "integer" },
                    "limit": { "type": "integer" }
                }
            }
        }
    ])
}

async fn call_tool(client: &Client, name: &str, args: &serde_json::Value) -> serde_json::Value {
    let outcome: Result<serde_json::Value, String> = match name {
        "rrf_ask" => {
            let query = args.get("query").and_then(|q| q.as_str()).unwrap_or("");
            client
                .ask(query)
                .await
                .map(|r| serde_json::to_value(r).unwrap_or_default())
                .map_err(|e| e.to_string())
        }
        "rrf_index" => {
            let docs = args.get("docs").cloned().unwrap_or(serde_json::json!([]));
            match serde_json::from_value::<Vec<rrf_core::Document>>(docs) {
                Ok(docs) => client
                    .index(docs)
                    .await
                    .map(|t| serde_json::json!({ "total": t }))
                    .map_err(|e| e.to_string()),
                Err(e) => Err(format!("bad docs: {e}")),
            }
        }
        "rrf_changes" => {
            let since = args.get("since_seq").and_then(|v| v.as_u64()).unwrap_or(0);
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
            client
                .changes(since, limit)
                .await
                .map(|p| serde_json::json!({ "changes": p.changes, "next_seq": p.next_seq }))
                .map_err(|e| e.to_string())
        }
        other => Err(format!("unknown tool: {other}")),
    };

    match outcome {
        Ok(value) => serde_json::json!({
            "content": [{ "type": "text", "text": value.to_string() }],
            "isError": false
        }),
        Err(err) => serde_json::json!({
            "content": [{ "type": "text", "text": err }],
            "isError": true
        }),
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let addr = std::env::var("RRF_ADDR").unwrap_or_else(|_| "127.0.0.1:7878".to_string());
    let mut client = Client::new(addr).with_identity("rrf-mcp");
    if let Ok(token) = std::env::var("RRF_TOKEN") {
        client = client.with_token(token);
    }

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(req) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

        // Notifications (no id) get no response.
        if id.is_none() {
            continue;
        }

        let result: serde_json::Value = match method {
            "initialize" => serde_json::json!({
                "protocolVersion": req["params"]["protocolVersion"].as_str().unwrap_or("2024-11-05"),
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "rrf-mcp", "version": env!("CARGO_PKG_VERSION") }
            }),
            "ping" => serde_json::json!({}),
            "tools/list" => serde_json::json!({ "tools": tool_list() }),
            "tools/call" => {
                let name = req["params"]["name"].as_str().unwrap_or("");
                let args = req["params"]
                    .get("arguments")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));
                call_tool(&client, name, &args).await
            }
            other => {
                let err = serde_json::json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32601, "message": format!("method not found: {other}") }
                });
                stdout.write_all(format!("{err}\n").as_bytes()).await?;
                stdout.flush().await?;
                continue;
            }
        };

        let resp = serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result });
        stdout.write_all(format!("{resp}\n").as_bytes()).await?;
        stdout.flush().await?;
    }
    Ok(())
}
