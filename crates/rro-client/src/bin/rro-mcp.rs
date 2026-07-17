//! `rro-mcp` — the MCP binding: mount a Reason Ready node in any MCP client.
//!
//! Speaks the Model Context Protocol stdio transport (newline-delimited
//! JSON-RPC 2.0) and bridges to the node at `RRO_ADDR` (token: `RRO_TOKEN`)
//! over the a2a layer-2 protocol. Tools exposed:
//!
//! - `rro_ask`     — full pipeline: RRD gate → hybrid recall → rerank →
//!   readiness; returns candidates + verdict + intent.
//! - `rro_query`   — the typed query plane: filter DSL (must/should/
//!   must_not over eq/any/range/exists), score threshold, lean payload.
//! - `rro_index`   — ingest documents.
//! - `rro_tx`      — a sequence of upserts/removes applied as one atomic transaction.
//! - `rro_graphql` — a GraphQL query over the a2a transport (client picks the shape).
//! - `rro_changes` — page the durable changefeed.
//!
//! MCP client config (e.g. Claude-family desktop/CLI):
//! ```json
//! { "mcpServers": { "rro": { "command": "rro-mcp",
//!     "env": { "RRO_ADDR": "127.0.0.1:7878" } } } }
//! ```

use rro_client::Client;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn tool_list() -> serde_json::Value {
    serde_json::json!([
        {
            "name": "rro_sql",
            "description": "Run one RRQL statement — the Reason Ready Query Language. \
    Reads: SELECT * [FROM collection] [WHERE ...] [LIMIT n]; conditions are \
    `k = v`, `k IN (..)`, `k >= n`, `EXISTS(k)`, `k INSIDE RADIUS(lat,lon,m)`, \
    `k INSIDE BOX(...)`, combined with AND / OR / NOT. Graph: RELATE a -> verb -> b; \
    TRAVERSE a -> verb -> DEPTH n; INFO. Writes: DEFINE INDEX ON f | DEFINE ALIAS a FOR c | \
    REMOVE ALIAS a | REMOVE COLLECTION c | UPDATE id SET k = v (merges) | \
    UPDATE id CONTENT SET k = v (replaces) | DELETE id | DELETE PAYLOAD id [(k,..)]. \
    Set read_only:true to refuse writes. SELECT is embedded server-side.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": { "type": "string", "description": "One RRQL statement." },
                    "read_only": {
                        "type": "boolean",
                        "description": "Refuse any statement that mutates the estate (default false)."
                    }
                },
                "required": ["sql"]
            }
        },
        {
            "name": "rro_ask",
            "description": "Ask the Reason Ready engine: hybrid retrieval + rerank + readiness verdict + intent.",
            "inputSchema": {
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }
        },
        {
            "name": "rro_query",
            "description": "Typed retrieval against the node's estate: hybrid search with a filter DSL (must/should/must_not clauses over eq/any/range/exists), optional score threshold, lean id-only payloads.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "Query text (embedded server-side)." },
                    "top_k": { "type": "integer", "description": "Results wanted (default 10)." },
                    "dsl": {
                        "type": "object",
                        "description": "Filter: {must:[..], should:[..], must_not:[..]}; each condition is {op:'eq',key,value} | {op:'any',key,values} | {op:'range',key,gte?,lte?,gt?,lt?} | {op:'exists',key}."
                    },
                    "score_threshold": { "type": "number" },
                    "with_payload": { "type": "boolean" }
                },
                "required": ["text"]
            }
        },
        {
            "name": "rro_index",
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
            "name": "rro_tx",
            "description": "Apply a sequence of writes as ONE atomic transaction: all commit or none. Body: {\"ops\": [{\"upsert\": [<doc>...]}, {\"remove\": \"<id>\"}]}. Upserts are embedded server-side; a failure anywhere rolls the whole batch back.",
            "inputSchema": {
                "type": "object",
                "properties": { "ops": { "type": "array", "items": { "type": "object" } } },
                "required": ["ops"]
            }
        },
        {
            "name": "rro_graphql",
            "description": "Run a GraphQL query over the node (a2a transport, not HTTP). Schema: Query { health, collections { name count }, document(id) { id text metadata }, search(query, topK, mode) { id score text metadata } }. The client picks the response shape.",
            "inputSchema": {
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }
        },
        {
            "name": "rro_collections",
            "description": "Manage the estate's named collections and aliases: list collections (with counts), drop a collection, create/list/delete aliases.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "drop", "create_alias", "aliases", "delete_alias"] },
                    "name": { "type": "string", "description": "Collection name (drop)." },
                    "alias": { "type": "string" },
                    "collection": { "type": "string", "description": "Alias target (create_alias)." }
                },
                "required": ["action"]
            }
        },
        {
            "name": "rro_payload",
            "description": "Per-point payload ops: set (merge), overwrite, delete_keys, clear — payload indexes stay exactly consistent.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["set", "overwrite", "delete_keys", "clear"] },
                    "id": { "type": "string" },
                    "metadata": { "type": "object" },
                    "keys": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["action", "id"]
            }
        },
        {
            "name": "rro_changes",
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
        "rro_sql" => {
            let stmt = args.get("sql").and_then(|q| q.as_str()).unwrap_or("");
            let read_only = args
                .get("read_only")
                .and_then(|b| b.as_bool())
                .unwrap_or(false);
            client.sql(stmt, read_only).await.map_err(|e| e.to_string())
        }
        "rro_ask" => {
            let query = args.get("query").and_then(|q| q.as_str()).unwrap_or("");
            client
                .ask(query)
                .await
                .map(|r| serde_json::to_value(r).unwrap_or_default())
                .map_err(|e| e.to_string())
        }
        "rro_query" => {
            let mut q = rro_core::EstateQuery::text(
                args.get("text").and_then(|t| t.as_str()).unwrap_or(""),
                args.get("top_k").and_then(|k| k.as_u64()).unwrap_or(10) as usize,
            );
            if let Some(dsl) = args.get("dsl") {
                match serde_json::from_value(dsl.clone()) {
                    Ok(f) => q.dsl = Some(f),
                    Err(e) => {
                        return serde_json::json!({
                            "content": [{ "type": "text", "text": format!("bad dsl: {e}") }],
                            "isError": true
                        })
                    }
                }
            }
            if let Some(t) = args.get("score_threshold").and_then(|v| v.as_f64()) {
                q.score_threshold = Some(t as f32);
            }
            if let Some(p) = args.get("with_payload").and_then(|v| v.as_bool()) {
                q.with_payload = p;
            }
            client
                .query(&q)
                .await
                .map(|c| serde_json::json!({ "candidates": c }))
                .map_err(|e| e.to_string())
        }
        "rro_index" => {
            let docs = args.get("docs").cloned().unwrap_or(serde_json::json!([]));
            match serde_json::from_value::<Vec<rro_core::Document>>(docs) {
                Ok(docs) => client
                    .index(docs)
                    .await
                    .map(|t| serde_json::json!({ "total": t }))
                    .map_err(|e| e.to_string()),
                Err(e) => Err(format!("bad docs: {e}")),
            }
        }
        "rro_tx" => {
            let ops = args.get("ops").cloned().unwrap_or(serde_json::json!([]));
            client
                .transaction(ops)
                .await
                .map(|n| serde_json::json!({ "committed": n }))
                .map_err(|e| e.to_string())
        }
        "rro_graphql" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            client.graphql(query).await.map_err(|e| e.to_string())
        }
        "rro_changes" => {
            let since = args.get("since_seq").and_then(|v| v.as_u64()).unwrap_or(0);
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(256) as usize;
            client
                .changes(since, limit)
                .await
                .map(|p| serde_json::json!({ "changes": p.changes, "next_seq": p.next_seq }))
                .map_err(|e| e.to_string())
        }
        "rro_collections" => {
            let action = args.get("action").and_then(|a| a.as_str()).unwrap_or("");
            let str_of = |k: &str| {
                args.get(k)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            };
            match action {
                "list" => client
                    .collections()
                    .await
                    .map(|c| serde_json::json!({ "collections": c }))
                    .map_err(|e| e.to_string()),
                "drop" => client
                    .drop_collection(&str_of("name"))
                    .await
                    .map(|n| serde_json::json!({ "dropped": n }))
                    .map_err(|e| e.to_string()),
                "create_alias" => client
                    .create_alias(&str_of("alias"), &str_of("collection"))
                    .await
                    .map(|_| serde_json::json!({ "ok": true }))
                    .map_err(|e| e.to_string()),
                "aliases" => client
                    .aliases()
                    .await
                    .map(|a| serde_json::json!({ "aliases": a }))
                    .map_err(|e| e.to_string()),
                "delete_alias" => client
                    .delete_alias(&str_of("alias"))
                    .await
                    .map(|_| serde_json::json!({ "ok": true }))
                    .map_err(|e| e.to_string()),
                other => Err(format!("unknown action: {other}")),
            }
        }
        "rro_payload" => {
            let action = args.get("action").and_then(|a| a.as_str()).unwrap_or("");
            let id = args.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let meta = args
                .get("metadata")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let done = match action {
                "set" => client.set_payload(id, meta).await,
                "overwrite" => client.overwrite_payload(id, meta).await,
                "delete_keys" => {
                    let keys: Vec<String> = args
                        .get("keys")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    client.delete_payload_keys(id, &keys).await
                }
                "clear" => client.clear_payload(id).await,
                other => {
                    return serde_json::json!({
                        "content": [{ "type": "text", "text": format!("unknown action: {other}") }],
                        "isError": true
                    })
                }
            };
            done.map(|_| serde_json::json!({ "ok": true }))
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
    let addr = std::env::var("RRO_ADDR").unwrap_or_else(|_| "127.0.0.1:7878".to_string());
    let mut client = Client::new(addr).with_identity("rro-mcp");
    if let Ok(token) = std::env::var("RRO_TOKEN") {
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
                "serverInfo": { "name": "rro-mcp", "version": env!("CARGO_PKG_VERSION") }
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
