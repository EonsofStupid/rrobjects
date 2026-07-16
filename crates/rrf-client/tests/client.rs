//! Sprint 8 gates: the typed client and the MCP binding, against a live node.

use std::sync::Arc;

use rrf_client::Client;
use rrf_core::Document;
use rrf_flow::{FlowNode, ReasonReadyFlow};
use rrf_net::tcp;

async fn live_node() -> std::net::SocketAddr {
    let flow = Arc::new(ReasonReadyFlow::default_engine());
    flow.index(rrf_flow::sample_corpus()).await.unwrap();
    let node = Arc::new(FlowNode::new(flow, "rrf"));
    let (addr, task) = tcp::serve("127.0.0.1:0", node).await.unwrap();
    std::mem::forget(task); // keep serving for the test's lifetime
    addr
}

#[tokio::test(flavor = "multi_thread")]
async fn client_ping_index_ask_changes() {
    let addr = live_node().await;
    let client = Client::new(addr.to_string()).with_identity("clyffy");

    assert!(client.ping().await.unwrap());

    let total = client
        .index(vec![
            Document::new("clyffy bake-off corpus entry alpha").with_id("b1"),
            Document::new("clyffy bake-off corpus entry beta").with_id("b2"),
        ])
        .await
        .unwrap();
    assert!(total >= 8, "sample corpus + 2: {total}");

    let answer = client.ask("clyffy bake-off corpus").await.unwrap();
    assert!(!answer.candidates.is_empty());
    assert!(answer
        .candidates
        .iter()
        .any(|c| c.id.as_str() == "b1" || c.id.as_str() == "b2"));

    // Default engine has no estate → changes is refused; the client surfaces
    // it as a typed error instead of silence.
    let err = client.changes(0, 10).await;
    assert!(err.is_err(), "no estate attached ⇒ typed refusal");
}

#[tokio::test(flavor = "multi_thread")]
async fn mcp_binding_end_to_end() {
    use std::io::{BufRead, BufReader, Write};

    let addr = live_node().await;

    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_rrf-mcp"))
        .env("RRF_ADDR", addr.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();

    let mut rpc = |req: serde_json::Value| -> serde_json::Value {
        writeln!(stdin, "{req}").unwrap();
        serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap()
    };

    // initialize
    let init = rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": { "protocolVersion": "2024-11-05", "capabilities": {} }
    }));
    assert_eq!(init["result"]["serverInfo"]["name"], "rrf-mcp");

    // tools/list
    let tools = rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/list"
    }));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"rrf_ask") && names.contains(&"rrf_index"));

    // tools/call → the full pipeline answers through MCP.
    let ask = rpc(serde_json::json!({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": { "name": "rrf_ask", "arguments": { "query": "postgres upgrade" } }
    }));
    assert_eq!(ask["result"]["isError"], serde_json::json!(false));
    let text = ask["result"]["content"][0]["text"].as_str().unwrap();
    let result: serde_json::Value = serde_json::from_str(text).unwrap();
    assert!(
        result["candidates"]
            .as_array()
            .map(|c| !c.is_empty())
            .unwrap_or(false),
        "MCP-delivered answer carries candidates: {text}"
    );

    drop(stdin); // EOF → clean exit
    let _ = child.wait();
}
