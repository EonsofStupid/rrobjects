//! The HTTP doorway — a thin front door over the a2a verbs.
//!
//! a2a (NDJSON/TCP) is the real API. But a client that speaks only HTTP — curl,
//! a browser, a webhook — cannot open an a2a connection. This doorway accepts an
//! HTTP/1.1 request, translates it into the *exact same* [`Message`] the TCP
//! surface would receive, and calls the one [`FlowNode::handle`]. So every HTTP
//! response is byte-identical to the a2a reply for the same verb — there is no
//! second implementation of `query`/`ask`/`health` to drift out of sync.
//!
//! Zero HTTP dependencies (no hyper/axum/reqwest): the same hand-rolled,
//! GET-and-POST HTTP/1.1 responder as [`crate::ops`], extended to read a request
//! body and to carry a bearer token into the same capability gate a2a uses.
//!
//! ## Routes
//! - `GET  /healthz` · `/livez` · `/readyz` — liveness/readiness probes (`ok`).
//! - `GET  /health` — the `health` verb (uptime + estate snapshot + issues).
//! - `POST /query` — the `query` verb: body is an `EstateQuery`, reply is
//!   `{ "candidates": [...] }`.
//! - `POST /ask` — the `ask` verb: body is `{ "query": "...", ... }`, reply is
//!   the full pass (candidates + LLM-ready context, TOON-negotiated).
//! - `POST /v/{verb}` — the escape hatch: dispatch *any* request/reply verb the
//!   a2a handler serves, body forwarded verbatim. This is the honest statement
//!   that HTTP here is a doorway, not a curated subset.
//!
//! Streaming verbs (`watch`/`live`) are not served here — they need a push
//! channel (a2a/WS), not request/reply — so a POST to them replies once and
//! closes, exactly as `FlowNode::handle` (not `handle_stream`) would.

use std::sync::Arc;

use rro_core::{Result, RroError};
use rro_net::{Handler, Message};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use crate::handler::FlowNode;

/// A body larger than this is refused (`413`) rather than buffered — a doorway
/// on an untrusted network must not let a `Content-Length` allocate the box.
const MAX_BODY: usize = 16 * 1024 * 1024;

/// Serve the HTTP doorway for `node` until the returned task is dropped.
/// Returns the bound address (bind `127.0.0.1:0` for an OS-assigned port).
pub async fn serve_http(
    addr: &str,
    node: Arc<FlowNode>,
) -> Result<(std::net::SocketAddr, tokio::task::JoinHandle<()>)> {
    let listener = TcpListener::bind(addr)
        .await
        .map_err(|e| RroError::Net(format!("http bind: {e}")))?;
    let local = listener
        .local_addr()
        .map_err(|e| RroError::Net(format!("http local_addr: {e}")))?;

    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let node = node.clone();
            tokio::spawn(async move {
                let _ = answer(stream, node).await;
            });
        }
    });
    Ok((local, task))
}

async fn answer(stream: TcpStream, node: Arc<FlowNode>) -> std::io::Result<()> {
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    // Request line: METHOD PATH VERSION.
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).await? == 0 {
        return Ok(()); // peer hung up before sending anything
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts
        .next()
        .unwrap_or("/")
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();

    // Headers to the blank line. We only need Content-Length (POST body size)
    // and Authorization (the bearer token → the same capability gate as a2a).
    let mut content_length = 0usize;
    let mut token: Option<String> = None;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).await? == 0 {
            break;
        }
        let trimmed = header.trim_end();
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            match name.as_str() {
                "content-length" => content_length = value.parse().unwrap_or(0),
                "authorization" => {
                    token = value
                        .strip_prefix("Bearer ")
                        .or_else(|| value.strip_prefix("bearer "))
                        .map(str::to_string);
                }
                _ => {}
            }
        }
    }

    if content_length > MAX_BODY {
        return respond(
            &mut write_half,
            "413 Payload Too Large",
            "text/plain",
            format!("body exceeds {MAX_BODY} bytes\n"),
        )
        .await;
    }

    // Read exactly Content-Length bytes of body (same BufReader — read_line left
    // the reader positioned at the first body byte).
    let mut body_bytes = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body_bytes).await?;
    }

    // Route → (verb, body JSON). GET probes short-circuit; everything else maps
    // to an a2a verb and is dispatched through the one handler.
    let route = route(&method, &path);
    let (verb, body) = match route {
        Route::Probe => {
            return respond(&mut write_half, "200 OK", "text/plain", "ok\n".to_string()).await;
        }
        Route::NotFound => {
            return respond(
                &mut write_half,
                "404 Not Found",
                "text/plain",
                "not found\n".to_string(),
            )
            .await;
        }
        Route::MethodNotAllowed => {
            return respond(
                &mut write_half,
                "405 Method Not Allowed",
                "text/plain",
                "method not allowed\n".to_string(),
            )
            .await;
        }
        Route::Verb(verb) => {
            // GET verbs (health) carry no body; POST verbs parse JSON.
            let body = if content_length == 0 {
                serde_json::json!({})
            } else {
                match serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                    Ok(v) => v,
                    Err(e) => {
                        return respond(
                            &mut write_half,
                            "400 Bad Request",
                            "application/json",
                            serde_json::json!({ "error": format!("malformed JSON body: {e}") })
                                .to_string(),
                        )
                        .await;
                    }
                }
            };
            (verb, body)
        }
    };

    // Build the identical a2a message the TCP surface would hand the handler,
    // and dispatch it. The reply body IS the HTTP response body — byte-identical
    // to a2a for the same verb by construction.
    let mut msg = Message::request("http", "rro", verb, body);
    if let Some(token) = token {
        msg = msg.with_token(token);
    }
    let reply = match node.handle(msg).await {
        Ok(Some(reply)) => reply.body,
        Ok(None) => serde_json::json!({ "error": "handler produced no reply" }),
        Err(e) => serde_json::json!({ "error": e.to_string() }),
    };
    // Unauthorized is the one handler outcome that deserves a real HTTP status,
    // so a client (and a proxy) can react to it without parsing the body.
    let status = if reply.get("error").and_then(|v| v.as_str()) == Some("unauthorized") {
        "401 Unauthorized"
    } else {
        "200 OK"
    };
    respond(
        &mut write_half,
        status,
        "application/json",
        reply.to_string(),
    )
    .await
}

/// The route table. Kept separate so the HTTP method/path → a2a verb mapping is
/// one readable place.
enum Route {
    Probe,
    Verb(String),
    NotFound,
    MethodNotAllowed,
}

fn route(method: &str, path: &str) -> Route {
    match (method, path) {
        ("GET", "/healthz" | "/livez" | "/readyz") => Route::Probe,
        ("GET", "/health") => Route::Verb("health".into()),
        ("POST", "/query") => Route::Verb("query".into()),
        ("POST", "/ask") => Route::Verb("ask".into()),
        // The escape hatch: POST /v/<verb> dispatches any request/reply verb.
        ("POST", p) if p.starts_with("/v/") => {
            let verb = p.trim_start_matches("/v/");
            if verb.is_empty() || verb.contains('/') {
                Route::NotFound
            } else {
                Route::Verb(verb.to_string())
            }
        }
        // A known path with the wrong method is a 405, an unknown path a 404.
        (_, "/health" | "/query" | "/ask") => Route::MethodNotAllowed,
        (_, p) if p.starts_with("/v/") => Route::MethodNotAllowed,
        _ => Route::NotFound,
    }
}

async fn respond(
    write_half: &mut (impl AsyncWriteExt + Unpin),
    status: &str,
    content_type: &str,
    body: String,
) -> std::io::Result<()> {
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    write_half.write_all(response.as_bytes()).await?;
    write_half.shutdown().await
}
