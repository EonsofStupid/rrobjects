//! An estate is the operator's. Nothing goes in it that they did not put in it.
//!
//! The daemon seeds a small demo corpus (banana bread, Postgres upgrades, a2a
//! protocols) so that an in-memory node has something to answer with. That call
//! sat **outside** the `RRO_ESTATE` match, so it ran unconditionally: every
//! start of a *persistent* node wrote six demo documents into the operator's
//! durable memory, and did it again on every restart.
//!
//! For a product whose whole promise is "your AI remembers", quietly writing
//! demo data into that memory is close to the worst available default. It was
//! not theoretical — a live recall against a real estate returned `d5` ("agent-
//! to-agent protocols let independent agents exchange requests…") above the
//! document that actually answered the question.
//!
//! This is the gate. It spawns the real binary, because the bug lived in the
//! wiring rather than in any library function — a unit test of the flow could
//! not have caught it.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Ask the OS for a free port, then let it go — the daemon takes it next.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// One a2a request/response over the NDJSON wire.
fn a2a(addr: &str, msg: serde_json::Value) -> serde_json::Value {
    use std::io::Write;
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    writeln!(s, "{msg}").unwrap();
    let mut line = String::new();
    BufReader::new(&s).read_line(&mut line).unwrap();
    serde_json::from_str(&line).unwrap()
}

/// Wait for the daemon to answer, so the test isn't a race.
///
/// 90s, not 30s. This spawns the **debug** binary (`CARGO_BIN_EXE_rro`) — an
/// unoptimised RocksDB open plus HNSW rebuild — and `cargo test --workspace`
/// runs several suites at once. On a box already busy (this one was embedding
/// 50k documents at the time) 30s expired and the test failed with "daemon never
/// came up", which reads exactly like a real bug and is not one.
///
/// Recorded because the first diagnosis of that failure was a TOCTOU port race,
/// which was a guess: `free_port()` releases the port before the daemon binds it,
/// so the theory was plausible. It was also wrong — the test passes 8/8 in
/// isolation *and* 283/283 in a workspace run once the box is quiet. Fix the
/// cause you measured, not the one you thought of first.
fn wait_ready(addr: &str) {
    let deadline = Instant::now() + Duration::from_secs(90);
    while Instant::now() < deadline {
        if std::net::TcpStream::connect(addr).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("daemon never came up on {addr}");
}

/// THE gate: a node started with a durable estate must contain **nothing** it
/// was not given.
#[test]
fn a_persistent_estate_is_never_seeded() {
    let dir = tempfile::tempdir().unwrap();
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");

    let mut child = Command::new(env!("CARGO_BIN_EXE_rro"))
        .env("RRO_LISTEN", &addr)
        .env("RRO_ESTATE", dir.path())
        // Weightless on purpose: this is about what is *stored*, not about
        // recall quality, and the test must not need a model server.
        .env("RRO_EMBEDDER", "deterministic")
        .env("RRO_RERANKER", "identity")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_ready(&addr);

    let reply = a2a(
        &addr,
        serde_json::json!({
            "id": "1", "from": "test", "to": "rro",
            "verb": "sql", "body": { "sql": "SELECT * LIMIT 100" }
        }),
    );
    let _ = child.kill();
    let _ = child.wait();

    let rows = reply["body"]["candidates"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let ids: Vec<&str> = rows.iter().filter_map(|r| r["id"].as_str()).collect();
    assert!(
        ids.is_empty(),
        "a fresh persistent estate must be EMPTY; found {ids:?} — the daemon is \
         seeding demo data into the operator's durable memory"
    );
}

/// The other half of the contract: an in-memory node still gets its demo corpus,
/// because a throwaway node with nothing to answer is a worse first run than one
/// with six sample documents. Deleting the seeding entirely would have been the
/// easy fix and the wrong one.
#[test]
fn an_in_memory_node_still_gets_the_demo_corpus() {
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");

    let mut child = Command::new(env!("CARGO_BIN_EXE_rro"))
        .env("RRO_LISTEN", &addr)
        .env_remove("RRO_ESTATE")
        .env("RRO_EMBEDDER", "deterministic")
        .env("RRO_RERANKER", "identity")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_ready(&addr);

    let reply = a2a(
        &addr,
        serde_json::json!({
            "id": "1", "from": "test", "to": "rro",
            "verb": "ask", "body": { "query": "vector search", "top_k": 3 }
        }),
    );
    let _ = child.kill();
    let _ = child.wait();

    let n = reply["body"]["candidates"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(
        n > 0,
        "an in-memory node must still seed its demo corpus — it has no other \
         way to answer anything"
    );
}
