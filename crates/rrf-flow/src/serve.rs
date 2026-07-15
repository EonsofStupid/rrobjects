//! The embedded, signal-driven runtime.
//!
//! `serve` runs the engine as a long-lived daemon: it optionally opens the a2a
//! TCP surface and then parks on OS shutdown signals (Ctrl-C / SIGTERM),
//! draining cleanly. This is "embedded" in the sense that matters — it is one
//! tokio process you can drop into a host — without giving up the network.

use std::sync::Arc;

use rrf_core::Result;
use rrf_net::NodeId;

use crate::flow::ReasonReadyFlow;
use crate::handler::FlowNode;

/// How the daemon should run.
#[derive(Clone, Default)]
pub struct ServeOptions {
    /// This node's a2a id.
    pub node_id: String,
    /// If set, open the a2a TCP surface on this address (e.g. `127.0.0.1:7878`).
    pub listen: Option<String>,
    /// Estate to expose on the a2a surface (`changes` subscription paging).
    pub estate: Option<std::sync::Arc<connxism::Estate>>,
}

impl ServeOptions {
    /// Options with the default node id and no listener.
    pub fn named(node_id: impl Into<String>) -> Self {
        ServeOptions {
            node_id: node_id.into(),
            ..ServeOptions::default()
        }
    }
}

/// Run the engine until a shutdown signal arrives.
pub async fn serve(flow: Arc<ReasonReadyFlow>, opts: ServeOptions) -> Result<()> {
    for (stage, model) in flow.model_names() {
        tracing::info!(stage, model, "component");
    }

    let mut node = FlowNode::new(flow.clone(), NodeId::new(&opts.node_id));
    if let Some(estate) = &opts.estate {
        node = node.with_estate(estate.clone());
    }
    let node = Arc::new(node);

    let _server = match &opts.listen {
        Some(addr) => {
            let (bound, task) = rrf_net::tcp::serve(addr.clone(), node.clone()).await?;
            tracing::info!(%bound, "a2a surface listening");
            Some(task)
        }
        None => {
            tracing::info!("a2a surface disabled (set listen to enable)");
            None
        }
    };

    rrf_core::events::emit(
        "serve.start",
        serde_json::json!({ "node_id": opts.node_id, "listen": opts.listen }),
    );
    tracing::info!("reason ready — awaiting shutdown signal");
    let sig = wait_for_shutdown().await;
    tracing::info!(signal = sig, "shutdown signal received — stopping");
    rrf_core::events::emit("serve.stop", serde_json::json!({ "signal": sig }));
    Ok(())
}

/// Block until an OS shutdown signal arrives; returns which one.
///
/// The full Unix set is handled — `SIGHUP`, `SIGINT`, `SIGQUIT`, `SIGTERM` —
/// and every receipt is consistently emitted as a `signal.received` event
/// before this returns, so the analytics stream always shows *why* the
/// process stopped. Non-Unix falls back to Ctrl-C.
pub async fn wait_for_shutdown() -> &'static str {
    let sig = listen_for_signal().await;
    rrf_core::events::emit("signal.received", serde_json::json!({ "signal": sig }));
    sig
}

#[cfg(unix)]
async fn listen_for_signal() -> &'static str {
    use tokio::signal::unix::{signal, SignalKind};
    let installed = (
        signal(SignalKind::hangup()),
        signal(SignalKind::interrupt()),
        signal(SignalKind::quit()),
        signal(SignalKind::terminate()),
    );
    match installed {
        (Ok(mut hup), Ok(mut int), Ok(mut quit), Ok(mut term)) => {
            tokio::select! {
                _ = hup.recv() => "SIGHUP",
                _ = int.recv() => "SIGINT",
                _ = quit.recv() => "SIGQUIT",
                _ = term.recv() => "SIGTERM",
            }
        }
        _ => {
            tracing::warn!("cannot install full signal set; falling back to Ctrl-C");
            let _ = tokio::signal::ctrl_c().await;
            "SIGINT"
        }
    }
}

#[cfg(not(unix))]
async fn listen_for_signal() -> &'static str {
    let _ = tokio::signal::ctrl_c().await;
    "CTRL_C"
}
