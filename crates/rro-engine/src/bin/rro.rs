//! The `rro` daemon: the Reason Ready engine as an embedded, signal-driven
//! service with an optional a2a surface.
//!
//! Env:
//! - `RRO_LISTEN` — a2a TCP address (e.g. `127.0.0.1:7878`); unset = disabled.
//! - `RRO_ESTATE` — path to the persistent estate; unset = in-memory.
//! - `RRO_EVENTS` — JSONL event-stream path (DuckDB-ready); unset = disabled.
//! - `RUST_LOG`   — tracing filter (default `info`).
//! - `RRO_EMBEDDER` / `RRO_RERANKER` — model selection; see [`model_registry`].
//!   Unset = the weightless deterministic/lexical defaults.

use std::sync::Arc;

use model_registry::{build_embedder, build_reranker, EmbedderConfig, RerankerConfig};
use rro_engine::{estate_map, init_tracing, sample_corpus, serve, ReasonReadyObject, ServeOptions};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    // The event stream: every meaningful transition, consistently emitted,
    // straight into DuckDB via read_json_auto().
    if let Ok(path) = std::env::var("RRO_EVENTS") {
        let sink = rro_core::events::JsonlSink::open(&path)?;
        rro_core::events::set_sink(Box::new(sink));
        tracing::info!(path, "event stream enabled (JSONL)");
    }

    // With RRO_ESTATE set, memory is the persistent kvs estate (hybrid
    // dense + lexical recall); otherwise the in-memory default. Swap in
    // DevPULSE components here as they land.
    // RRD is the engine's front door: attached to every flow, baseline
    // restored from the estate so predictions are warm from the first query.
    let rrd = Arc::new(rrd::Rrd::new());

    // Model selection is data, not code (docs/MODELS.md §2): config -> boxed
    // trait. Resolving it HERE means a bad kind or a missing feature fails at
    // startup with an actionable message, instead of the daemon coming up and
    // quietly serving synthetic vectors under a real model's name.
    let embed_cfg = EmbedderConfig::from_env()?;
    let rerank_cfg = RerankerConfig::from_env()?;
    let embedder = build_embedder(&embed_cfg).await?;
    let reranker = build_reranker(&rerank_cfg).await?;
    tracing::info!(
        embedder = embed_cfg.kind.as_str(),
        model = embedder.model_name(),
        dim = embedder.dim(),
        reranker = rerank_cfg.kind.as_str(),
        device = ?embed_cfg.device,
        batch = embed_cfg.batch,
        "models selected"
    );

    // The estate must outlive the daemon: it owns the out-of-band ANN
    // applier thread (dropping it stops graph maintenance).
    let mut estate_handle: Option<Arc<connxism::Estate>> = None;
    let flow = match std::env::var("RRO_ESTATE").ok() {
        Some(path) => {
            let mut config = connxism::EstateConfig::default();
            if std::env::var("RRO_STRICT").map(|v| v == "1" || v == "true") == Ok(true) {
                config.quotas = connxism::Quotas::strict();
                tracing::info!("strict mode: {:?}", config.quotas);
            }
            let estate = Arc::new(connxism::Estate::open_with(&path, "rro", config)?);
            if let Some(snap) =
                estate.get_component_json::<rrd::BaselineSnapshot>("rrd:baseline")?
            {
                tracing::info!(
                    version = snap.version,
                    observations = snap.observations,
                    "rrd baseline restored"
                );
                rrd.restore_baseline(snap);
            }
            let map = estate_map(&estate)?;
            tracing::info!(
                estate = %estate.info().name,
                nodes = map.nodes.len(),
                edges = map.edges.len(),
                "opened estate"
            );
            let flow = ReasonReadyObject::builder()
                .rrd(rrd.clone())
                .recall(Arc::new(estate.recall()))
                .embedder(embedder.clone())
                .reranker(reranker.clone())
                .build();
            estate_handle = Some(estate);
            flow
        }
        None => ReasonReadyObject::builder()
            .rrd(rrd.clone())
            .embedder(embedder.clone())
            .reranker(reranker.clone())
            .build(),
    };
    // Seed the demo corpus ONLY into a throwaway in-memory node.
    //
    // This used to run unconditionally, outside the match — so every start of a
    // daemon with `RRO_ESTATE` set wrote six documents about banana bread and
    // Postgres upgrades into the operator's durable memory, and did it again on
    // every restart. For a product whose entire promise is "your AI remembers",
    // silently injecting demo data into that memory is about the worst possible
    // default. Observed live: a real recall returned `d5` (a2a protocols) over
    // the document that actually answered the question.
    //
    // An estate is the user's. We do not put anything in it that the user did
    // not put in it.
    if estate_handle.is_none() {
        let n = flow.index(sample_corpus()).await?;
        tracing::info!(
            indexed = n,
            "in-memory node: seeded the demo corpus (set RRO_ESTATE for a real, unseeded estate)"
        );
    }

    let opts = ServeOptions {
        node_id: std::env::var("RRO_NODE").unwrap_or_else(|_| "rro".to_string()),
        listen: std::env::var("RRO_LISTEN").ok(),
        http_listen: std::env::var("RRO_HTTP").ok(),
        estate: estate_handle,
        token: std::env::var("RRO_TOKEN").ok(),
    };

    // Ops HTTP surface (prometheus /metrics + health probes), when asked.
    let mut _ops_task = None;
    if let (Ok(ops_addr), Some(estate)) = (std::env::var("RRO_OPS_ADDR"), opts.estate.clone()) {
        let (bound, task) = rro_engine::ops::serve_ops(&ops_addr, estate).await?;
        tracing::info!(%bound, "ops surface up: /metrics /healthz /livez /readyz");
        _ops_task = Some(task);
    }

    let estate_for_shutdown = opts.estate.clone();
    serve(Arc::new(flow), opts).await?;

    // Commit the evolved baseline on the way out — the next boot restores it.
    if let Some(estate) = estate_for_shutdown {
        estate.put_component_json("rrd:baseline", &rrd.baseline_snapshot())?;
        tracing::info!("rrd baseline snapshot committed");
    }
    Ok(())
}
