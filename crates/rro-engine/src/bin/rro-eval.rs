//! `rro-eval` — the honest number. Real IR benchmarks, real models, ablations.
//!
//! Every accuracy figure RRO has ever published came from the deterministic
//! hash embedder scoring synthetic vectors against synthetic vectors. This
//! binary exists to replace those with numbers that mean something, on public
//! benchmarks with third-party relevance judgments nobody here wrote.
//!
//! ## The ablation ladder
//!
//! Results are dismissible unless you can see what each layer bought. So every
//! run reports the same queries through progressively more of the engine:
//!
//! | arm | path |
//! |---|---|
//! | `bm25`   | lexical only — the floor. No model, no vectors. |
//! | `dense`  | ANN over embeddings only. |
//! | `hybrid` | dense + BM25, reciprocal-rank-fused. |
//! | `rro`    | hybrid + cross-encoder rerank (the full object). |
//!
//! If `hybrid` doesn't beat `dense`, fusion isn't earning its cost. If `rro`
//! doesn't beat `hybrid`, the reranker isn't earning its latency. Either is a
//! real finding and gets printed, not buried.
//!
//! ## Metrics
//!
//! nDCG@10 (the BEIR/BRIGHT standard, and graded — nfcorpus qrels score 0..2),
//! plus Recall@10 and MRR@10. nDCG is the headline because it is what published
//! baselines report, which is the only way a claim here is checkable.
//!
//! ## Usage
//!
//! ```sh
//! RRO_EMBEDDER=llamacpp \
//! RRO_EVAL_DATA=eval-data/nfcorpus \
//!   cargo run --release --bin rro-eval
//!
//! # add the reranker arm
//! RRO_RERANKER=vllm RRO_EMBEDDER=llamacpp RRO_EVAL_DATA=eval-data/nfcorpus \
//!   cargo run --release --bin rro-eval
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use model_registry::{build_embedder, build_reranker, EmbedderConfig, RerankerConfig};
use rro_core::events::{Event, EventSink};
use rro_core::{Candidate, Document, Recall, Reranker, VectorRecord};

/// One benchmark query with its graded judgments.
struct EvalQuery {
    id: String,
    text: String,
    /// doc id -> relevance grade (>0 is relevant; nfcorpus uses 0..2).
    rels: HashMap<String, u8>,
}

/// Collects the `flow.stage` timings the engine already emits, so latency can be
/// attributed per stage instead of wall-clocked around the whole call.
///
/// This exists because the first version of this harness timed its own loop and
/// printed the total under a column called `ms/query` — which meant an HTTP
/// round-trip plus a 4B model forward plus 100 cross-encoder pairs were all
/// reported as if they were engine latency (1167 ms). The engine's own `recall`
/// stage is 0.404 ms. Model cost and engine cost are different things, and a
/// harness that adds them together is measuring nothing anyone can act on.
#[derive(Default)]
struct StageCollector {
    /// stage name -> observed durations (ms)
    stages: Mutex<HashMap<String, Vec<f64>>>,
}

impl EventSink for StageCollector {
    fn record(&self, event: Event) {
        if event.kind != rro_core::semconv::EVENT_STAGE {
            return;
        }
        let (Some(name), Some(ms)) = (
            event
                .fields
                .get(rro_core::semconv::attr::STAGE)
                .and_then(|v| v.as_str()),
            event
                .fields
                .get(rro_core::semconv::attr::LATENCY_MS)
                .and_then(|v| v.as_f64()),
        ) else {
            return;
        };
        self.stages
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_default()
            .push(ms);
    }
}

impl StageCollector {
    /// Mean ms per stage, and clear — so each arm reports only its own passes.
    fn drain_means(&self) -> Vec<(String, f64, usize)> {
        let mut g = self.stages.lock().unwrap();
        let mut out: Vec<(String, f64, usize)> = g
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().sum::<f64>() / v.len() as f64, v.len()))
            .collect();
        g.clear();
        // Pipeline order, not alphabetical — this is a pipeline.
        let order = ["shape", "embed", "intent", "recall", "rerank", "reason"];
        out.sort_by_key(|(k, _, _)| order.iter().position(|o| o == k).unwrap_or(99));
        out
    }
}

fn main() -> anyhow::Result<()> {
    rro_engine::init_tracing();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run())
}

/// The collector must outlive the sink registration (`set_sink` takes a Box and
/// the sink is global), so it lives here and both sides share it.
static STAGES: OnceLock<Arc<StageCollector>> = OnceLock::new();

/// A thin forwarder, because `set_sink` consumes a Box but we also want to read
/// the collected timings back out.
struct SinkHandle(Arc<StageCollector>);
impl EventSink for SinkHandle {
    fn record(&self, event: Event) {
        self.0.record(event)
    }
}

async fn run() -> anyhow::Result<()> {
    let collector = STAGES
        .get_or_init(|| Arc::new(StageCollector::default()))
        .clone();
    rro_core::events::set_sink(Box::new(SinkHandle(collector.clone())));

    let dir: PathBuf = std::env::var("RRO_EVAL_DATA")
        .unwrap_or_else(|_| "eval-data/nfcorpus".to_string())
        .into();
    let top_k: usize = env_usize("RRO_EVAL_K", 10);
    let recall_k: usize = env_usize("RRO_EVAL_RECALL_K", 100);
    let max_queries: usize = env_usize("RRO_EVAL_MAX_QUERIES", 0);
    let max_docs: usize = env_usize("RRO_EVAL_MAX_DOCS", 0);

    // ---- load real data ---------------------------------------------------
    let docs = load_corpus(&dir, max_docs)?;
    let queries = load_queries(&dir, max_queries)?;
    println!("corpus: {} docs   queries: {}", docs.len(), queries.len());
    if queries.is_empty() || docs.is_empty() {
        anyhow::bail!("no data at {} — is RRO_EVAL_DATA right?", dir.display());
    }

    // ---- models -----------------------------------------------------------
    let ecfg = EmbedderConfig::from_env()?;
    let embedder = build_embedder(&ecfg).await?;
    println!(
        "embedder: {} ({}) dim={}",
        ecfg.kind.as_str(),
        embedder.model_name(),
        embedder.dim()
    );

    // The reranker arm only runs when explicitly asked for; the default lexical
    // reranker would make `rro` identical to `hybrid` and imply a lift that
    // isn't there.
    let rcfg = RerankerConfig::from_env()?;
    let reranker: Option<Arc<dyn Reranker>> = if std::env::var("RRO_RERANKER").is_ok() {
        let r = build_reranker(&rcfg).await?;
        println!("reranker: {} ({})", rcfg.kind.as_str(), r.model_name());
        Some(r)
    } else {
        println!("reranker: none (set RRO_RERANKER to add the `rro` arm)");
        None
    };

    // ---- index ------------------------------------------------------------
    let estate_dir = tempfile::tempdir()?;
    // THE ANALYZER IS NOT A DETAIL. `Analyzer::default()` is the LEGACY pipeline
    // — word tokens, lowercased, stopword-filtered, and UNSTEMMED. Running the
    // lexical arm unstemmed on morphology-heavy text (nfcorpus is biomedical:
    // statin/statins, cancer/cancers, treat/treating) cripples BM25, and then
    // fusing that crippled ranking into a strong dense one looks exactly like
    // "fusion hurts" — a missed optimization wearing a regression's clothes.
    // Default to stemming here and make it switchable, so the claim is testable
    // rather than an artifact of a default nobody looked at.
    let analyzer = match std::env::var("RRO_EVAL_ANALYZER").as_deref() {
        Ok("legacy") => rro_core::text::Analyzer::default(),
        _ => rro_core::text::Analyzer::stemming(),
    };
    println!(
        "analyzer: {} (RRO_EVAL_ANALYZER=legacy|stemming)",
        if analyzer.stem {
            "stemming"
        } else {
            "legacy/unstemmed"
        }
    );
    let estate = Arc::new(connxism::Estate::open_with(
        estate_dir.path().to_str().unwrap(),
        "eval",
        connxism::EstateConfig {
            analyzer,
            ..Default::default()
        },
    )?);
    let recall = estate.recall();

    let t = Instant::now();
    let texts: Vec<String> = docs.iter().map(|d| d.text.clone()).collect();
    let embeddings = embedder.embed_documents(&texts).await?;
    let embed_secs = t.elapsed().as_secs_f64();
    println!(
        "embedded {} docs in {embed_secs:.1}s ({:.0} docs/sec)",
        docs.len(),
        docs.len() as f64 / embed_secs
    );

    // RRO_EVAL_EXPORT_VECTORS: dump the REAL embeddings we just paid for, so the
    // ANN ef sweep (crates/recall/tests/real_vector_ef.rs) can re-tune against
    // real vectors without re-running the model.
    if let Ok(path) = std::env::var("RRO_EVAL_EXPORT_VECTORS") {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for (d, e) in docs.iter().zip(embeddings.iter()) {
            let line = serde_json::json!({
                "kind": "doc", "id": d.id.as_str(), "vector": e.0,
            });
            writeln!(f, "{line}")?;
        }
        println!("exported {} real vectors -> {path}", docs.len());
    }

    let records: Vec<VectorRecord> = docs
        .iter()
        .zip(embeddings)
        .map(|(d, e)| VectorRecord::new(d.id.clone(), e, d.text.clone()))
        .collect();
    let t = Instant::now();
    recall.upsert(records).await?;
    println!("indexed in {:.1}s", t.elapsed().as_secs_f64());

    // ---- the full engine, for the `rro` arm -------------------------------
    // The lower rungs call the estate directly (that IS what they are: raw
    // retrieval). The `rro` rung must be the REAL pass — RRD gate -> embed ->
    // intent -> recall -> rerank -> classify -> connectome map — which
    // `ReasonReadyObject::ask()` already composes. The previous version of this
    // harness hand-rolled hybrid_search+rerank and called that "rro": two stages
    // of a six-stage engine, published under the product's name.
    let rrd = Arc::new(rrd::Rrd::new());
    let flow = rro_engine::ReasonReadyObject::builder()
        .rrd(rrd.clone())
        .recall(Arc::new(estate.recall()))
        .embedder(embedder.clone())
        .config(rro_engine::ObjectConfig {
            recall_k,
            rerank_k: top_k,
        });
    let flow = match reranker.clone() {
        Some(r) => flow.reranker(r).build(),
        None => flow.build(),
    };

    // ---- run the ladder ---------------------------------------------------
    // (arm name, per-query nDCG@10, recall@k, MRR, wall-seconds).
    type ArmResult<'a> = (&'a str, Vec<f64>, Vec<f64>, Vec<f64>, f64);
    let mut arms: Vec<ArmResult> = Vec::new();
    let names: Vec<&str> = if reranker.is_some() {
        vec!["bm25", "dense", "hybrid", "hybrid+rerank", "rro"]
    } else {
        vec!["bm25", "dense", "hybrid", "rro"]
    };
    let mut names: Vec<String> = names.into_iter().map(String::from).collect();

    // The fusion weight sweep. Fusion is query-time only, so every one of these
    // arms reuses the single (expensive) embed+index pass above — the sweep costs
    // milliseconds, not another model run per weight.
    //
    // This exists because `hybrid` scoring BELOW `dense` was nearly published as
    // "fusion hurts". Unweighted RRF gives a 0.3283 retriever exactly the same
    // vote as a 0.4119 one; the honest question is not "does fusion hurt" but
    // "at what weight does it help, and does it EVER beat dense". The sweep is
    // what makes that answerable instead of arguable.
    let sweep: Vec<f32> = std::env::var("RRO_EVAL_SWEEP")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    for w in &sweep {
        names.push(format!("hybrid:w{w}"));
    }

    for arm in &names {
        let arm = arm.as_str();
        let mut ndcg = Vec::new();
        let mut rec = Vec::new();
        let mut mrr = Vec::new();
        let mut per_query: Vec<(String, f64)> = Vec::new();
        let t = Instant::now();

        for q in &queries {
            // The `rro` arm embeds inside ask(); embedding here too would pay the
            // model cost twice and misattribute it.
            let qv = if arm == "rro" {
                rro_core::Embedding(Vec::new())
            } else {
                embedder.embed_query_one(&q.text).await?
            };
            let hits: Vec<Candidate> = match arm {
                // TRUE lexical-only. The first version of this faked BM25 by
                // handing hybrid_search a zero vector — but fusion still ran,
                // blending a degenerate dense ranking into the lexical one, and
                // scored 0.159 against a published BM25 baseline of ~0.325.
                // Half the real number. A baseline that is broken LOW makes
                // every arm above it look good, which is the most flattering
                // possible bug and therefore the one to be most suspicious of.
                "bm25" => recall.lexical_search(&q.text, recall_k).await?,
                "dense" => recall.search(&qv, recall_k).await?,
                "hybrid" => recall.hybrid_search(&q.text, &qv, recall_k).await?,
                // hybrid:w<D> — dense weighted D:1 against lexical, via the typed
                // query path (fusion is a property of the query, not the estate).
                a if a.starts_with("hybrid:w") => {
                    let d: f32 = a.trim_start_matches("hybrid:w").parse().unwrap_or(1.0);
                    recall
                        .query(connxism::EstateQuery {
                            text: Some(q.text.clone()),
                            vector: Some(qv.clone()),
                            top_k: recall_k,
                            fusion: connxism::HybridWeights {
                                dense: d,
                                lexical: 1.0,
                            },
                            ..Default::default()
                        })
                        .await?
                }
                "hybrid+rerank" => {
                    let c = recall.hybrid_search(&q.text, &qv, recall_k).await?;
                    reranker.as_ref().unwrap().rerank(&q.text, c, top_k).await?
                }
                // The real engine: RRD gate -> embed -> intent -> recall ->
                // rerank -> classify. Its per-stage timings are captured by the
                // event sink, not by wall-clocking this call.
                "rro" => flow.ask(&q.text).await?.candidates,
                _ => unreachable!(),
            };
            let ids: Vec<&str> = hits.iter().take(top_k).map(|c| c.id.as_str()).collect();
            let n = ndcg_at_k(&ids, &q.rels, top_k);
            per_query.push((q.id.clone(), n));
            ndcg.push(n);
            rec.push(recall_at_k(&ids, &q.rels));
            mrr.push(mrr_at_k(&ids, &q.rels));
        }
        let secs = t.elapsed().as_secs_f64();
        let stage_ms = collector.drain_means();
        // The mean hides everything. Show where this arm actually fails, so a
        // headline number can be argued with rather than taken on faith.
        per_query.sort_by(|a, b| a.1.total_cmp(&b.1));
        let worst: Vec<String> = per_query
            .iter()
            .take(3)
            .map(|(id, n)| format!("{id}={n:.3}"))
            .collect();
        println!("  {arm:<14} worst queries: {}", worst.join("  "));
        if !stage_ms.is_empty() {
            let parts: Vec<String> = stage_ms
                .iter()
                .map(|(k, ms, n)| format!("{k}={ms:.3}ms(n={n})"))
                .collect();
            println!("  {:<14} engine stages: {}", "", parts.join("  "));
        }
        arms.push((arm, ndcg, rec, mrr, secs));
    }

    // ---- report -----------------------------------------------------------
    println!(
        "\n=== {} | {} queries | top_k={top_k} recall_k={recall_k} ===",
        dir.display(),
        queries.len()
    );
    println!(
        "{:<8} {:>9} {:>10} {:>9} {:>12}",
        "arm", "nDCG@10", "Recall@10", "MRR@10", "wall ms/query"
    );
    let mut prev_ndcg = 0.0;
    for (name, ndcg, rec, mrr, secs) in &arms {
        let n = mean(ndcg);
        let delta = if prev_ndcg > 0.0 {
            format!("  ({:+.1}% vs prev)", (n / prev_ndcg - 1.0) * 100.0)
        } else {
            String::new()
        };
        println!(
            "{:<14} {:>9.4} {:>10.4} {:>9.4} {:>14.1}{delta}",
            name,
            n,
            mean(rec),
            mean(mrr),
            secs * 1000.0 / queries.len() as f64
        );
        prev_ndcg = n;
    }

    // ---- significance -------------------------------------------------------
    // A mean without a confidence interval is a claim without evidence. Every arm
    // scored the same queries, so the scores are paired: bootstrap each arm's CI,
    // and test each step of the ladder (arm vs the one before it) with a paired
    // sign-flip permutation test. This is what lets "dense beats BM25" or "fusion
    // hurts" be stated as findings rather than impressions.
    let iters = env_usize("RRO_EVAL_BOOTSTRAP", 10_000);
    let seed = std::env::var("RRO_EVAL_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x00C0_FFEE);
    println!("\n=== significance (paired, {iters} resamples, seed 0x{seed:X}) ===");
    println!("{:<14} {:>9}  {:>20}", "arm", "nDCG@10", "95% CI");
    for (name, ndcg, ..) in &arms {
        let (lo, hi) = sig::bootstrap_ci(ndcg, iters, seed);
        println!("{name:<14} {:>9.4}  [{lo:.4}, {hi:.4}]", mean(ndcg));
    }
    println!(
        "\n{:<26} {:>9}  {:>20}  {:>8}",
        "step (arm vs previous)", "Δ nDCG", "95% CI", "p"
    );
    for pair in arms.windows(2) {
        let (base_name, base, ..) = &pair[0];
        let (arm_name, arm, ..) = &pair[1];
        let r = sig::paired(arm, base, iters, seed);
        let verdict = if r.p < 0.05 { "" } else { "  (n.s.)" };
        println!(
            "{:<26} {:>+9.4}  [{:+.4}, {:+.4}]  {:>8.4}{verdict}",
            format!("{arm_name} vs {base_name}"),
            r.delta,
            r.ci.0,
            r.ci.1,
            r.p
        );
    }
    println!(
        "\nA step marked (n.s.) is not statistically distinguishable at p<0.05 — its\n\
         apparent lift or drop is within paired-resampling noise on these queries."
    );

    println!(
        "\nNOTE: `wall ms/query` is END-TO-END and is dominated by MODEL time (an HTTP\n\
         round-trip + a forward pass, plus rerank pairs). It is NOT engine latency —\n\
         see the per-stage `engine stages` lines above for that (recall is sub-ms).\n"
    );
    println!("Published BEIR nDCG@10 for reference — BM25 ~0.325, and strong dense");
    println!("models land ~0.32-0.38 on nfcorpus. A number far above that band means");
    println!("the harness is wrong, not that the engine is magic.");
    Ok(())
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

/// Significance: turn per-query point estimates into defensible claims.
///
/// A single nDCG mean is a point estimate — "dense beats BM25" could be a real
/// effect or per-query noise. Because every arm scores the SAME queries in the
/// SAME order, the scores are **paired**, so two rigorous, distribution-free
/// tools apply:
///   - a **percentile bootstrap** 95% CI on each arm's mean (resample queries
///     with replacement), and
///   - a **paired sign-flip permutation test** on the per-query delta between two
///     arms (the textbook exact-ish test for paired data): under H0 the sign of
///     each query's delta is exchangeable, so flipping signs at random builds the
///     null distribution of the mean delta.
///
/// Both are seeded (default 0xC0FFEE) so a published number reproduces exactly.
mod sig {
    /// SplitMix64 — a tiny, well-distributed, deterministic PRNG. No `rand` dep,
    /// matching the tree; a fixed seed makes every CI and p-value reproducible.
    pub struct Rng(u64);
    impl Rng {
        pub fn new(seed: u64) -> Self {
            Rng(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        /// Uniform in `[0, n)`.
        fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
        /// A random sign, +1.0 or -1.0.
        fn sign(&mut self) -> f64 {
            if self.next_u64() & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        }
    }

    fn mean(xs: &[f64]) -> f64 {
        xs.iter().sum::<f64>() / xs.len() as f64
    }

    fn percentile(sorted: &[f64], p: f64) -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let idx = ((p * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
        sorted[idx]
    }

    /// Percentile bootstrap 95% CI of the mean of `values`.
    pub fn bootstrap_ci(values: &[f64], iters: usize, seed: u64) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        let mut rng = Rng::new(seed);
        let n = values.len();
        let mut means: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut acc = 0.0;
            for _ in 0..n {
                acc += values[rng.below(n)];
            }
            means.push(acc / n as f64);
        }
        means.sort_by(f64::total_cmp);
        (percentile(&means, 0.025), percentile(&means, 0.975))
    }

    /// The outcome of comparing two arms on the same queries.
    pub struct Paired {
        /// Mean per-query delta (a − b).
        pub delta: f64,
        /// 95% bootstrap CI of the mean delta.
        pub ci: (f64, f64),
        /// Two-sided p-value from the sign-flip permutation test.
        pub p: f64,
    }

    /// Paired comparison of two equal-length, query-aligned score vectors.
    /// CI by bootstrap of the mean delta; p by sign-flip permutation.
    pub fn paired(a: &[f64], b: &[f64], iters: usize, seed: u64) -> Paired {
        assert_eq!(a.len(), b.len(), "paired arms must be query-aligned");
        let deltas: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        let observed = mean(&deltas);

        // Bootstrap CI of the mean delta.
        let mut rng = Rng::new(seed);
        let n = deltas.len();
        let mut boot: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut acc = 0.0;
            for _ in 0..n {
                acc += deltas[rng.below(n)];
            }
            boot.push(acc / n as f64);
        }
        boot.sort_by(f64::total_cmp);
        let ci = (percentile(&boot, 0.025), percentile(&boot, 0.975));

        // Sign-flip permutation test: under H0 (no effect), each delta's sign is
        // exchangeable. Count permutations whose |mean| ≥ |observed|.
        let mut rng = Rng::new(seed ^ 0x5555_5555_5555_5555);
        let mut extreme = 0usize;
        for _ in 0..iters {
            let m: f64 = deltas.iter().map(|d| d * rng.sign()).sum::<f64>() / n as f64;
            if m.abs() >= observed.abs() {
                extreme += 1;
            }
        }
        // Add-one smoothing: a p-value is never exactly 0 from a finite sample.
        let p = (extreme as f64 + 1.0) / (iters as f64 + 1.0);
        Paired {
            delta: observed,
            ci,
            p,
        }
    }
}

/// nDCG@k with graded relevance (the BEIR standard).
fn ndcg_at_k(ids: &[&str], rels: &HashMap<String, u8>, k: usize) -> f64 {
    let dcg: f64 = ids
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let g = *rels.get(*id).unwrap_or(&0) as f64;
            (2f64.powf(g) - 1.0) / ((i + 2) as f64).log2()
        })
        .sum();
    // Ideal: the best k grades this query could possibly have returned.
    let mut ideal: Vec<u8> = rels.values().copied().filter(|g| *g > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, g)| (2f64.powf(*g as f64) - 1.0) / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

fn recall_at_k(ids: &[&str], rels: &HashMap<String, u8>) -> f64 {
    let total = rels.values().filter(|g| **g > 0).count();
    if total == 0 {
        return 0.0;
    }
    let found = ids
        .iter()
        .filter(|id| rels.get(**id).is_some_and(|g| *g > 0))
        .count();
    found as f64 / total as f64
}

fn mrr_at_k(ids: &[&str], rels: &HashMap<String, u8>) -> f64 {
    for (i, id) in ids.iter().enumerate() {
        if rels.get(*id).is_some_and(|g| *g > 0) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// BEIR/MTEB `corpus.jsonl`: `{_id, title, text}`. Title is prepended — the
/// standard BEIR convention, and dropping it measurably hurts every model
/// equally, so it would flatter nothing but would not be comparable to
/// published numbers.
fn load_corpus(dir: &Path, limit: usize) -> anyhow::Result<Vec<Document>> {
    let f = std::fs::read_to_string(dir.join("corpus.jsonl"))?;
    let mut out = Vec::new();
    for line in f.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let title = v["title"].as_str().unwrap_or_default();
        let text = v["text"].as_str().unwrap_or_default();
        let full = if title.is_empty() {
            text.to_string()
        } else {
            format!("{title}. {text}")
        };
        out.push(Document::new(full).with_id(id));
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

/// `queries.jsonl` + `qrels/test.tsv`. Only queries WITH judgments are scored:
/// an unjudged query has no ground truth, so including it would score 0 and
/// silently drag every arm down by the same amount — noise, not signal.
fn load_queries(dir: &Path, limit: usize) -> anyhow::Result<Vec<EvalQuery>> {
    let mut qrels: HashMap<String, HashMap<String, u8>> = HashMap::new();
    let tsv = std::fs::read_to_string(dir.join("qrels/test.tsv"))?;
    for (i, line) in tsv.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue; // header
        }
        let mut it = line.split('\t');
        let (Some(q), Some(d), Some(s)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let score: u8 = s.trim().parse().unwrap_or(0);
        if score > 0 {
            qrels
                .entry(q.to_string())
                .or_default()
                .insert(d.to_string(), score);
        }
    }

    let f = std::fs::read_to_string(dir.join("queries.jsonl"))?;
    let mut out = Vec::new();
    for line in f.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let Some(rels) = qrels.remove(&id) else {
            continue;
        };
        out.push(EvalQuery {
            id,
            text: v["text"].as_str().unwrap_or_default().to_string(),
            rels,
        });
        if limit > 0 && out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::sig;

    #[test]
    fn bootstrap_ci_brackets_the_mean() {
        let v: Vec<f64> = (0..100).map(|i| (i as f64) / 100.0).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let (lo, hi) = sig::bootstrap_ci(&v, 5000, 0xC0FFEE);
        assert!(lo < m && m < hi, "CI [{lo},{hi}] must bracket mean {m}");
        assert!(hi - lo < 0.15, "CI width {} unexpectedly wide", hi - lo);
    }

    #[test]
    fn identical_arms_are_not_significant() {
        let a: Vec<f64> = (0..80).map(|i| ((i * 7) % 11) as f64 / 10.0).collect();
        let r = sig::paired(&a, &a, 5000, 0xC0FFEE);
        assert_eq!(r.delta, 0.0, "identical arms have zero delta");
        assert!(
            r.p > 0.99,
            "identical arms cannot be significant, p={}",
            r.p
        );
        assert!(
            r.ci.0 <= 0.0 && r.ci.1 >= 0.0,
            "CI must contain 0 for identical arms: {:?}",
            r.ci
        );
    }

    #[test]
    fn a_consistent_per_query_lift_is_significant() {
        // Arm A beats arm B on every query by ~0.1 (with a little jitter). A
        // real, consistent effect must come back significant with a CI above 0.
        let b: Vec<f64> = (0..60).map(|i| 0.3 + ((i % 5) as f64) * 0.02).collect();
        let a: Vec<f64> = b
            .iter()
            .enumerate()
            .map(|(i, x)| x + 0.10 + if i % 3 == 0 { 0.02 } else { -0.01 })
            .collect();
        let r = sig::paired(&a, &b, 10_000, 0xC0FFEE);
        assert!(r.delta > 0.08, "delta {} should be ~0.1", r.delta);
        assert!(
            r.p < 0.01,
            "a consistent lift must be significant, p={}",
            r.p
        );
        assert!(r.ci.0 > 0.0, "95% CI {:?} must exclude 0", r.ci);
    }

    #[test]
    fn pure_noise_is_not_significant() {
        // Symmetric per-query differences with no real effect → mean ≈ 0, n.s.
        let a: Vec<f64> = (0..80)
            .map(|i| if i % 2 == 0 { 0.5 } else { 0.3 })
            .collect();
        let b: Vec<f64> = (0..80)
            .map(|i| if i % 2 == 0 { 0.3 } else { 0.5 })
            .collect();
        let r = sig::paired(&a, &b, 10_000, 0xC0FFEE);
        assert!(r.delta.abs() < 1e-9, "symmetric deltas cancel: {}", r.delta);
        assert!(r.p > 0.05, "noise must be n.s., p={}", r.p);
    }

    #[test]
    fn same_seed_reproduces_exactly() {
        let a: Vec<f64> = (0..50).map(|i| (i as f64).sin().abs()).collect();
        let b: Vec<f64> = (0..50).map(|i| (i as f64 * 1.3).cos().abs()).collect();
        let r1 = sig::paired(&a, &b, 3000, 42);
        let r2 = sig::paired(&a, &b, 3000, 42);
        assert_eq!(r1.p, r2.p, "same seed → same p");
        assert_eq!(r1.ci, r2.ci, "same seed → same CI");
        let r3 = sig::paired(&a, &b, 3000, 43);
        assert!(r3.ci.0.is_finite());
    }
}
