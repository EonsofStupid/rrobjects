//! `rro-train` — contrastive fine-tuning of Qwen3-Embedding on nfcorpus.
//!
//! Feature-gated (`training`); needs weights + (ideally) a GPU. Produces a drop-in
//! weights dir that `RRO_EMBEDDER_WEIGHTS` can point the candle embedder at, so
//! the gate is just `rro-eval` on the tuned dir vs the stock dir.
//!
//! ```sh
//! RRO_TRAIN_WEIGHTS=$HOME/Projects/clyffy/models/embedders/qwen3-embedding-0-6b \
//! RRO_TRAIN_DATA=eval-data/nfcorpus \
//! RRO_TRAIN_OUT=models/qwen3-0.6b-nfcorpus-ft \
//! RRO_TRAIN_EPOCHS=1 RRO_TRAIN_BATCH=16 RRO_TRAIN_LR=2e-5 \
//!   cargo run --release --features training --bin rro-train
//! ```

use std::path::PathBuf;

use embedder::training::{load_pairs, nfcorpus_ndcg, TrainConfig, Trainer};

fn env_path(k: &str, default: &str) -> PathBuf {
    std::env::var(k).unwrap_or_else(|_| default.to_string()).into()
}
fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn env_f64(k: &str, default: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut cfg = TrainConfig::new(
        env_path(
            "RRO_TRAIN_WEIGHTS",
            &format!("{home}/Projects/clyffy/models/embedders/qwen3-embedding-0-6b"),
        ),
        env_path("RRO_TRAIN_DATA", "eval-data/nfcorpus"),
        env_path("RRO_TRAIN_OUT", "models/qwen3-0.6b-nfcorpus-ft"),
    );
    cfg.epochs = env_usize("RRO_TRAIN_EPOCHS", cfg.epochs);
    cfg.batch = env_usize("RRO_TRAIN_BATCH", cfg.batch);
    cfg.max_pairs = env_usize("RRO_TRAIN_MAX_PAIRS", cfg.max_pairs);
    cfg.max_len = env_usize("RRO_TRAIN_MAX_LEN", cfg.max_len);
    cfg.lr = env_f64("RRO_TRAIN_LR", cfg.lr);
    cfg.temperature = env_f64("RRO_TRAIN_TEMP", cfg.temperature);

    // Eval mode: load a weights dir and report its nfcorpus-test nDCG@10, using
    // the same GPU forward as training so stock vs tuned differ only in weights.
    if std::env::var("RRO_TRAIN_MODE").as_deref() == Ok("eval") {
        let trainer = Trainer::load(cfg.clone())?;
        let n = nfcorpus_ndcg(&trainer, &cfg.data_dir, 0)?;
        println!("nDCG@10 for {} = {n:.4}", cfg.weights_dir.display());
        return Ok(());
    }

    println!(
        "device: {:?}  epochs={} batch={} lr={} temp={} max_len={}",
        cfg.device, cfg.epochs, cfg.batch, cfg.lr, cfg.temperature, cfg.max_len
    );
    if matches!(cfg.device, candle_core::Device::Cpu) {
        eprintln!(
            "WARNING: training on CPU — fine for a small smoke run, impractical for a full run. \
             Build candle with the cuda feature to use the GPU."
        );
    }

    let pairs = load_pairs(&cfg.data_dir, cfg.max_pairs)?;
    println!("loaded {} (query, positive) pairs from {}", pairs.len(), cfg.data_dir.display());

    let started = std::time::Instant::now();
    let mut trainer = Trainer::load(cfg.clone())?;
    println!("model loaded in {:.1}s; training…", started.elapsed().as_secs_f64());

    let mut window_sum = 0.0f32;
    let mut window_n = 0usize;
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    let t = std::time::Instant::now();
    trainer.train(pairs, |step, loss| {
        if first.is_nan() {
            first = loss;
        }
        last = loss;
        window_sum += loss;
        window_n += 1;
        if step % 10 == 0 {
            let avg = window_sum / window_n as f32;
            println!(
                "step {step:>5}  loss {loss:.4}  (avg10 {avg:.4})  {:.1}s",
                t.elapsed().as_secs_f64()
            );
            window_sum = 0.0;
            window_n = 0;
        }
    })?;

    // In-memory eval BEFORE saving — isolates a training effect from a save/reload
    // bug: if this differs from stock but the reloaded checkpoint does not, the
    // round-trip is dropping the trained weights.
    let in_mem = nfcorpus_ndcg(&trainer, &cfg.data_dir, 0)?;
    println!("in-memory (live encoder) tuned nDCG@10 = {in_mem:.4}");

    // Diagnostic: rebuild the encoder from the varmap (the save source).
    trainer.rebuild_encoder_from_varmap()?;
    let via_varmap = nfcorpus_ndcg(&trainer, &cfg.data_dir, 0)?;
    println!("via-varmap (rebuilt encoder) nDCG@10 = {via_varmap:.4}");

    trainer.save()?;
    println!(
        "\nfirst-step loss {first:.4} → last-step loss {last:.4}\n\
         saved checkpoint to {}\n\
         GATE: eval it vs stock —\n  \
         RRO_EMBEDDER=candle-qwen RRO_EMBEDDER_WEIGHTS={} RRO_RERANKER=vllm \\\n  \
         RRO_EVAL_DATA={} cargo run --release --features … --bin rro-eval",
        cfg.out_dir.display(),
        cfg.out_dir.display(),
        cfg.data_dir.display()
    );
    Ok(())
}
