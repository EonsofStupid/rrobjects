//! Contrastive fine-tuning of the Qwen3 embedder (candle-nn).
//!
//! ## Why this exists, and what it honestly is
//!
//! The `candle_qwen` encoder is a hand-rolled, stateless, autodiff-friendly Qwen3
//! forward built from `candle_nn` primitives. Loaded through a [`VarMap`]-backed
//! [`VarBuilder`], every weight becomes a trainable [`Var`], so the *same* forward
//! that serves inference also backpropagates — no second model.
//!
//! Training is **contrastive**: batches of `(query, positive passage)` pairs from
//! the nfcorpus TRAIN split, with the other positives in the batch as negatives
//! (in-batch InfoNCE). The produced checkpoint is a drop-in weights dir the
//! [`crate::CandleQwenEmbedder`] loads unchanged.
//!
//! **The gate is honest and uncertain.** Stock Qwen3-0.6B is already strong;
//! fine-tuning it on a few thousand in-domain queries may or may not raise
//! nDCG@10 on the held-out test split. This code produces the checkpoint and the
//! number; whether the number wins is for `rro-eval` (and the operator) to judge,
//! never asserted here.

mod data;

pub use data::{load_pairs, TrainPair};

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Optimizer, VarBuilder, VarMap};
use rro_core::{Result, RroError};
use tokenizers::Tokenizer;

use crate::{Qwen3Encoder, DEFAULT_QUERY_TASK};

fn err(m: impl Into<String>) -> RroError {
    RroError::Embed(m.into())
}

/// Fine-tuning hyperparameters and paths.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Stock Qwen3 weights dir (the starting point and the stock baseline).
    pub weights_dir: PathBuf,
    /// nfcorpus-layout data dir (corpus/queries/qrels).
    pub data_dir: PathBuf,
    /// Where the fine-tuned checkpoint is written (a drop-in weights dir).
    pub out_dir: PathBuf,
    /// Where to train.
    pub device: Device,
    /// Pairs per step (also the number of in-batch negatives).
    pub batch: usize,
    /// Passes over the data.
    pub epochs: usize,
    /// AdamW learning rate.
    pub lr: f64,
    /// InfoNCE temperature (lower = sharper).
    pub temperature: f64,
    /// Max tokens per text (short for retrieval passages; caps activation memory).
    pub max_len: usize,
    /// Cap on training pairs (0 = all).
    pub max_pairs: usize,
}

impl TrainConfig {
    /// Defaults tuned for a single-GPU nfcorpus run.
    pub fn new(
        weights_dir: impl Into<PathBuf>,
        data_dir: impl Into<PathBuf>,
        out_dir: impl Into<PathBuf>,
    ) -> Self {
        TrainConfig {
            weights_dir: weights_dir.into(),
            data_dir: data_dir.into(),
            out_dir: out_dir.into(),
            device: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            batch: 16,
            epochs: 1,
            lr: 2e-5,
            temperature: 0.05,
            max_len: 192,
            max_pairs: 0,
        }
    }
}

/// The symmetric in-batch InfoNCE loss over L2-normalized query/passage rows.
///
/// `q` and `d` are `[B, H]` and unit-normalized. Row `i` of `q` matches row `i` of
/// `d` (the positive); every other row is a negative. The loss is the mean of the
/// query→passage and passage→query cross-entropies over the `[B, B]` similarity
/// matrix scaled by `1/temperature`.
pub fn info_nce(q: &Tensor, d: &Tensor, temperature: f64) -> candle_core::Result<Tensor> {
    let sims = q.matmul(&d.t()?)?; // [B, B] cosine sims (rows are unit vectors)
    let logits = (sims / temperature)?;
    let b = q.dim(0)?;
    let targets = Tensor::arange(0u32, b as u32, q.device())?;
    let q_to_d = candle_nn::loss::cross_entropy(&logits, &targets)?;
    let d_to_q = candle_nn::loss::cross_entropy(&logits.t()?.contiguous()?, &targets)?;
    (q_to_d + d_to_q)? / 2.0
}

/// A fine-tuning session: the trainable encoder, its varmap, the tokenizer, and
/// the optimizer.
pub struct Trainer {
    cfg: TrainConfig,
    varmap: VarMap,
    encoder: Qwen3Encoder,
    tokenizer: Tokenizer,
    pad_id: u32,
    opt: candle_nn::AdamW,
    config: candle_transformers::models::qwen3::Config,
    prefix: &'static str,
}

impl Trainer {
    /// Load the stock weights into trainable vars and prepare to train.
    pub fn load(cfg: TrainConfig) -> Result<Self> {
        let dir = &cfg.weights_dir;
        let config: candle_transformers::models::qwen3::Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .map_err(|e| err(format!("read config.json: {e}")))?,
        )
        .map_err(|e| err(format!("parse config.json: {e}")))?;

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err(format!("load tokenizer.json: {e}")))?;
        let pad_id = tokenizer
            .token_to_id("<|endoftext|>")
            .unwrap_or(config.vocab_size as u32 - 1);

        // Match the checkpoint dtype so `VarMap::load` (the canonical path that
        // round-trips through `save`) fills the vars directly — a manual
        // build-then-cast-then-set path was silently breaking the save round-trip
        // (AdamW updated the model but `save` re-emitted the stock weights). The
        // checkpoint is bf16, which is also the GPU-native dtype.
        let dtype = DType::BF16;
        let weights = single_safetensors(dir)?;
        let prefix = prefix_from_safetensors(&weights)?;

        // Build the model on a fresh VarMap (creates the vars), then fill them from
        // the checkpoint by name. `detect_prefix` can't run on an empty varmap, so
        // the prefix is read from the file header above.
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &cfg.device);
        let encoder =
            Qwen3Encoder::load(&config, vb, prefix).map_err(|e| err(format!("build: {e}")))?;
        varmap
            .load(&weights)
            .map_err(|e| err(format!("load pretrained weights into vars: {e}")))?;

        let opt = candle_nn::AdamW::new(
            varmap.all_vars(),
            candle_nn::ParamsAdamW {
                lr: cfg.lr,
                ..Default::default()
            },
        )
        .map_err(|e| err(format!("optimizer: {e}")))?;

        Ok(Trainer {
            cfg,
            varmap,
            encoder,
            tokenizer,
            pad_id,
            opt,
            config,
            prefix,
        })
    }

    /// Diagnostic: rebuild the encoder from the varmap's CURRENT vars. If eval via
    /// this differs from eval via the live `self.encoder`, the two have diverged
    /// (the save source — the varmap — is not what training updated).
    pub fn rebuild_encoder_from_varmap(&mut self) -> Result<()> {
        let vb = VarBuilder::from_varmap(&self.varmap, DType::BF16, &self.cfg.device);
        self.encoder = Qwen3Encoder::load(&self.config, vb, self.prefix)
            .map_err(|e| err(format!("rebuild: {e}")))?;
        Ok(())
    }

    /// Encode a batch to `[B, H]` unit vectors — the differentiable forward.
    /// `is_query` prepends the instruction (queries only), matching inference.
    fn forward_batch(&self, texts: &[String], is_query: bool) -> candle_core::Result<Tensor> {
        let prepared: Vec<String> = if is_query {
            texts
                .iter()
                .map(|q| format!("Instruct: {DEFAULT_QUERY_TASK}\nQuery:{q}"))
                .collect()
        } else {
            texts.to_vec()
        };

        let encodings = self
            .tokenizer
            .encode_batch(prepared, true)
            .map_err(|e| candle_core::Error::Msg(format!("tokenize: {e}")))?;
        let mut ids: Vec<Vec<u32>> = encodings
            .iter()
            .map(|e| {
                let mut v = e.get_ids().to_vec();
                if v.len() > self.cfg.max_len {
                    v = v[v.len() - self.cfg.max_len..].to_vec(); // keep the tail (EOS)
                }
                v
            })
            .collect();

        let l = ids.iter().map(|v| v.len()).max().unwrap_or(1).max(1);
        let pad_lens: Vec<usize> = ids.iter().map(|v| l - v.len()).collect();
        for v in ids.iter_mut() {
            let pad = l - v.len();
            if pad > 0 {
                let mut padded = vec![self.pad_id; pad]; // LEFT pad
                padded.extend_from_slice(v);
                *v = padded;
            }
        }

        let b = ids.len();
        let input = Tensor::from_vec(ids.concat(), (b, l), self.encoder.device())?;
        let mask = self.encoder.left_pad_mask(&pad_lens, l)?;
        let hidden = self.encoder.forward(&input, Some(&mask))?;
        let pooled = self.encoder.pool_last(&hidden)?.to_dtype(DType::F32)?;
        // L2-normalize (differentiable), so InfoNCE similarities are cosines.
        let norm = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        pooled.broadcast_div(&norm)
    }

    /// One optimization step over a batch of pairs. Returns the batch loss.
    pub fn step(&mut self, batch: &[TrainPair]) -> Result<f32> {
        let queries: Vec<String> = batch.iter().map(|p| p.query.clone()).collect();
        let docs: Vec<String> = batch.iter().map(|p| p.positive.clone()).collect();
        let loss = (|| -> candle_core::Result<Tensor> {
            let q = self.forward_batch(&queries, true)?;
            let d = self.forward_batch(&docs, false)?;
            info_nce(&q, &d, self.cfg.temperature)
        })()
        .map_err(|e| err(format!("forward/loss: {e}")))?;
        self.opt
            .backward_step(&loss)
            .map_err(|e| err(format!("backward: {e}")))?;
        loss.to_scalar::<f32>().map_err(|e| err(format!("loss scalar: {e}")))
    }

    /// Train over `pairs` for `cfg.epochs`, invoking `on_step(step, loss)` after
    /// each batch (for logging). A deterministic shuffle (seeded, no `rand` dep)
    /// decorrelates adjacent batches without breaking reproducibility.
    pub fn train(
        &mut self,
        mut pairs: Vec<TrainPair>,
        mut on_step: impl FnMut(usize, f32),
    ) -> Result<()> {
        let batch = self.cfg.batch.max(1);
        let mut step = 0usize;
        for epoch in 0..self.cfg.epochs.max(1) {
            deterministic_shuffle(&mut pairs, 0xA11CE ^ epoch as u64);
            for chunk in pairs.chunks(batch) {
                if chunk.len() < 2 {
                    continue; // InfoNCE needs at least one negative
                }
                let loss = self.step(chunk)?;
                on_step(step, loss);
                step += 1;
            }
        }
        Ok(())
    }

    /// Embed texts to unit vectors on the GPU (no training). Detaches per batch so
    /// the autograd graph is not retained across thousands of documents.
    pub fn embed_texts(&self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.cfg.batch.max(1)) {
            let v = self
                .forward_batch(chunk, is_query)
                .and_then(|t| t.detach().to_vec2::<f32>())
                .map_err(|e| err(format!("embed: {e}")))?;
            out.extend(v);
        }
        Ok(out)
    }

    /// Save the fine-tuned checkpoint as a drop-in weights dir: the trained
    /// tensors plus the stock `config.json`/`tokenizer.json`, so
    /// `CandleQwenEmbedder::load(out_dir)` works unchanged.
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cfg.out_dir)
            .map_err(|e| err(format!("create out_dir: {e}")))?;

        // NOT `VarMap::save`: on CUDA it serialized the pre-training values (the
        // trained weights live on the GPU; its safetensors extraction did not read
        // them back). Copy each var explicitly to the CPU first, then serialize —
        // the copy forces the real current GPU values across.
        let tensors: std::collections::HashMap<String, Tensor> = {
            let vars = self.varmap.data().lock().expect("varmap lock");
            vars.iter()
                .map(|(name, var)| {
                    // `* 1.0` forces a kernel to read the LIVE GPU storage into a
                    // fresh tensor (a plain to_device / VarMap::save was extracting
                    // the pre-training values), then copy that to the CPU.
                    var.as_tensor()
                        .affine(1.0, 0.0)
                        .and_then(|t| t.to_device(&Device::Cpu))
                        .map(|t| (name.clone(), t))
                        .map_err(|e| err(format!("materialize {name}: {e}")))
                })
                .collect::<Result<_>>()?
        };
        candle_core::safetensors::save(&tensors, self.cfg.out_dir.join("model.safetensors"))
            .map_err(|e| err(format!("save weights: {e}")))?;

        for f in ["config.json", "tokenizer.json"] {
            std::fs::copy(self.cfg.weights_dir.join(f), self.cfg.out_dir.join(f))
                .map_err(|e| err(format!("copy {f}: {e}")))?;
        }
        Ok(())
    }
}

/// Evaluate a loaded model on nfcorpus **test**: embed the corpus + test queries
/// through the same GPU forward, brute-force cosine top-10, and mean nDCG@10 with
/// graded qrels. Using the trainer's own forward for BOTH stock and tuned makes
/// the comparison control for the forward — the only difference is the weights.
pub fn nfcorpus_ndcg(trainer: &Trainer, data_dir: &Path, max_docs: usize) -> Result<f64> {
    // Corpus: id -> (title. text).
    let corpus_raw = std::fs::read_to_string(data_dir.join("corpus.jsonl"))
        .map_err(|e| err(format!("read corpus: {e}")))?;
    let (mut doc_ids, mut doc_texts) = (Vec::new(), Vec::new());
    for line in corpus_raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| err(e.to_string()))?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let title = v["title"].as_str().unwrap_or_default();
        let text = v["text"].as_str().unwrap_or_default();
        doc_ids.push(id);
        doc_texts.push(if title.is_empty() {
            text.to_string()
        } else {
            format!("{title}. {text}")
        });
        if max_docs > 0 && doc_ids.len() >= max_docs {
            break;
        }
    }

    // qrels/test: query-id -> {doc-id -> grade>0}.
    let mut qrels: std::collections::HashMap<String, std::collections::HashMap<String, u8>> =
        std::collections::HashMap::new();
    let tsv = std::fs::read_to_string(data_dir.join("qrels/test.tsv"))
        .map_err(|e| err(format!("read qrels/test: {e}")))?;
    for (i, line) in tsv.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        let (Some(q), Some(d), Some(s)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        let g: u8 = s.trim().parse().unwrap_or(0);
        if g > 0 {
            qrels.entry(q.into()).or_default().insert(d.into(), g);
        }
    }

    // Queries with judgments only.
    let qraw = std::fs::read_to_string(data_dir.join("queries.jsonl"))
        .map_err(|e| err(format!("read queries: {e}")))?;
    let (mut q_texts, mut q_rels) = (Vec::new(), Vec::new());
    for line in qraw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| err(e.to_string()))?;
        let id = v["_id"].as_str().unwrap_or_default();
        if let Some(rels) = qrels.remove(id) {
            q_texts.push(v["text"].as_str().unwrap_or_default().to_string());
            q_rels.push(rels);
        }
    }

    let doc_vecs = trainer.embed_texts(&doc_texts, false)?;
    let q_vecs = trainer.embed_texts(&q_texts, true)?;

    let mut ndcg_sum = 0.0;
    for (qv, rels) in q_vecs.iter().zip(&q_rels) {
        let mut scored: Vec<(usize, f32)> = doc_vecs
            .iter()
            .enumerate()
            .map(|(i, dv)| (i, dot(qv, dv)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: Vec<&str> = scored.iter().take(10).map(|(i, _)| doc_ids[*i].as_str()).collect();
        ndcg_sum += ndcg_at_10(&top, rels);
    }
    Ok(ndcg_sum / q_rels.len().max(1) as f64)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn ndcg_at_10(ranked: &[&str], rels: &std::collections::HashMap<String, u8>) -> f64 {
    let dcg: f64 = ranked
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, id)| {
            let g = *rels.get(*id).unwrap_or(&0) as f64;
            (2f64.powf(g) - 1.0) / ((i + 2) as f64).log2()
        })
        .sum();
    let mut ideal: Vec<u8> = rels.values().copied().collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .take(10)
        .enumerate()
        .map(|(i, g)| (2f64.powf(*g as f64) - 1.0) / ((i + 2) as f64).log2())
        .sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// The single-file `model.safetensors` in `dir` (0.6B is unsharded). A sharded
/// checkpoint would need `VarMap::load` per shard — out of scope for the 0.6B.
fn single_safetensors(dir: &Path) -> Result<PathBuf> {
    let p = dir.join("model.safetensors");
    if p.exists() {
        Ok(p)
    } else {
        Err(err(format!(
            "no model.safetensors in {} (sharded checkpoints unsupported here)",
            dir.display()
        )))
    }
}

/// Read only the safetensors header (not the 2.4 GB of tensors) and return the
/// encoder prefix — `""` if tensors sit at the root, `"model"` if nested. Safe:
/// reads the 8-byte length then that many header bytes, no mmap.
fn prefix_from_safetensors(path: &Path) -> Result<&'static str> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| err(format!("open weights: {e}")))?;
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)
        .map_err(|e| err(format!("read header len: {e}")))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)
        .map_err(|e| err(format!("read header: {e}")))?;
    let json: serde_json::Value =
        serde_json::from_slice(&header).map_err(|e| err(format!("parse st header: {e}")))?;
    let has = |k: &str| json.get(k).is_some();
    if has("embed_tokens.weight") {
        Ok("")
    } else if has("model.embed_tokens.weight") {
        Ok("model")
    } else {
        Err(err(
            "safetensors has neither embed_tokens.weight nor model.embed_tokens.weight",
        ))
    }
}

/// A seeded Fisher–Yates shuffle (SplitMix64) — no `rand` dep, and reproducible.
fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_nce_rewards_aligned_pairs() {
        // Two pairs whose positives are perfectly aligned and mutually orthogonal
        // → the loss should be near zero (each query points at its own passage).
        let dev = Device::Cpu;
        let q = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &dev).unwrap();
        let d = q.clone();
        let aligned = info_nce(&q, &d, 0.05).unwrap().to_scalar::<f32>().unwrap();

        // Swap the passages → every query now points at the wrong one → high loss.
        let d_bad = Tensor::new(&[[0.0f32, 1.0], [1.0, 0.0]], &dev).unwrap();
        let misaligned = info_nce(&q, &d_bad, 0.05).unwrap().to_scalar::<f32>().unwrap();

        assert!(aligned < 0.01, "aligned pairs → ~0 loss, got {aligned}");
        assert!(
            misaligned > aligned + 1.0,
            "misaligned must cost much more: {misaligned} vs {aligned}"
        );
    }

    #[test]
    fn deterministic_shuffle_is_seeded_and_permutes() {
        let mut a: Vec<u32> = (0..100).collect();
        let mut b = a.clone();
        deterministic_shuffle(&mut a, 42);
        deterministic_shuffle(&mut b, 42);
        assert_eq!(a, b, "same seed → same permutation");
        assert_ne!(a, (0..100).collect::<Vec<_>>(), "it actually shuffles");
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>(), "a permutation, nothing lost");
    }
}
