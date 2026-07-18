//! Training data: nfcorpus (query, positive passage) pairs from the TRAIN split.
//!
//! Contrastive fine-tuning needs (anchor, positive) pairs; the negatives come for
//! free as the other positives in a batch (in-batch negatives, InfoNCE). We build
//! pairs from the graded qrels: a query paired with each of its judged-relevant
//! passages. Only the **train** split is read here — the test split is the gate
//! and must never be seen in training.

use std::collections::HashMap;
use std::path::Path;

use rro_core::{Result, RroError};

/// One contrastive training example: a query and a passage judged relevant to it.
#[derive(Debug, Clone)]
pub struct TrainPair {
    /// The query text (an instruction prefix is added at encode time).
    pub query: String,
    /// A passage judged relevant (title + text, the BEIR convention).
    pub positive: String,
}

fn err(m: impl Into<String>) -> RroError {
    RroError::Embed(m.into())
}

/// Load `(query, positive)` pairs from an nfcorpus-layout directory using the
/// `qrels/train.tsv` judgments. `max_pairs = 0` means all.
///
/// A qrels row `(query-id, corpus-id, score)` with `score > 0` becomes one pair.
/// Rows whose query or passage text is missing are skipped (not faked).
pub fn load_pairs(dir: &Path, max_pairs: usize) -> Result<Vec<TrainPair>> {
    let corpus = load_jsonl_text(&dir.join("corpus.jsonl"), true)?;
    let queries = load_jsonl_text(&dir.join("queries.jsonl"), false)?;

    let tsv = std::fs::read_to_string(dir.join("qrels/train.tsv"))
        .map_err(|e| err(format!("read qrels/train.tsv: {e}")))?;
    let mut pairs = Vec::new();
    for (i, line) in tsv.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue; // header
        }
        let mut it = line.split('\t');
        let (Some(qid), Some(cid), Some(score)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        if score.trim().parse::<i64>().unwrap_or(0) <= 0 {
            continue; // only positives
        }
        let (Some(q), Some(p)) = (queries.get(qid), corpus.get(cid)) else {
            continue; // missing text — skip, don't fabricate
        };
        pairs.push(TrainPair {
            query: q.clone(),
            positive: p.clone(),
        });
        if max_pairs > 0 && pairs.len() >= max_pairs {
            break;
        }
    }
    if pairs.is_empty() {
        return Err(err(format!(
            "no training pairs built from {} — check corpus/queries/qrels",
            dir.display()
        )));
    }
    Ok(pairs)
}

/// Read a BEIR `*.jsonl` (`{_id, title?, text}`) into `id -> text`. For the corpus,
/// title is prepended (the BEIR convention the eval harness also uses, so training
/// and evaluation see the passage the same way).
fn load_jsonl_text(path: &Path, with_title: bool) -> Result<HashMap<String, String>> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| err(format!("read {}: {e}", path.display())))?;
    let mut out = HashMap::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| err(format!("parse jsonl: {e}")))?;
        let id = v["_id"].as_str().unwrap_or_default().to_string();
        let text = v["text"].as_str().unwrap_or_default();
        let full = match (with_title, v["title"].as_str()) {
            (true, Some(t)) if !t.is_empty() => format!("{t}. {text}"),
            _ => text.to_string(),
        };
        if !id.is_empty() {
            out.insert(id, full);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_real_nfcorpus_train_pairs_when_present() {
        // Runs only where the eval data is checked out (the box); skips otherwise
        // rather than failing, so CI without the data stays green.
        let dir = Path::new("../../eval-data/nfcorpus");
        if !dir.join("qrels/train.tsv").exists() {
            eprintln!("nfcorpus not present — skipping");
            return;
        }
        let pairs = load_pairs(dir, 50).unwrap();
        assert_eq!(pairs.len(), 50, "max_pairs caps the count");
        assert!(
            pairs.iter().all(|p| !p.query.is_empty() && !p.positive.is_empty()),
            "no empty query/positive text"
        );
    }
}
