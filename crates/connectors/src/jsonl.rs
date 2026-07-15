//! The JSONL-feed driver: a newline-delimited JSON file as a connector.
//!
//! Each line is one payload: `{"id": ..., "text": ..., ...metadata}` (missing
//! ids are derived from the line number). The cursor is the line offset —
//! exactly the shape exports, event dumps, and DuckDB extracts arrive in.

use async_trait::async_trait;

use rrf_core::{Document, Result, RrfError};

use crate::{Batch, Driver};

/// Line-oriented JSON feed driver.
pub struct JsonlDriver {
    path: std::path::PathBuf,
    batch_size: usize,
}

impl JsonlDriver {
    /// A driver over the feed file at `path`.
    pub fn new(path: impl Into<std::path::PathBuf>, batch_size: usize) -> Self {
        JsonlDriver {
            path: path.into(),
            batch_size: batch_size.max(1),
        }
    }
}

#[async_trait]
impl Driver for JsonlDriver {
    fn provider(&self) -> &str {
        "jsonl"
    }

    async fn pull(&self, cursor: Option<&str>) -> Result<Batch> {
        let start: usize = cursor.and_then(|c| c.parse().ok()).unwrap_or(0);
        let content = std::fs::read_to_string(&self.path).map_err(RrfError::Io)?;

        let mut docs = Vec::new();
        let mut line_no = 0usize;
        for line in content.lines() {
            line_no += 1;
            if line_no <= start || line.trim().is_empty() {
                continue;
            }
            let mut value: serde_json::Value = serde_json::from_str(line)?;
            let obj = value
                .as_object_mut()
                .ok_or_else(|| RrfError::msg(format!("line {line_no}: not a JSON object")))?;

            let id = obj
                .remove("id")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_else(|| format!("line{line_no}"));
            let text = obj
                .remove("text")
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();

            let mut doc = Document::new(text).with_id(format!("jsonl:{id}"));
            doc.metadata = obj.clone().into_iter().collect();
            docs.push(doc);

            if docs.len() >= self.batch_size {
                return Ok(Batch {
                    docs,
                    next_cursor: Some(line_no.to_string()),
                });
            }
        }

        Ok(Batch {
            docs,
            next_cursor: None,
        })
    }
}
