//! The filesystem driver: a shared directory as a connector.
//!
//! Walks text-like files under a root; the cursor is the last-ingested path
//! (lexicographic), so a re-run resumes after it and newly added files (which
//! sort later or are re-listed) are picked up on the next pass. Metadata
//! carries `path`, `modified_at`, `bytes` — enough for RRD to shape it.

use async_trait::async_trait;

use rrf_core::{Document, Result, RrfError};

use crate::{Batch, Driver};

/// Directory-walking driver.
pub struct FsDriver {
    root: std::path::PathBuf,
    batch_size: usize,
}

impl FsDriver {
    /// A driver over `root`, pulling `batch_size` files per batch.
    pub fn new(root: impl Into<std::path::PathBuf>, batch_size: usize) -> Self {
        FsDriver {
            root: root.into(),
            batch_size: batch_size.max(1),
        }
    }

    /// All readable text files under the root, sorted by path.
    fn listing(&self) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();
        let mut stack = vec![self.root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).map_err(RrfError::Io)? {
                let path = entry.map_err(RrfError::Io)?.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    files.push(path);
                }
            }
        }
        files.sort();
        Ok(files)
    }
}

#[async_trait]
impl Driver for FsDriver {
    fn provider(&self) -> &str {
        "fs"
    }

    async fn pull(&self, cursor: Option<&str>) -> Result<Batch> {
        let files = self.listing()?;
        let mut docs = Vec::new();

        for path in files {
            let key = path.to_string_lossy().into_owned();
            if let Some(cur) = cursor {
                if key.as_str() <= cur {
                    continue; // already ingested
                }
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue; // non-utf8/binary: skipped (media driver later)
            };
            let meta = std::fs::metadata(&path).map_err(RrfError::Io)?;
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            let title = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| key.clone());
            let mut doc = Document::new(text).with_id(format!("fs:{key}"));
            doc.metadata
                .insert("title".into(), serde_json::json!(title));
            doc.metadata
                .insert("source_path".into(), serde_json::json!(key.clone()));
            doc.metadata
                .insert("modified_at".into(), serde_json::json!(modified));
            doc.metadata
                .insert("size_bytes".into(), serde_json::json!(meta.len()));
            docs.push(doc);

            if docs.len() >= self.batch_size {
                return Ok(Batch {
                    docs,
                    next_cursor: Some(key),
                });
            }
        }

        // Final (possibly empty) batch: drained.
        Ok(Batch {
            docs,
            next_cursor: None,
        })
    }
}
