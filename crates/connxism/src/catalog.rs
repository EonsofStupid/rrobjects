//! Namespaces and databases above collections.
//!
//! SurrealDB scopes data as namespace → database → table; RRO's unit of full
//! isolation is the [`Estate`] (one physically separate store). So a [`Catalog`]
//! routes `(namespace, database)` to its own estate under `<root>/<ns>/<db>`, and
//! cross-namespace isolation is **by construction** — a query in one namespace
//! physically cannot reach another's store, not merely filtered from it.
//!
//! Collections stay *within* a database (the existing `collection` scope), so the
//! full hierarchy is namespace → database → collection.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rro_core::{Result, RroError};

use crate::estate::{Estate, EstateConfig};

/// A registry of namespaces and databases, each backed by its own [`Estate`].
pub struct Catalog {
    root: PathBuf,
    config: EstateConfig,
    open: Mutex<HashMap<(String, String), Arc<Estate>>>,
}

impl Catalog {
    /// Open (or create) a catalog rooted at `root` with default estate config.
    pub fn open(root: impl AsRef<Path>) -> Self {
        Self::open_with(root, EstateConfig::default())
    }

    /// Open a catalog with a shared estate configuration (applied to every
    /// database opened under it).
    pub fn open_with(root: impl AsRef<Path>, config: EstateConfig) -> Self {
        Catalog {
            root: root.as_ref().to_path_buf(),
            config,
            open: Mutex::new(HashMap::new()),
        }
    }

    /// Get (or open) the estate for `(namespace, database)`. Cached, so repeated
    /// calls return the same handle. Names must be simple identifiers — no path
    /// separators — so a namespace can never escape the catalog root.
    pub fn database(&self, namespace: &str, database: &str) -> Result<Arc<Estate>> {
        validate_name(namespace)?;
        validate_name(database)?;
        let key = (namespace.to_string(), database.to_string());
        let mut open = self.open.lock().expect("catalog lock");
        if let Some(estate) = open.get(&key) {
            return Ok(estate.clone());
        }
        let dir = self.root.join(namespace).join(database);
        std::fs::create_dir_all(&dir)
            .map_err(|e| RroError::msg(format!("create {}: {e}", dir.display())))?;
        let estate = Arc::new(Estate::open_with(&dir, database, self.config.clone())?);
        open.insert(key, estate.clone());
        Ok(estate)
    }

    /// The namespaces present under the catalog root (directory names).
    pub fn namespaces(&self) -> Result<Vec<String>> {
        list_dir(&self.root)
    }

    /// The databases within a namespace.
    pub fn databases(&self, namespace: &str) -> Result<Vec<String>> {
        validate_name(namespace)?;
        list_dir(&self.root.join(namespace))
    }

    /// Drop a database: close its estate and delete its store. All external
    /// handles to it must already be dropped, or the store's lock keeps the
    /// directory in use and the removal errors.
    pub fn drop_database(&self, namespace: &str, database: &str) -> Result<()> {
        validate_name(namespace)?;
        validate_name(database)?;
        // Release the catalog's own handle first so the estate can close.
        self.open
            .lock()
            .expect("catalog lock")
            .remove(&(namespace.to_string(), database.to_string()));
        let dir = self.root.join(namespace).join(database);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)
                .map_err(|e| RroError::msg(format!("drop {}: {e}", dir.display())))?;
        }
        rro_core::events::emit(
            "catalog.drop_database",
            serde_json::json!({ "namespace": namespace, "database": database }),
        );
        Ok(())
    }
}

/// Names are the on-disk path segments, so they must be plain identifiers —
/// rejecting `.`, `/`, `\` and `..` closes off any traversal out of the root.
fn validate_name(name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 128
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    if ok {
        Ok(())
    } else {
        Err(RroError::msg(format!(
            "invalid namespace/database name `{name}` (letters, digits, `_`, `-` only)"
        )))
    }
}

fn list_dir(path: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    if !path.exists() {
        return Ok(names);
    }
    for entry in std::fs::read_dir(path).map_err(|e| RroError::msg(format!("read_dir: {e}")))? {
        let entry = entry.map_err(|e| RroError::msg(format!("read_dir entry: {e}")))?;
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}
