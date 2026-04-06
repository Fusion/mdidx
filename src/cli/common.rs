use std::path::PathBuf;

use anyhow::Result;
use mdidx::default_db_path;

pub(crate) fn resolve_db_path(db: Option<PathBuf>) -> Result<PathBuf> {
    match db {
        Some(path) => Ok(path),
        None => default_db_path(),
    }
}
