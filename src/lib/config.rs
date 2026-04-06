use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::embedding::{EmbeddingModelChoice, EmbeddingProviderChoice, OpenAIModelChoice};
use crate::paths::ensure_dir;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexConfig {
    pub provider: EmbeddingProviderChoice,
    pub model: EmbeddingModelChoice,
    pub openai_model: OpenAIModelChoice,
    pub dim: i32,
}

pub fn load_index_config(db_path: &Path) -> Result<Option<IndexConfig>> {
    let path = config_path(db_path);
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("read config {}", path.display()))?;
    let config = serde_json::from_str(&content)
        .with_context(|| format!("parse config {}", path.display()))?;
    Ok(Some(config))
}

pub fn save_index_config(db_path: &Path, config: &IndexConfig) -> Result<()> {
    ensure_dir(db_path)?;
    let path = config_path(db_path);
    let payload = serde_json::to_string_pretty(config)?;
    fs::write(&path, payload)
        .with_context(|| format!("write config {}", path.display()))?;
    Ok(())
}

fn config_path(db_path: &Path) -> PathBuf {
    db_path.join("mdidx-config.json")
}
