use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::{app_data_dir, ensure_dir};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AiConfig {
    pub classify_model: Option<String>,
}

pub fn load_ai_config() -> Result<AiConfig> {
    let path = ai_config_path()?;
    if !path.exists() {
        return Ok(AiConfig::default());
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("read ai config {}", path.display()))?;
    let config = serde_json::from_str(&content)
        .with_context(|| format!("parse ai config {}", path.display()))?;
    Ok(config)
}

pub fn save_ai_config(config: &AiConfig) -> Result<()> {
    let path = ai_config_path()?;
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    let payload = serde_json::to_string_pretty(config)?;
    fs::write(&path, payload)
        .with_context(|| format!("write ai config {}", path.display()))?;
    Ok(())
}

pub fn set_classify_model(model: Option<String>) -> Result<()> {
    let mut config = load_ai_config()?;
    config.classify_model = model;
    save_ai_config(&config)
}

fn ai_config_path() -> Result<PathBuf> {
    let base = app_data_dir()?;
    Ok(base.join("mdidx-ai.json"))
}
