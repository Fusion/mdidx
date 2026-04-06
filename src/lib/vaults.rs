use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::paths::{app_data_dir, ensure_dir};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VaultConfig {
    pub vaults: HashMap<String, String>,
}

pub fn load_vault_config() -> Result<VaultConfig> {
    let path = vaults_path()?;
    if !path.exists() {
        return Ok(VaultConfig::default());
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("read vault config {}", path.display()))?;
    let config = serde_json::from_str(&content)
        .with_context(|| format!("parse vault config {}", path.display()))?;
    Ok(config)
}

pub fn save_vault_config(config: &VaultConfig) -> Result<()> {
    let path = vaults_path()?;
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    let payload = serde_json::to_string_pretty(config)?;
    fs::write(&path, payload)
        .with_context(|| format!("write vault config {}", path.display()))?;
    Ok(())
}

pub fn set_vault_path(id: &str, path: &Path) -> Result<()> {
    let mut config = load_vault_config()?;
    config.vaults.insert(id.to_string(), path.to_string_lossy().to_string());
    save_vault_config(&config)
}

pub fn get_vault_path(id: &str) -> Result<Option<PathBuf>> {
    let config = load_vault_config()?;
    Ok(config.vaults.get(id).map(PathBuf::from))
}

fn vaults_path() -> Result<PathBuf> {
    let base = app_data_dir()?;
    Ok(base.join("mdidx-vaults.json"))
}
