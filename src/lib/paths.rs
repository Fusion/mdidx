use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub fn default_db_path() -> Result<PathBuf> {
    let app_dir = app_data_dir()?;
    let db_dir = app_dir.join("lancedb");
    ensure_dir(&db_dir)?;
    Ok(db_dir)
}

pub fn app_data_dir() -> Result<PathBuf> {
    let base = platform_data_dir()?;
    let app_dir = base.join("mdidx");
    ensure_dir(&app_dir)?;
    Ok(app_dir)
}

pub(crate) fn ensure_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .with_context(|| format!("create directory {}", path.display()))?;
    Ok(())
}

fn platform_data_dir() -> Result<PathBuf> {
    if cfg!(target_os = "windows") {
        if let Ok(dir) = env::var("APPDATA")
            && !dir.is_empty()
        {
            return Ok(PathBuf::from(dir));
        }
        if let Ok(dir) = env::var("LOCALAPPDATA")
            && !dir.is_empty()
        {
            return Ok(PathBuf::from(dir));
        }
        if let Ok(dir) = env::var("USERPROFILE")
            && !dir.is_empty()
        {
            return Ok(PathBuf::from(dir));
        }
        anyhow::bail!("unable to determine Windows data directory (APPDATA/LOCALAPPDATA)");
    }

    if cfg!(target_os = "macos") {
        let home = env::var("HOME").context("HOME is not set")?;
        return Ok(PathBuf::from(home).join("Library").join("Application Support"));
    }

    if let Ok(dir) = env::var("XDG_DATA_HOME")
        && !dir.is_empty()
    {
        return Ok(PathBuf::from(dir));
    }
    let home = env::var("HOME").context("HOME is not set")?;
    Ok(PathBuf::from(home).join(".local").join("share"))
}
