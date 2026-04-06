use std::fs;

use anyhow::Result;
use sha2::{Digest, Sha256};

pub fn truncate_chars(value: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return value.to_string();
    }
    let mut count = 0usize;
    let mut end = value.len();
    for (idx, _) in value.char_indices() {
        if count == max_chars {
            end = idx;
            break;
        }
        count += 1;
    }
    if count < max_chars {
        return value.to_string();
    }
    format!("{}...", &value[..end])
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

pub fn modified_nanos(metadata: &fs::Metadata) -> Result<i64> {
    let modified = metadata.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    let duration = modified.duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap_or_default();
    Ok(duration.as_nanos() as i64)
}

pub fn now_epoch_seconds() -> i64 {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_secs() as i64
}

pub fn sql_escape(value: &str) -> String {
    value.replace('\'', "''")
}
