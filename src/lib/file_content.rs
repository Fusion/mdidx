use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContentParams {
    pub file_path: PathBuf,
    pub max_lines: usize,
    pub start_line: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContentResult {
    pub file_path: String,
    pub content: String,
    pub truncated: bool,
    pub start_line: usize,
    pub end_line: usize,
    pub total_lines: usize,
}

pub fn get_file_content(params: FileContentParams) -> Result<FileContentResult> {
    let content = fs::read_to_string(&params.file_path)
        .with_context(|| format!("read file {}", params.file_path.display()))?;
    let file_path = params.file_path.to_string_lossy().to_string();
    let lines: Vec<&str> = content.split('\n').collect();
    let total_lines = lines.len();

    let start_line = params.start_line.unwrap_or(1);
    if start_line == 0 {
        anyhow::bail!("start_line must be >= 1");
    }
    if start_line > total_lines {
        anyhow::bail!(
            "start_line {start_line} exceeds total lines {total_lines}"
        );
    }

    let max_lines = params.max_lines;
    let end_line = if max_lines == 0 {
        total_lines
    } else {
        (start_line + max_lines - 1).min(total_lines)
    };

    let output = lines[start_line - 1..end_line].join("\n");
    let truncated = end_line < total_lines;

    Ok(FileContentResult {
        file_path,
        content: output,
        truncated,
        start_line,
        end_line,
        total_lines,
    })
}
