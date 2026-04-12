use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use chrono::{NaiveDate, Utc};
use model_context_protocol::macros::mcp_tool;
use model_context_protocol::protocol::{LoggingLevel, McpCapabilities, McpToolDefinition};
use model_context_protocol::{McpServerConfig, tools};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;
use reqwest::Client;

use crate::{
    EmbeddingModelChoice, EmbeddingProviderChoice, FileContentParams, FileContentResult,
    FtsSearchParams, FtsSearchResults, HybridSearchParams, HybridSearchResults, IndexConfig,
    IndexSingleFileParams, IndexSingleFileResult, OpenAIModelChoice, SearchParams, SearchResults,
    StatsParams, StatsResults, app_data_dir, default_db_path, get_file_content as read_file_content,
    get_vault_path, index_single_file, load_ai_config, load_index_config, search, search_fts,
    search_hybrid, stats, MDIDX_VERSION,
};

static MCP_LOGGER: OnceLock<FileLogger> = OnceLock::new();

#[derive(Clone, Copy)]
enum LogSeverity {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
}

struct FileLogger {
    path: PathBuf,
    level: AtomicU8,
    lock: Mutex<()>,
}

impl FileLogger {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            level: AtomicU8::new(LogSeverity::Info as u8),
            lock: Mutex::new(()),
        }
    }

    fn set_level(&self, level: LoggingLevel) {
        let severity = log_severity(level);
        self.level.store(severity as u8, Ordering::Relaxed);
    }

    fn enabled(&self, level: LogSeverity) -> bool {
        level as u8 <= self.level.load(Ordering::Relaxed)
    }

    fn log(&self, level: LogSeverity, message: &str) {
        if !self.enabled(level) {
            return;
        }
        let _guard = self.lock.lock();
        let timestamp = Utc::now().to_rfc3339();
        let level_label = match level {
            LogSeverity::Error => "ERROR",
            LogSeverity::Warn => "WARN",
            LogSeverity::Info => "INFO",
            LogSeverity::Debug => "DEBUG",
        };
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&self.path) {
            let _ = writeln!(file, "{timestamp} [{level_label}] {message}");
        }
    }
}

fn log_severity(level: LoggingLevel) -> LogSeverity {
    match level {
        LoggingLevel::Error => LogSeverity::Error,
        LoggingLevel::Warning => LogSeverity::Warn,
        LoggingLevel::Info => LogSeverity::Info,
        LoggingLevel::Notice => LogSeverity::Info,
        LoggingLevel::Critical => LogSeverity::Error,
        LoggingLevel::Alert => LogSeverity::Error,
        LoggingLevel::Emergency => LogSeverity::Error,
        LoggingLevel::Debug => LogSeverity::Debug,
    }
}

pub fn init_mcp_logger() -> Result<PathBuf> {
    let path = app_data_dir()?.join("mcp.log");
    if MCP_LOGGER.get().is_none() {
        let _ = MCP_LOGGER.set(FileLogger::new(path.clone()));
    }
    Ok(path)
}

pub fn set_mcp_log_level(level: LoggingLevel) -> Result<()> {
    let _ = init_mcp_logger()?;
    if let Some(logger) = MCP_LOGGER.get() {
        logger.set_level(level);
    }
    Ok(())
}

fn log_mcp(level: LoggingLevel, message: impl AsRef<str>) {
    if MCP_LOGGER.get().is_none() {
        if init_mcp_logger().is_err() {
            return;
        }
    }
    if let Some(logger) = MCP_LOGGER.get() {
        logger.log(log_severity(level), message.as_ref());
    }
}

#[allow(clippy::too_many_arguments)]
#[mcp_tool("Semantic vector search over indexed markdown chunks (uses embeddings; returns nearest chunks)")]
async fn search_chunks(
    #[param("Query text (semantic search)")] query: String,
    #[param("LanceDB directory path (defaults to user data dir)")] db: Option<String>,
    #[param("Embedding provider: local or openai (ignored if db config exists)")] provider: Option<EmbeddingProviderChoice>,
    #[param("Local model: nomic or bge-m3 (ignored if db config exists)")] model: Option<EmbeddingModelChoice>,
    #[param("OpenAI model (ignored if db config exists)")] openai_model: Option<OpenAIModelChoice>,
    #[param("Vector dimension override (ignored if db config exists)")] dim: Option<i32>,
    #[param("Max results to return")] limit: Option<usize>,
    #[param("SQL filter for metadata columns (e.g. file_path = '...')")] filter: Option<String>,
    #[param("Apply filter after vector search (postfilter)")] postfilter: Option<bool>,
    #[param("Max content chars per result (0 = no limit)")] max_content_chars: Option<usize>,
    #[param("Show embedding model download progress")] show_download_progress: Option<bool>,
) -> Result<SearchResults, String> {
    let db_path = match db {
        Some(path) => PathBuf::from(path),
        None => default_db_path().map_err(|err| err.to_string())?,
    };
    log_mcp(
        LoggingLevel::Info,
        format!(
            "search_chunks start db={} limit={} filter={:?} postfilter={} max_content_chars={}",
            db_path.display(),
            limit.unwrap_or(5),
            filter.as_deref(),
            postfilter.unwrap_or(false),
            max_content_chars.unwrap_or(200)
        ),
    );
    let config = match load_index_config(&db_path).map_err(|err| err.to_string())? {
        Some(existing) => existing,
        None => {
            let provider = provider.unwrap_or(EmbeddingProviderChoice::Local);
            let model = model.unwrap_or(EmbeddingModelChoice::Nomic);
            let openai_model = openai_model.unwrap_or(OpenAIModelChoice::TextEmbedding3Small);
            let dim = dim.unwrap_or(match provider {
                EmbeddingProviderChoice::Local => match model {
                    EmbeddingModelChoice::Nomic => 768,
                    EmbeddingModelChoice::BgeM3 => 1024,
                },
                EmbeddingProviderChoice::OpenAI => match openai_model {
                    OpenAIModelChoice::TextEmbedding3Small => 1536,
                    OpenAIModelChoice::TextEmbedding3Large => 3072,
                },
            });
            IndexConfig {
                provider,
                model,
                openai_model,
                dim,
            }
        }
    };
    log_mcp(
        LoggingLevel::Debug,
        format!(
            "search_chunks config provider={:?} model={:?} openai_model={:?} dim={}",
            config.provider, config.model, config.openai_model, config.dim
        ),
    );
    let params = SearchParams {
        query,
        db: db_path,
        provider: config.provider,
        model: config.model,
        openai_model: config.openai_model,
        dim: Some(config.dim),
        show_download_progress: show_download_progress.unwrap_or(false),
        limit: limit.unwrap_or(5),
        filter,
        postfilter: postfilter.unwrap_or(false),
        max_content_chars: max_content_chars.unwrap_or(200),
    };

    match search(params).await {
        Ok(results) => {
            log_mcp(
                LoggingLevel::Info,
                format!("search_chunks ok results={}", results.results.len()),
            );
            Ok(results)
        }
        Err(err) => {
            log_mcp(
                LoggingLevel::Error,
                format!("search_chunks failed error={}", err),
            );
            Err(err.to_string())
        }
    }
}

#[mcp_tool("Lexical BM25 search over indexed chunks (requires FTS index; returns _score)")]
async fn search_chunks_bm25(
    #[param("Query text (lexical BM25)")] query: String,
    #[param("LanceDB directory path (defaults to user data dir)")] db: Option<String>,
    #[param("Max results to return")] limit: Option<usize>,
    #[param("SQL filter for metadata columns (e.g. file_path = '...')")] filter: Option<String>,
    #[param("Max content chars per result (0 = no limit)")] max_content_chars: Option<usize>,
) -> Result<FtsSearchResults, String> {
    let db_path = match db {
        Some(path) => PathBuf::from(path),
        None => default_db_path().map_err(|err| err.to_string())?,
    };
    log_mcp(
        LoggingLevel::Info,
        format!(
            "search_chunks_bm25 start db={} limit={} filter={:?} max_content_chars={}",
            db_path.display(),
            limit.unwrap_or(5),
            filter.as_deref(),
            max_content_chars.unwrap_or(200)
        ),
    );
    let params = FtsSearchParams {
        query,
        db: db_path,
        limit: limit.unwrap_or(5),
        filter,
        max_content_chars: max_content_chars.unwrap_or(200),
    };

    match search_fts(params).await {
        Ok(results) => {
            log_mcp(
                LoggingLevel::Info,
                format!("search_chunks_bm25 ok results={}", results.results.len()),
            );
            Ok(results)
        }
        Err(err) => {
            log_mcp(
                LoggingLevel::Error,
                format!("search_chunks_bm25 failed error={}", err),
            );
            Err(err.to_string())
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[mcp_tool("Hybrid search (vector + BM25) with RRF fusion; returns _distance, _score, _relevance_score")]
async fn search_chunks_hybrid(
    #[param("Query text (used for both vector + BM25)")] query: String,
    #[param("LanceDB directory path (defaults to user data dir)")] db: Option<String>,
    #[param("Embedding provider: local or openai (ignored if db config exists)")] provider: Option<EmbeddingProviderChoice>,
    #[param("Local model: nomic or bge-m3 (ignored if db config exists)")] model: Option<EmbeddingModelChoice>,
    #[param("OpenAI model (ignored if db config exists)")] openai_model: Option<OpenAIModelChoice>,
    #[param("Vector dimension override (ignored if db config exists)")] dim: Option<i32>,
    #[param("Max results to return")] limit: Option<usize>,
    #[param("SQL filter for metadata columns (e.g. file_path = '...')")] filter: Option<String>,
    #[param("Apply filter after vector search (postfilter)")] postfilter: Option<bool>,
    #[param("Max content chars per result (0 = no limit)")] max_content_chars: Option<usize>,
    #[param("Show embedding model download progress")] show_download_progress: Option<bool>,
) -> Result<HybridSearchResults, String> {
    let db_path = match db {
        Some(path) => PathBuf::from(path),
        None => default_db_path().map_err(|err| err.to_string())?,
    };
    log_mcp(
        LoggingLevel::Info,
        format!(
            "search_chunks_hybrid start db={} limit={} filter={:?} postfilter={} max_content_chars={}",
            db_path.display(),
            limit.unwrap_or(5),
            filter.as_deref(),
            postfilter.unwrap_or(false),
            max_content_chars.unwrap_or(200)
        ),
    );
    let config = match load_index_config(&db_path).map_err(|err| err.to_string())? {
        Some(existing) => existing,
        None => {
            let provider = provider.unwrap_or(EmbeddingProviderChoice::Local);
            let model = model.unwrap_or(EmbeddingModelChoice::Nomic);
            let openai_model = openai_model.unwrap_or(OpenAIModelChoice::TextEmbedding3Small);
            let dim = dim.unwrap_or(match provider {
                EmbeddingProviderChoice::Local => match model {
                    EmbeddingModelChoice::Nomic => 768,
                    EmbeddingModelChoice::BgeM3 => 1024,
                },
                EmbeddingProviderChoice::OpenAI => match openai_model {
                    OpenAIModelChoice::TextEmbedding3Small => 1536,
                    OpenAIModelChoice::TextEmbedding3Large => 3072,
                },
            });
            IndexConfig {
                provider,
                model,
                openai_model,
                dim,
            }
        }
    };
    log_mcp(
        LoggingLevel::Debug,
        format!(
            "search_chunks_hybrid config provider={:?} model={:?} openai_model={:?} dim={}",
            config.provider, config.model, config.openai_model, config.dim
        ),
    );
    let params = HybridSearchParams {
        query,
        db: db_path,
        provider: config.provider,
        model: config.model,
        openai_model: config.openai_model,
        dim: Some(config.dim),
        show_download_progress: show_download_progress.unwrap_or(false),
        limit: limit.unwrap_or(5),
        filter,
        postfilter: postfilter.unwrap_or(false),
        max_content_chars: max_content_chars.unwrap_or(200),
    };

    match search_hybrid(params).await {
        Ok(results) => {
            log_mcp(
                LoggingLevel::Info,
                format!("search_chunks_hybrid ok results={}", results.results.len()),
            );
            Ok(results)
        }
        Err(err) => {
            log_mcp(
                LoggingLevel::Error,
                format!("search_chunks_hybrid failed error={}", err),
            );
            Err(err.to_string())
        }
    }
}

#[mcp_tool("Get index stats (counts, last indexed time)")]
async fn stats_index(
    #[param("LanceDB directory path (defaults to user data dir)")] db: Option<String>,
) -> Result<StatsResults, String> {
    let db_path = match db {
        Some(path) => PathBuf::from(path),
        None => default_db_path().map_err(|err| err.to_string())?,
    };
    let params = StatsParams { db: db_path };

    stats(params).await.map_err(|err| err.to_string())
}

#[mcp_tool("Get mdidx version information")]
async fn version() -> Result<VersionInfo, String> {
    Ok(VersionInfo {
        name: "mdidx",
        version: MDIDX_VERSION,
    })
}

#[mcp_tool("Read file content from disk (use after search tools to fetch full context)")]
async fn get_file_content(
    #[param("File path to read (absolute or from search results)")] file_path: String,
    #[param("Max lines to return (0 = no limit)")] max_lines: Option<usize>,
    #[param("1-based start line (optional)")] start_line: Option<usize>,
) -> Result<FileContentResult, String> {
    let params = FileContentParams {
        file_path: PathBuf::from(file_path),
        max_lines: max_lines.unwrap_or(200),
        start_line,
    };

    read_file_content(params).map_err(|err| err.to_string())
}

#[mcp_tool("Upsert a synthesized note into a configured vault with strict frontmatter + optional indexing")]
async fn synth_note_upsert(
    #[param("Configured vault id (e.g., distilled)")] vault_id: String,
    #[param("Synthesized note payload")] note: SynthNote,
    #[param("Optional classification hints (category/domain/subpath)")] classification: Option<SynthClassificationHint>,
    #[param("Upsert and indexing options")] options: SynthUpsertOptions,
    #[param("Idempotency key for retries")] idempotency_key: Option<String>,
) -> Result<SynthNoteUpsertResponse, String> {
    /*
    Synthesis upsert pipeline.
    What: validate payload, resolve classification/path, build frontmatter, write markdown, and optionally index.
    Why: ensures deterministic note identity, safe idempotent retries, and consistent frontmatter semantics across tools.
    How: validate inputs, locate vault, resolve classification (hint/AI/rules), choose target path with conflict policy,
    merge frontmatter when requested, write the file, then trigger indexing based on scope.
    */
    let mut warnings = Vec::new();
    if let Some(key) = idempotency_key.as_ref()
        && (key.len() < 8 || key.len() > 128)
    {
        return Ok(SynthNoteUpsertResponse::error(
            vault_id,
            options.index_scope.clone(),
            "VALIDATION_ERROR",
            "idempotency_key must be 8-128 chars".to_string(),
        ));
    }
    if let Err(err) = validate_synth_note(&note) {
        return Ok(SynthNoteUpsertResponse::error(
            vault_id,
            options.index_scope.clone(),
            "VALIDATION_ERROR",
            err,
        ));
    }
    if let Err(err) = validate_upsert_options(&options) {
        return Ok(SynthNoteUpsertResponse::error(
            vault_id,
            options.index_scope.clone(),
            "VALIDATION_ERROR",
            err,
        ));
    }

    let vault_path = match get_vault_path(&vault_id).map_err(|err| err.to_string())? {
        Some(path) => path,
        None => {
            return Ok(SynthNoteUpsertResponse::error(
                vault_id,
                options.index_scope.clone(),
                "VAULT_NOT_FOUND",
                "vault_id not configured".to_string(),
            ));
        }
    };
    log_mcp(
        LoggingLevel::Info,
        format!(
            "synth_note_upsert start vault_id={} vault_path={} path_strategy={} upsert_mode={} refresh_index={} conflict_policy={} index_scope={} dry_run={}",
            vault_id,
            vault_path.display(),
            path_strategy_label(&options.path_strategy),
            upsert_mode_label(&options.upsert_key.mode),
            options.refresh_index,
            conflict_policy_label(&options.conflict_policy),
            index_scope_label(&options.index_scope),
            options.dry_run
        ),
    );

    let payload_hash = hash_payload(&note, &classification, &options, idempotency_key.as_deref());
    if let Some(key) = idempotency_key.as_ref() {
        match load_idempotency_entry(&vault_id, key) {
            Ok(Some(entry)) => {
                if entry.payload_hash == payload_hash {
                    if let Ok(resp) = serde_json::from_value::<SynthNoteUpsertResponse>(entry.response) {
                        log_mcp(
                            LoggingLevel::Info,
                            format!("synth_note_upsert idempotent hit vault_id={} key={}", vault_id, key),
                        );
                        return Ok(resp);
                    }
                    warnings.push("idempotency cache invalid; recomputing response".to_string());
                } else {
                    return Ok(SynthNoteUpsertResponse::error(
                        vault_id,
                        options.index_scope.clone(),
                        "CONFLICT",
                        "idempotency_key reused with different payload".to_string(),
                    ));
                }
            }
            Ok(None) => {}
            Err(err) => {
                warnings.push(format!("idempotency lookup failed: {err}"));
            }
        }
    }

    let resolved_classification = resolve_classification(&note, classification, &options, &mut warnings).await;
    if let Some(ref classification) = resolved_classification {
        log_mcp(
            LoggingLevel::Debug,
            format!(
                "synth_note_upsert classification vault_id={} category={} domain={:?} subpath={:?}",
                vault_id, classification.category, classification.domain, classification.subpath
            ),
        );
    }
    let target_dir = match build_target_dir(&vault_path, &note, &resolved_classification, &options) {
        Ok(value) => value,
        Err(err) => {
            log_mcp(
                LoggingLevel::Error,
                format!("synth_note_upsert build_target_dir failed vault_id={} error={}", vault_id, err),
            );
            return Ok(SynthNoteUpsertResponse::error(
                vault_id,
                options.index_scope.clone(),
                "VALIDATION_ERROR",
                err,
            ));
        }
    };

    let slug = match build_slug(&note, options.slug_override.as_deref()) {
        Ok(value) => value,
        Err(err) => {
            log_mcp(
                LoggingLevel::Error,
                format!("synth_note_upsert build_slug failed vault_id={} error={}", vault_id, err),
            );
            return Ok(SynthNoteUpsertResponse::error(
                vault_id,
                options.index_scope.clone(),
                "VALIDATION_ERROR",
                err,
            ));
        }
    };
    let filename = build_filename(&note, &slug, &options.filename_strategy);
    let target_path = target_dir.join(filename);
    log_mcp(
        LoggingLevel::Debug,
        format!(
            "synth_note_upsert target_dir={} slug={} target_path={}",
            target_dir.display(),
            slug,
            target_path.display()
        ),
    );

    let existing_path = match find_existing_path(&vault_path, &note, &options) {
        Ok(path) => path,
        Err(err) => {
            log_mcp(
                LoggingLevel::Error,
                format!("synth_note_upsert find_existing_path failed vault_id={} error={}", vault_id, err),
            );
            return Ok(SynthNoteUpsertResponse::error(
                vault_id,
                options.index_scope.clone(),
                "WRITE_FAILED",
                err,
            ));
        }
    };

    let mut final_path = if let Some(path) = existing_path.clone() {
        path
    } else {
        target_path.clone()
    };

    let mut operation = if final_path.exists() { "updated" } else { "created" };

    if operation == "updated" && options.conflict_policy == ConflictPolicy::AppendVersion {
        final_path = append_version_path(&final_path);
        operation = "created";
    }
    log_mcp(
        LoggingLevel::Info,
        format!(
            "synth_note_upsert write_decision operation={} final_path={} existed_before={}",
            operation,
            final_path.display(),
            final_path.exists()
        ),
    );

    let existing_frontmatter = if final_path.exists() {
        read_frontmatter(&final_path).unwrap_or_default()
    } else {
        Frontmatter::default()
    };

    let frontmatter = build_frontmatter(&note, &options, existing_frontmatter)?;
    let canonical_markdown = build_markdown(&frontmatter, &note.markdown_body)?;
    let fingerprint = format!("sha256:{}", sha256_hex(canonical_markdown.as_bytes()));

    if final_path.exists()
        && let Ok(existing) = fs::read_to_string(&final_path)
    {
        let existing_fingerprint = format!("sha256:{}", sha256_hex(existing.as_bytes()));
        if existing_fingerprint == fingerprint {
            operation = "noop";
        }
    }
    if operation == "noop" {
        log_mcp(
            LoggingLevel::Info,
            format!("synth_note_upsert noop vault_id={} path={}", vault_id, final_path.display()),
        );
    }

    if options.dry_run {
        warnings.push("dry_run enabled: no file written".to_string());
        log_mcp(
            LoggingLevel::Info,
            format!("synth_note_upsert dry_run vault_id={} path={}", vault_id, final_path.display()),
        );
    } else if operation != "noop" {
        if let Some(parent) = final_path.parent()
            && let Err(err) = fs::create_dir_all(parent)
        {
            log_mcp(
                LoggingLevel::Error,
                format!(
                    "synth_note_upsert create_dir failed vault_id={} path={} error={}",
                    vault_id,
                    parent.display(),
                    err
                ),
            );
            return Ok(SynthNoteUpsertResponse::error(
                vault_id,
                options.index_scope.clone(),
                "WRITE_FAILED",
                format!("create directory failed: {err}"),
            ));
        }
        if let Err(err) = fs::write(&final_path, canonical_markdown) {
            log_mcp(
                LoggingLevel::Error,
                format!(
                    "synth_note_upsert write failed vault_id={} path={} error={}",
                    vault_id,
                    final_path.display(),
                    err
                ),
            );
            return Ok(SynthNoteUpsertResponse::error(
                vault_id,
                options.index_scope.clone(),
                "WRITE_FAILED",
                format!("write failed: {err}"),
            ));
        }
        log_mcp(
            LoggingLevel::Info,
            format!("synth_note_upsert write ok vault_id={} path={}", vault_id, final_path.display()),
        );
    }

    let index_result = if options.refresh_index {
        log_mcp(
            LoggingLevel::Info,
            format!(
                "synth_note_upsert index refresh start vault_id={} scope={}",
                vault_id,
                index_scope_label(&options.index_scope)
            ),
        );
        match run_index_refresh(&final_path, &vault_path, options.index_scope.clone()).await {
            Ok(stats) => {
                log_mcp(
                    LoggingLevel::Info,
                    format!("synth_note_upsert index refresh ok vault_id={} path={}", vault_id, final_path.display()),
                );
                IndexResult {
                    requested: true,
                    executed: true,
                    scope: options.index_scope.clone(),
                    indexed_at: Some(Utc::now().to_rfc3339()),
                    stats: stats.map(|value| serde_json::json!(value)),
                }
            }
            Err(err) => {
                warnings.push(format!("index refresh failed: {err}"));
                log_mcp(
                    LoggingLevel::Error,
                    format!("synth_note_upsert index refresh failed vault_id={} error={}", vault_id, err),
                );
                IndexResult {
                    requested: true,
                    executed: false,
                    scope: options.index_scope.clone(),
                    indexed_at: None,
                    stats: None,
                }
            }
        }
    } else {
        IndexResult {
            requested: false,
            executed: false,
            scope: options.index_scope.clone(),
            indexed_at: None,
            stats: None,
        }
    };

    let response = SynthNoteUpsertResponse {
        ok: true,
        operation: operation.to_string(),
        vault_id: vault_id.clone(),
        final_path: path_to_relative_string(&vault_path, &final_path),
        note_id: frontmatter.note_id.clone(),
        fingerprint,
        resolved_classification: resolved_classification.map(|value| value.into_response()),
        frontmatter: Some(frontmatter.to_json()),
        index: index_result,
        warnings,
        error: None,
    };

    if let Some(key) = idempotency_key.as_ref() {
        let _ = save_idempotency_entry(&vault_id, key, payload_hash, &response);
    }

    Ok(response)
}

#[mcp_tool("Classify a synthesized note into vault categories/tags (no file writes)")]
async fn synth_classify_path(
    #[param("Configured vault id (e.g., distilled)")] vault_id: String,
    #[param("Note draft to classify")] note: ClassifyNote,
    #[param("Optional taxonomy context")] taxonomy_context: Option<TaxonomyContext>,
    #[param("Classification options")] options: ClassifyOptions,
    #[param("Idempotency key for retries")] idempotency_key: Option<String>,
) -> Result<SynthClassifyResponse, String> {
    let mut warnings = Vec::new();
    if let Some(key) = idempotency_key.as_ref()
        && (key.len() < 8 || key.len() > 128)
    {
        return Ok(SynthClassifyResponse::error(
            "VALIDATION_ERROR",
            "idempotency_key must be 8-128 chars".to_string(),
        ));
    }
    if let Err(err) = validate_classify_note(&note) {
        return Ok(SynthClassifyResponse::error("VALIDATION_ERROR", err));
    }

    if get_vault_path(&vault_id).map_err(|err| err.to_string())?.is_none() {
        return Ok(SynthClassifyResponse::error(
            "TAXONOMY_UNAVAILABLE",
            "vault_id not configured".to_string(),
        ));
    }

    let payload_hash = hash_payload_classify(&note, &taxonomy_context, &options, idempotency_key.as_deref());
    if let Some(key) = idempotency_key.as_ref()
        && let Ok(Some(entry)) = load_idempotency_entry(&vault_id, key)
    {
        if entry.payload_hash == payload_hash {
            if let Ok(resp) = serde_json::from_value::<SynthClassifyResponse>(entry.response) {
                return Ok(resp);
            }
        } else {
            return Ok(SynthClassifyResponse::error(
                "CONFLICT",
                "idempotency_key reused with different payload".to_string(),
            ));
        }
    }

    let classification = match options.mode.as_str() {
        "rule_only" => classify_note(&note, taxonomy_context.as_ref(), &options, &mut warnings),
        "ai_only" | "hybrid_ai_rules" => {
            match classify_note_ai(&note, taxonomy_context.as_ref(), &options).await {
                Ok(value) => value,
                Err(err) => {
                    if options.mode == "ai_only" {
                        return Ok(SynthClassifyResponse::error("CLASSIFICATION_FAILED", err.to_string()));
                    }
                    warnings.push(format!("AI classification failed: {err}; falling back to rules"));
                    classify_note(&note, taxonomy_context.as_ref(), &options, &mut warnings)
                }
            }
        }
        _ => {
            warnings.push("unknown mode; using rule_only".to_string());
            classify_note(&note, taxonomy_context.as_ref(), &options, &mut warnings)
        }
    };
    let response = SynthClassifyResponse {
        ok: true,
        classification: classification.classification,
        tags: classification.tags,
        scores: classification.scores,
        alternatives: classification.alternatives,
        rationale: classification.rationale,
        warnings,
        error: None,
    };

    if let Some(key) = idempotency_key.as_ref() {
        let _ = save_idempotency_entry(&vault_id, key, payload_hash, &response);
    }

    Ok(response)
}

pub fn build_config() -> McpServerConfig {
    build_config_with_capabilities().0
}

pub fn build_config_with_capabilities() -> (McpServerConfig, McpCapabilities) {
    let capabilities = McpCapabilities {
        tools: Some(serde_json::json!({})),
        logging: Some(serde_json::json!({})),
        ..Default::default()
    };

    let config = McpServerConfig::builder()
        .name("mdidx")
        .version(env!("CARGO_PKG_VERSION"))
        .with_capabilities(capabilities.clone())
        .with_tools(tools![
            SearchChunksTool,
            SearchChunksBm25Tool,
            SearchChunksHybridTool,
            StatsIndexTool,
            VersionTool,
            GetFileContentTool,
            SynthNoteUpsertTool,
            SynthClassifyPathTool
        ])
        .build();

    (config, capabilities)
}

pub fn add_tool_usage_hints(mut tools: Vec<McpToolDefinition>) -> Vec<McpToolDefinition> {
    let hints = tool_usage_hints();
    for tool in &mut tools {
        if let Some(hint) = hints.get(tool.name.as_str()) {
            tool.meta = Some(merge_meta(tool.meta.take(), "usageHint", Value::String((*hint).to_string())));
        }
    }
    tools
}

fn merge_meta(existing: Option<Value>, key: &str, value: Value) -> Value {
    match existing {
        Some(Value::Object(mut map)) => {
            map.insert(key.to_string(), value);
            Value::Object(map)
        }
        Some(other) => {
            let mut map = Map::new();
            map.insert("legacyMeta".to_string(), other);
            map.insert(key.to_string(), value);
            Value::Object(map)
        }
        None => {
            let mut map = Map::new();
            map.insert(key.to_string(), value);
            Value::Object(map)
        }
    }
}

fn tool_usage_hints() -> std::collections::HashMap<&'static str, &'static str> {
    use std::collections::HashMap;

    let mut hints = HashMap::new();
    hints.insert(
        "search_chunks",
        "Semantic vector search for meaning-based matches. Use this when synonyms or paraphrases should match. Returns _distance.",
    );
    hints.insert(
        "search_chunks_bm25",
        "Lexical BM25 search for exact terms/phrases. Requires FTS index on chunks.content. Returns _score.",
    );
    hints.insert(
        "search_chunks_hybrid",
        "Hybrid vector + BM25 search with RRF fusion. Use when you want both semantic + lexical signals. Returns _relevance_score plus _distance/_score.",
    );
    hints.insert(
        "stats_index",
        "Get counts and last indexed time to verify DB status and freshness.",
    );
    hints.insert(
        "get_file_content",
        "Read full file text from disk after search results. Use start_line/max_lines for targeted context.",
    );
    hints.insert(
        "version",
        "Return the mdidx build version baked into the binary.",
    );
    hints.insert(
        "synth_note_upsert",
        "Write/update a synthesized note into a configured vault with strict frontmatter, optional indexing, and idempotency. Required options: path_strategy (ai_suggested|fixed_path|inbox), refresh_index. upsert_key.mode defaults to hash; if frontmatter_key then provide frontmatter_key_name/value; if path then provide upsert_key.path.",
    );
    hints.insert(
        "synth_classify_path",
        "Classify a note into category/domain/subpath and normalized tags without writing files.",
    );
    hints
}

async fn classify_note_ai(
    note: &ClassifyNote,
    taxonomy_context: Option<&TaxonomyContext>,
    options: &ClassifyOptions,
) -> Result<ClassificationOutput> {
    /*
    AI classification is schema-constrained.
    What: call OpenAI with a JSON schema response format and validate the returned payload.
    Why: the downstream pipeline needs consistent categories/tags, so we enforce shape and guardrails.
    How: send note + taxonomy context, parse the JSON payload, then apply additional sanity checks.
    */
    let api_key = env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY is not set")?;
    let model = resolve_classify_model()?;

    let system = "You are a classification engine. Return JSON only, conforming to the provided schema. Do not include markdown. JSON must be the entire response.";
    let user = serde_json::json!({
        "note": note,
        "taxonomy_context": taxonomy_context,
        "options": options
    });

    let schema = serde_json::json!({
        "name": "synth_classify_path_response",
        "schema": {
            "type": "object",
            "additionalProperties": false,
            "required": ["classification", "tags", "scores"],
            "properties": {
                "classification": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["category", "domain", "subpath", "path_confidence", "resolved_path", "is_new_domain"],
                    "properties": {
                        "category": { "type": "string", "enum": ["00_Inbox", "10_Topics", "20_Playbooks", "30_Architecture", "90_Reference"] },
                        "domain": { "type": "string", "maxLength": 80 },
                        "subpath": { "type": "string", "maxLength": 180 },
                        "path_confidence": { "type": "number", "minimum": 0, "maximum": 1 },
                        "resolved_path": { "type": "string" },
                        "is_new_domain": { "type": "boolean" }
                    }
                },
                "tags": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["selected", "normalized", "dropped"],
                    "properties": {
                        "selected": {
                            "type": "array",
                            "items": { "type": "string", "pattern": "^[a-z0-9][a-z0-9_\\-/]{0,63}$" },
                            "minItems": 1,
                            "maxItems": 20,
                            "uniqueItems": true
                        },
                        "normalized": { "type": "array", "items": { "type": "string" }, "maxItems": 100 },
                        "dropped": { "type": "array", "items": { "type": "string" }, "maxItems": 100 }
                    }
                },
                "scores": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["category_scores", "domain_score", "tag_score"],
                    "properties": {
                        "category_scores": {
                            "type": "object",
                            "additionalProperties": { "type": "number", "minimum": 0, "maximum": 1 }
                        },
                        "domain_score": { "type": "number", "minimum": 0, "maximum": 1 },
                        "tag_score": { "type": "number", "minimum": 0, "maximum": 1 }
                    }
                },
                "alternatives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": false,
                        "required": ["category", "resolved_path", "score"],
                        "properties": {
                            "category": { "type": "string" },
                            "resolved_path": { "type": "string" },
                            "score": { "type": "number", "minimum": 0, "maximum": 1 }
                        }
                    },
                    "maxItems": 5
                },
                "rationale": {
                    "type": "array",
                    "items": { "type": "string", "maxLength": 500 },
                    "maxItems": 20
                }
            }
        }
    });

    let payload = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "system", "content": system },
            { "role": "user", "content": user.to_string() }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema
        }
    });

    let client = Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await
        .context("OpenAI request failed")?;
    let status = response.status();
    let body = response.text().await.context("OpenAI response read failed")?;
    if !status.is_success() {
        anyhow::bail!("OpenAI classification failed ({status}): {body}");
    }

    let parsed: serde_json::Value = serde_json::from_str(&body)
        .context("OpenAI response JSON parse failed")?;
    let content = parsed
        .get("choices")
        .and_then(|v| v.get(0))
        .and_then(|v| v.get("message"))
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
        .context("OpenAI response missing message content")?;
    let response_value: serde_json::Value = serde_json::from_str(content)
        .context("OpenAI content JSON parse failed")?;

    let classification: ClassificationOutput = serde_json::from_value(response_value)
        .context("OpenAI content does not match schema")?;

    validate_ai_classification(&classification)?;
    Ok(classification)
}

fn validate_ai_classification(value: &ClassificationOutput) -> Result<()> {
    if !matches!(
        value.classification.category.as_str(),
        "00_Inbox" | "10_Topics" | "20_Playbooks" | "30_Architecture" | "90_Reference"
    ) {
        anyhow::bail!("invalid category from AI");
    }
    if value.tags.selected.is_empty() {
        anyhow::bail!("AI returned empty tags");
    }
    Ok(())
}

fn resolve_classify_model() -> Result<String> {
    if let Ok(value) = env::var("MDIDX_CLASSIFY_MODEL")
        && !value.trim().is_empty()
    {
        return Ok(value);
    }
    let config = load_ai_config()?;
    if let Some(model) = config.classify_model
        && !model.trim().is_empty()
    {
        return Ok(model);
    }
    Ok("gpt-4o-mini".to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SynthNote {
    title: String,
    date: String,
    topic: String,
    status: String,
    tags: Vec<String>,
    source_sessions: Vec<String>,
    confidence: String,
    summary: Option<String>,
    key_decisions: Option<Vec<String>>,
    open_questions: Option<Vec<String>>,
    action_items: Option<Vec<String>>,
    related_topics: Option<Vec<String>>,
    markdown_body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SynthClassificationHint {
    category: Option<String>,
    domain: Option<String>,
    subpath: Option<String>,
    path_confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SynthUpsertOptions {
    path_strategy: PathStrategy,
    fixed_path: Option<String>,
    #[serde(default = "default_filename_strategy")]
    filename_strategy: FilenameStrategy,
    slug_override: Option<String>,
    #[serde(default)]
    upsert_key: UpsertKey,
    #[serde(default = "default_conflict_policy")]
    conflict_policy: ConflictPolicy,
    refresh_index: bool,
    #[serde(default = "default_index_scope")]
    index_scope: IndexScope,
    #[serde(default)]
    dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpsertKey {
    #[serde(default = "default_upsert_mode")]
    mode: UpsertMode,
    frontmatter_key_name: Option<String>,
    frontmatter_key_value: Option<String>,
    path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum UpsertMode {
    Hash,
    FrontmatterKey,
    Path,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PathStrategy {
    AiSuggested,
    FixedPath,
    Inbox,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum FilenameStrategy {
    DateTitle,
    TopicDate,
    SlugOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ConflictPolicy {
    Replace,
    AppendVersion,
    MergeFrontmatter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum IndexScope {
    ChangedFileOnly,
    VaultDelta,
    FullVault,
}

fn default_filename_strategy() -> FilenameStrategy {
    FilenameStrategy::DateTitle
}

fn default_conflict_policy() -> ConflictPolicy {
    ConflictPolicy::Replace
}

fn default_index_scope() -> IndexScope {
    IndexScope::VaultDelta
}

fn default_upsert_mode() -> UpsertMode {
    UpsertMode::Hash
}

impl Default for UpsertKey {
    fn default() -> Self {
        Self {
            mode: default_upsert_mode(),
            frontmatter_key_name: None,
            frontmatter_key_value: None,
            path: None,
        }
    }
}

fn default_max_tags() -> usize {
    8
}

fn default_min_tags() -> usize {
    3
}

fn default_allow_new_tags() -> bool {
    true
}

fn default_domain_strategy() -> String {
    "reuse_if_close".to_string()
}

fn default_subpath_depth_max() -> usize {
    3
}

fn default_alternatives_limit() -> usize {
    3
}

fn default_include_rationale() -> bool {
    true
}

fn path_strategy_label(value: &PathStrategy) -> &'static str {
    match value {
        PathStrategy::AiSuggested => "ai_suggested",
        PathStrategy::FixedPath => "fixed_path",
        PathStrategy::Inbox => "inbox",
    }
}

fn upsert_mode_label(value: &UpsertMode) -> &'static str {
    match value {
        UpsertMode::Hash => "hash",
        UpsertMode::FrontmatterKey => "frontmatter_key",
        UpsertMode::Path => "path",
    }
}

fn conflict_policy_label(value: &ConflictPolicy) -> &'static str {
    match value {
        ConflictPolicy::Replace => "replace",
        ConflictPolicy::AppendVersion => "append_version",
        ConflictPolicy::MergeFrontmatter => "merge_frontmatter",
    }
}

fn index_scope_label(value: &IndexScope) -> &'static str {
    match value {
        IndexScope::ChangedFileOnly => "changed_file_only",
        IndexScope::VaultDelta => "vault_delta",
        IndexScope::FullVault => "full_vault",
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassifyNote {
    title: String,
    topic: String,
    summary: Option<String>,
    status: Option<String>,
    confidence: Option<String>,
    existing_tags: Option<Vec<String>>,
    markdown_body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaxonomyContext {
    known_categories: Option<Vec<String>>,
    known_domains: Option<Vec<String>>,
    tag_dictionary: Option<Vec<String>>,
    recent_paths: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassifyOptions {
    mode: String,
    #[serde(default = "default_max_tags")]
    max_tags: usize,
    #[serde(default = "default_min_tags")]
    min_tags: usize,
    #[serde(default)]
    enforce_tag_dictionary: bool,
    #[serde(default = "default_allow_new_tags")]
    allow_new_tags: bool,
    #[serde(default = "default_domain_strategy")]
    domain_strategy: String,
    #[serde(default = "default_subpath_depth_max")]
    subpath_depth_max: usize,
    return_alternatives: bool,
    #[serde(default = "default_alternatives_limit")]
    alternatives_limit: usize,
    #[serde(default = "default_include_rationale")]
    include_rationale: bool,
    strict_category: Option<Vec<String>>,
    #[serde(default)]
    dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SynthNoteUpsertResponse {
    ok: bool,
    operation: String,
    vault_id: String,
    final_path: String,
    note_id: String,
    fingerprint: String,
    resolved_classification: Option<ResolvedClassification>,
    frontmatter: Option<Value>,
    index: IndexResult,
    warnings: Vec<String>,
    error: Option<ErrorDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResolvedClassification {
    category: String,
    domain: Option<String>,
    subpath: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexResult {
    requested: bool,
    executed: bool,
    scope: IndexScope,
    indexed_at: Option<String>,
    stats: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ErrorDetails {
    code: String,
    message: String,
    details: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SynthClassifyResponse {
    ok: bool,
    classification: ClassificationResult,
    tags: TagResult,
    scores: ScoreResult,
    alternatives: Option<Vec<ClassificationAlternative>>,
    rationale: Option<Vec<String>>,
    warnings: Vec<String>,
    error: Option<ErrorDetails>,
}

#[derive(Debug, Clone, Serialize)]
struct VersionInfo {
    name: &'static str,
    version: &'static str,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassificationResult {
    category: String,
    domain: String,
    subpath: String,
    path_confidence: f32,
    resolved_path: String,
    is_new_domain: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TagResult {
    selected: Vec<String>,
    normalized: Vec<String>,
    dropped: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScoreResult {
    category_scores: HashMap<String, f32>,
    domain_score: f32,
    tag_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassificationAlternative {
    category: String,
    resolved_path: String,
    score: f32,
}

#[derive(Debug, Clone)]
struct ResolvedClassificationInternal {
    category: String,
    domain: Option<String>,
    subpath: Option<String>,
    _path_confidence: f32,
}

#[derive(Default, Debug, Clone)]
struct Frontmatter {
    title: String,
    date: String,
    topic: String,
    status: String,
    tags: Vec<String>,
    source_sessions: Vec<String>,
    confidence: String,
    note_id: String,
    updated_at: String,
    summary: Option<String>,
    key_decisions: Vec<String>,
    open_questions: Vec<String>,
    action_items: Vec<String>,
    related_topics: Vec<String>,
}

impl Frontmatter {
    fn to_json(&self) -> Value {
        let mut map = Map::new();
        map.insert("title".to_string(), Value::String(self.title.clone()));
        map.insert("date".to_string(), Value::String(self.date.clone()));
        map.insert("topic".to_string(), Value::String(self.topic.clone()));
        map.insert("status".to_string(), Value::String(self.status.clone()));
        map.insert(
            "tags".to_string(),
            Value::Array(self.tags.iter().cloned().map(Value::String).collect()),
        );
        map.insert(
            "source_sessions".to_string(),
            Value::Array(
                self.source_sessions
                    .iter()
                    .cloned()
                    .map(Value::String)
                    .collect(),
            ),
        );
        map.insert(
            "confidence".to_string(),
            Value::String(self.confidence.clone()),
        );
        map.insert("note_id".to_string(), Value::String(self.note_id.clone()));
        map.insert(
            "updated_at".to_string(),
            Value::String(self.updated_at.clone()),
        );
        if let Some(summary) = &self.summary {
            map.insert("summary".to_string(), Value::String(summary.clone()));
        }
        if !self.key_decisions.is_empty() {
            map.insert(
                "key_decisions".to_string(),
                Value::Array(
                    self.key_decisions
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        if !self.open_questions.is_empty() {
            map.insert(
                "open_questions".to_string(),
                Value::Array(
                    self.open_questions
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        if !self.action_items.is_empty() {
            map.insert(
                "action_items".to_string(),
                Value::Array(
                    self.action_items
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        if !self.related_topics.is_empty() {
            map.insert(
                "related_topics".to_string(),
                Value::Array(
                    self.related_topics
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );
        }
        Value::Object(map)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IdempotencyEntry {
    payload_hash: String,
    response: Value,
}

impl SynthNoteUpsertResponse {
    fn error(vault_id: String, scope: IndexScope, code: &str, message: String) -> Self {
        Self {
            ok: false,
            operation: "error".to_string(),
            vault_id,
            final_path: "".to_string(),
            note_id: "".to_string(),
            fingerprint: "".to_string(),
            resolved_classification: None,
            frontmatter: None,
            index: IndexResult {
                requested: false,
                executed: false,
                scope,
                indexed_at: None,
                stats: None,
            },
            warnings: Vec::new(),
            error: Some(ErrorDetails {
                code: code.to_string(),
                message,
                details: None,
            }),
        }
    }
}

impl ResolvedClassificationInternal {
    fn into_response(self) -> ResolvedClassification {
        ResolvedClassification {
            category: self.category,
            domain: self.domain,
            subpath: self.subpath,
        }
    }
}

impl SynthClassifyResponse {
    fn error(code: &str, message: String) -> Self {
        Self {
            ok: false,
            classification: ClassificationResult {
                category: "00_Inbox".to_string(),
                domain: "".to_string(),
                subpath: "".to_string(),
                path_confidence: 0.0,
                resolved_path: "00_Inbox".to_string(),
                is_new_domain: false,
            },
            tags: TagResult {
                selected: Vec::new(),
                normalized: Vec::new(),
                dropped: Vec::new(),
            },
            scores: ScoreResult {
                category_scores: HashMap::new(),
                domain_score: 0.0,
                tag_score: 0.0,
            },
            alternatives: None,
            rationale: None,
            warnings: Vec::new(),
            error: Some(ErrorDetails {
                code: code.to_string(),
                message,
                details: None,
            }),
        }
    }
}

fn validate_synth_note(note: &SynthNote) -> Result<(), String> {
    if note.title.trim().len() < 3 || note.title.len() > 180 {
        return Err("title must be 3-180 chars".to_string());
    }
    if NaiveDate::parse_from_str(note.date.trim(), "%Y-%m-%d").is_err() {
        return Err("date must be YYYY-MM-DD".to_string());
    }
    if note.topic.trim().len() < 2 || note.topic.len() > 120 {
        return Err("topic must be 2-120 chars".to_string());
    }
    if !matches!(note.status.as_str(), "draft" | "stable") {
        return Err("status must be draft|stable".to_string());
    }
    if !matches!(note.confidence.as_str(), "low" | "med" | "high") {
        return Err("confidence must be low|med|high".to_string());
    }
    if note.tags.len() < 3 || note.tags.len() > 12 {
        return Err("tags must contain 3-12 items".to_string());
    }
    if !are_unique(&note.tags) {
        return Err("tags must be unique".to_string());
    }
    for tag in &note.tags {
        if !is_valid_tag(tag) {
            return Err(format!("invalid tag: {tag}"));
        }
    }
    if note.source_sessions.is_empty() || note.source_sessions.len() > 20 {
        return Err("source_sessions must contain 1-20 items".to_string());
    }
    if !are_unique(&note.source_sessions) {
        return Err("source_sessions must be unique".to_string());
    }
    if let Some(summary) = &note.summary
        && summary.len() > 1000
    {
        return Err("summary must be <= 1000 chars".to_string());
    }
    if let Some(items) = &note.key_decisions {
        if items.len() > 30 {
            return Err("key_decisions must be <= 30 items".to_string());
        }
        if items.iter().any(|item| item.len() > 500) {
            return Err("key_decisions item too long".to_string());
        }
    }
    if let Some(items) = &note.open_questions {
        if items.len() > 30 {
            return Err("open_questions must be <= 30 items".to_string());
        }
        if items.iter().any(|item| item.len() > 500) {
            return Err("open_questions item too long".to_string());
        }
    }
    if let Some(items) = &note.action_items {
        if items.len() > 50 {
            return Err("action_items must be <= 50 items".to_string());
        }
        if items.iter().any(|item| item.len() > 500) {
            return Err("action_items item too long".to_string());
        }
    }
    if let Some(items) = &note.related_topics {
        if items.len() > 20 {
            return Err("related_topics must be <= 20 items".to_string());
        }
        if !are_unique(items) {
            return Err("related_topics must be unique".to_string());
        }
    }
    if note.markdown_body.trim().len() < 20 {
        return Err("markdown_body must be at least 20 chars".to_string());
    }
    Ok(())
}

fn validate_upsert_options(options: &SynthUpsertOptions) -> Result<(), String> {
    if options.path_strategy == PathStrategy::FixedPath && options.fixed_path.is_none() {
        return Err("fixed_path required when path_strategy=fixed_path".to_string());
    }
    if options.upsert_key.mode == UpsertMode::FrontmatterKey
        && (options.upsert_key.frontmatter_key_name.as_deref().unwrap_or("").is_empty()
            || options.upsert_key.frontmatter_key_value.as_deref().unwrap_or("").is_empty())
    {
        return Err("frontmatter_key_name/value required for upsert_key.mode=frontmatter_key".to_string());
    }
    if options.upsert_key.mode == UpsertMode::Path
        && options.upsert_key.path.as_deref().unwrap_or("").is_empty()
    {
        return Err("upsert_key.path required for upsert_key.mode=path".to_string());
    }
    if let Some(slug) = &options.slug_override
        && !is_valid_slug(slug)
    {
        return Err("slug_override must be kebab-case".to_string());
    }
    Ok(())
}

fn validate_classify_note(note: &ClassifyNote) -> Result<(), String> {
    if note.title.trim().len() < 3 || note.title.len() > 180 {
        return Err("title must be 3-180 chars".to_string());
    }
    if note.topic.trim().len() < 2 || note.topic.len() > 120 {
        return Err("topic must be 2-120 chars".to_string());
    }
    if note.markdown_body.trim().len() < 20 {
        return Err("markdown_body must be at least 20 chars".to_string());
    }
    Ok(())
}

async fn resolve_classification(
    note: &SynthNote,
    hint: Option<SynthClassificationHint>,
    options: &SynthUpsertOptions,
    warnings: &mut Vec<String>,
) -> Option<ResolvedClassificationInternal> {
    /*
    Resolve classification for AI-suggested paths.
    Priority: explicit hint -> AI classification -> rule-based fallback.
    We emit warnings when the AI fails so callers can surface degraded behavior.
    */
    if options.path_strategy != PathStrategy::AiSuggested {
        return None;
    }
    if let Some(hint) = hint {
        let category = hint.category.unwrap_or_else(|| "00_Inbox".to_string());
        return Some(ResolvedClassificationInternal {
            category,
            domain: hint.domain,
            subpath: hint.subpath,
            _path_confidence: hint.path_confidence.unwrap_or(0.5),
        });
    }
    warnings.push("classification missing; using AI then rules fallback".to_string());
    let note_for_classification = ClassifyNote {
        title: note.title.clone(),
        topic: note.topic.clone(),
        summary: note.summary.clone(),
        status: Some(note.status.clone()),
        confidence: Some(note.confidence.clone()),
        existing_tags: Some(note.tags.clone()),
        markdown_body: note.markdown_body.clone(),
    };
    let options = ClassifyOptions {
        mode: "hybrid_ai_rules".to_string(),
        max_tags: 8,
        min_tags: 3,
        enforce_tag_dictionary: false,
        allow_new_tags: true,
        domain_strategy: "allow_new".to_string(),
        subpath_depth_max: 2,
        return_alternatives: false,
        alternatives_limit: 3,
        include_rationale: false,
        strict_category: None,
        dry_run: true,
    };
    let mut scratch = Vec::new();
    let classification = match classify_note_ai(&note_for_classification, None, &options).await {
        Ok(value) => value,
        Err(err) => {
            warnings.push(format!("AI classification failed: {err}; using rules"));
            classify_note(&note_for_classification, None, &options, &mut scratch)
        }
    };
    Some(ResolvedClassificationInternal {
        category: classification.classification.category.clone(),
        domain: Some(classification.classification.domain),
        subpath: Some(classification.classification.subpath),
        _path_confidence: classification.classification.path_confidence,
    })
}

fn build_target_dir(
    vault_root: &Path,
    note: &SynthNote,
    classification: &Option<ResolvedClassificationInternal>,
    options: &SynthUpsertOptions,
) -> Result<PathBuf, String> {
    match options.path_strategy {
        PathStrategy::Inbox => {
            let date = NaiveDate::parse_from_str(&note.date, "%Y-%m-%d")
                .map_err(|_| "invalid date for inbox path".to_string())?;
            let year = date.format("%Y").to_string();
            let day = date.format("%Y-%m-%d").to_string();
            Ok(vault_root.join("00_Inbox").join(year).join(day))
        }
        PathStrategy::FixedPath => {
            let fixed = options.fixed_path.as_ref().ok_or("fixed_path missing")?;
            let rel = sanitize_relative_path(fixed)?;
            Ok(vault_root.join(rel))
        }
        PathStrategy::AiSuggested => {
            let classification = classification
                .as_ref()
                .ok_or("classification required for ai_suggested path")?;
            let mut path = vault_root.join(&classification.category);
            if let Some(domain) = classification.domain.as_ref()
                && !domain.is_empty()
            {
                path = path.join(domain);
            }
            if let Some(subpath) = classification.subpath.as_ref()
                && !subpath.is_empty()
            {
                path = path.join(subpath);
            }
            Ok(path)
        }
    }
}

fn build_slug(note: &SynthNote, override_slug: Option<&str>) -> Result<String, String> {
    if let Some(override_slug) = override_slug {
        if !is_valid_slug(override_slug) {
            return Err("slug_override must be kebab-case".to_string());
        }
        return Ok(override_slug.to_string());
    }
    let slug = slugify(&note.title);
    if slug.is_empty() {
        return Ok("note".to_string());
    }
    Ok(slug)
}

fn build_filename(note: &SynthNote, slug: &str, strategy: &FilenameStrategy) -> String {
    match strategy {
        FilenameStrategy::DateTitle => format!("{}--{}.md", note.date, slug),
        FilenameStrategy::TopicDate => {
            let topic_slug = slugify(&note.topic);
            if topic_slug.is_empty() {
                format!("{}--{}.md", note.date, slug)
            } else {
                format!("{}--{}.md", topic_slug, note.date)
            }
        }
        FilenameStrategy::SlugOnly => format!("{slug}.md"),
    }
}

fn find_existing_path(
    vault_root: &Path,
    note: &SynthNote,
    options: &SynthUpsertOptions,
) -> Result<Option<PathBuf>, String> {
    match options.upsert_key.mode {
        UpsertMode::FrontmatterKey => {
            let key = options
                .upsert_key
                .frontmatter_key_name
                .as_ref()
                .ok_or("frontmatter_key_name missing")?;
            let value = options
                .upsert_key
                .frontmatter_key_value
                .as_ref()
                .ok_or("frontmatter_key_value missing")?;
            return Ok(find_by_frontmatter_key(vault_root, key, value));
        }
        UpsertMode::Path => {
            let path = options
                .upsert_key
                .path
                .as_ref()
                .ok_or("upsert_key.path missing")?;
            let rel = sanitize_relative_path(path)?;
            let full = vault_root.join(rel);
            if full.exists() {
                return Ok(Some(full));
            }
            return Ok(None);
        }
        UpsertMode::Hash => {}
    }

    let normalized = normalize_body(&note.markdown_body);
    let target_hash = sha256_hex(normalized.as_bytes());
    Ok(find_by_body_hash(vault_root, &target_hash))
}

fn append_version_path(path: &Path) -> PathBuf {
    let mut counter = 2u32;
    loop {
        let candidate = append_version_suffix(path, counter);
        if !candidate.exists() {
            return candidate;
        }
        counter += 1;
    }
}

fn append_version_suffix(path: &Path, version: u32) -> PathBuf {
    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("note.md");
    if let Some((base, _ext)) = file_name.rsplit_once(".md") {
        let new_name = format!("{base}--v{version}.md");
        return path.with_file_name(new_name);
    }
    let new_name = format!("{file_name}--v{version}");
    path.with_file_name(new_name)
}

fn read_frontmatter(path: &Path) -> Result<Frontmatter> {
    let content = fs::read_to_string(path)?;
    let (frontmatter, _) = split_frontmatter(&content)?;
    Ok(frontmatter)
}

fn build_frontmatter(
    note: &SynthNote,
    options: &SynthUpsertOptions,
    existing: Frontmatter,
) -> Result<Frontmatter, String> {
    /*
    Build frontmatter with deterministic IDs and optional merges.
    We preserve or synthesize note_id, update timestamps, and merge tags/sessions when requested
    to avoid losing metadata across repeated upserts.
    */
    let note_id = if options.upsert_key.mode == UpsertMode::FrontmatterKey {
        if options.upsert_key.frontmatter_key_name.as_deref() == Some("note_id") {
            options
                .upsert_key
                .frontmatter_key_value
                .clone()
                .unwrap_or(existing.note_id)
        } else {
            existing.note_id.clone()
        }
    } else {
        existing.note_id.clone()
    };

    let note_id = if note_id.is_empty() {
        deterministic_note_id(&note.title, &note.date, &note.topic)
    } else {
        note_id
    };

    let updated_at = Utc::now().to_rfc3339();

    let mut tags = note.tags.clone();
    let mut source_sessions = note.source_sessions.clone();

    if options.conflict_policy == ConflictPolicy::MergeFrontmatter {
        tags = union_strings(tags, existing.tags);
        source_sessions = union_strings(source_sessions, existing.source_sessions);
    }

    Ok(Frontmatter {
        title: note.title.clone(),
        date: note.date.clone(),
        topic: note.topic.clone(),
        status: note.status.clone(),
        tags,
        source_sessions,
        confidence: note.confidence.clone(),
        note_id,
        updated_at,
        summary: note.summary.clone(),
        key_decisions: note.key_decisions.clone().unwrap_or_default(),
        open_questions: note.open_questions.clone().unwrap_or_default(),
        action_items: note.action_items.clone().unwrap_or_default(),
        related_topics: note.related_topics.clone().unwrap_or_default(),
    })
}

fn build_markdown(frontmatter: &Frontmatter, body: &str) -> Result<String, String> {
    /*
    Serialize frontmatter as a stable YAML-like header followed by the markdown body.
    Values are quoted/escaped to keep round-trip parsing predictable.
    */
    let mut output = String::new();
    output.push_str("---\n");
    output.push_str(&format!("title: {}\n", yaml_quote(&frontmatter.title)));
    output.push_str(&format!("date: {}\n", yaml_quote(&frontmatter.date)));
    output.push_str(&format!("topic: {}\n", yaml_quote(&frontmatter.topic)));
    output.push_str(&format!("status: {}\n", yaml_quote(&frontmatter.status)));
    output.push_str("tags:\n");
    for tag in &frontmatter.tags {
        output.push_str(&format!("  - {}\n", yaml_quote(tag)));
    }
    output.push_str("source_sessions:\n");
    for session in &frontmatter.source_sessions {
        output.push_str(&format!("  - {}\n", yaml_quote(session)));
    }
    output.push_str(&format!(
        "confidence: {}\n",
        yaml_quote(&frontmatter.confidence)
    ));
    output.push_str(&format!("note_id: {}\n", yaml_quote(&frontmatter.note_id)));
    output.push_str(&format!(
        "updated_at: {}\n",
        yaml_quote(&frontmatter.updated_at)
    ));
    if let Some(summary) = &frontmatter.summary {
        output.push_str(&format!("summary: {}\n", yaml_quote(summary)));
    }
    write_list_field(&mut output, "key_decisions", &frontmatter.key_decisions);
    write_list_field(&mut output, "open_questions", &frontmatter.open_questions);
    write_list_field(&mut output, "action_items", &frontmatter.action_items);
    write_list_field(&mut output, "related_topics", &frontmatter.related_topics);
    output.push_str("---\n");
    output.push_str(body.trim());
    output.push('\n');
    Ok(output)
}

fn write_list_field(output: &mut String, key: &str, items: &[String]) {
    if items.is_empty() {
        return;
    }
    output.push_str(&format!("{key}:\n"));
    for item in items {
        output.push_str(&format!("  - {}\n", yaml_quote(item)));
    }
}

fn path_to_relative_string(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

async fn run_index_refresh(
    file_path: &Path,
    vault_root: &Path,
    scope: IndexScope,
) -> Result<Option<IndexSingleFileResult>> {
    match scope {
        IndexScope::ChangedFileOnly => {
            let db_path = default_db_path()?;
            let config = load_index_config(&db_path)?
                .context("mdidx-config.json not found; run mdidx index first")?;
            let params = IndexSingleFileParams {
                db: db_path,
                file_path: file_path.to_path_buf(),
                provider: config.provider,
                model: config.model,
                openai_model: config.openai_model,
                dim: config.dim,
                show_download_progress: false,
                chunk_size: 1000,
                chunk_overlap: 100,
                batch_size: 512,
            };
            let result = index_single_file(params).await?;
            Ok(Some(result))
        }
        IndexScope::VaultDelta | IndexScope::FullVault => {
            let db_path = default_db_path()?;
            let mdidx_bin = find_mdidx_bin().context("mdidx binary not found")?;
            let status = Command::new(mdidx_bin)
                .arg("index")
                .arg(vault_root)
                .arg("--db")
                .arg(db_path)
                .arg("--algorithm")
                .arg("mtime")
                .arg("--quiet")
                .status()
                .context("failed to execute mdidx index")?;
            if !status.success() {
                anyhow::bail!("mdidx index failed");
            }
            Ok(None)
        }
    }
}

fn find_mdidx_bin() -> Option<PathBuf> {
    if let Some(paths) = env::var_os("PATH") {
        for dir in env::split_paths(&paths) {
            let candidate = dir.join("mdidx");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    if let Ok(current) = std::env::current_exe()
        && let Some(dir) = current.parent()
    {
        let candidate = dir.join("mdidx");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn find_by_frontmatter_key(vault_root: &Path, key: &str, value: &str) -> Option<PathBuf> {
    for entry in WalkDir::new(vault_root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        if let Ok(content) = fs::read_to_string(entry.path())
            && frontmatter_matches_key_value(&content, key, value)
        {
            return Some(entry.path().to_path_buf());
        }
    }
    None
}

fn frontmatter_matches_key_value(content: &str, key: &str, value: &str) -> bool {
    if !content.starts_with("---\n") {
        return false;
    }
    let rest = &content[4..];
    let end = rest.find("\n---");
    let Some(end_idx) = end else {
        return false;
    };
    let frontmatter_text = &rest[..end_idx];
    let mut current_key: Option<&str> = None;
    for raw_line in frontmatter_text.lines() {
        let line = raw_line.trim_end();
        if line.starts_with("  - ") {
            if current_key == Some(key) {
                let item = strip_quotes(line.trim_start_matches("  - ").trim());
                if item == value {
                    return true;
                }
            }
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            let v = v.trim();
            current_key = Some(k);
            if k == key && !v.is_empty() && strip_quotes(v) == value {
                return true;
            }
            if !v.is_empty() {
                current_key = None;
            }
        }
    }
    false
}

fn find_by_body_hash(vault_root: &Path, target_hash: &str) -> Option<PathBuf> {
    for entry in WalkDir::new(vault_root).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        if let Ok(content) = fs::read_to_string(entry.path())
            && let Ok((_frontmatter, body)) = split_frontmatter(&content)
        {
            let normalized = normalize_body(body);
            let hash = sha256_hex(normalized.as_bytes());
            if hash == target_hash {
                return Some(entry.path().to_path_buf());
            }
        }
    }
    None
}

fn split_frontmatter(content: &str) -> Result<(Frontmatter, &str)> {
    /*
    Lightweight frontmatter splitter.
    We only parse the subset we emit; malformed or missing frontmatter falls back to defaults
    so edits do not block the upsert pipeline.
    */
    if !content.starts_with("---\n") {
        return Ok((Frontmatter::default(), content));
    }
    let rest = &content[4..];
    let end = rest.find("\n---");
    let Some(end_idx) = end else {
        return Ok((Frontmatter::default(), content));
    };
    let (frontmatter_text, remainder) = rest.split_at(end_idx);
    let body = remainder.trim_start_matches("\n---").trim_start_matches('\n');
    Ok((parse_frontmatter(frontmatter_text), body))
}

fn parse_frontmatter(frontmatter_text: &str) -> Frontmatter {
    let mut fm = Frontmatter::default();
    let mut current_list: Option<&str> = None;
    for raw_line in frontmatter_text.lines() {
        let line = raw_line.trim_end();
        if line.trim().is_empty() {
            continue;
        }
        if line.starts_with("  - ") {
            if let Some(list_key) = current_list {
                let item = line.trim_start_matches("  - ").trim();
                let item = strip_quotes(item);
                match list_key {
                    "tags" => fm.tags.push(item.to_string()),
                    "source_sessions" => fm.source_sessions.push(item.to_string()),
                    "key_decisions" => fm.key_decisions.push(item.to_string()),
                    "open_questions" => fm.open_questions.push(item.to_string()),
                    "action_items" => fm.action_items.push(item.to_string()),
                    "related_topics" => fm.related_topics.push(item.to_string()),
                    _ => {}
                }
            }
            continue;
        }
        current_list = None;
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim();
            let value = value.trim();
            if value.is_empty() {
                current_list = Some(key);
                continue;
            }
            let value = strip_quotes(value);
            match key {
                "title" => fm.title = value.to_string(),
                "date" => fm.date = value.to_string(),
                "topic" => fm.topic = value.to_string(),
                "status" => fm.status = value.to_string(),
                "confidence" => fm.confidence = value.to_string(),
                "note_id" => fm.note_id = value.to_string(),
                "updated_at" => fm.updated_at = value.to_string(),
                "summary" => fm.summary = Some(value.to_string()),
                _ => {}
            }
        }
    }
    fm
}

fn strip_quotes(value: &str) -> &str {
    let trimmed = value.trim();
    if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
        return &trimmed[1..trimmed.len() - 1];
    }
    trimmed
}

fn yaml_quote(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            _ => escaped.push(ch),
        }
    }
    format!("\"{}\"", escaped)
}

fn slugify(value: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in value.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_dash = false;
        } else if (c == ' ' || c == '-' || c == '_')
            && !prev_dash
        {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    while out.starts_with('-') {
        out.remove(0);
    }
    out
}

fn normalize_body(body: &str) -> String {
    body.trim().replace("\r\n", "\n")
}

fn is_valid_tag(tag: &str) -> bool {
    let bytes = tag.as_bytes();
    if bytes.is_empty() || bytes.len() > 64 {
        return false;
    }
    let first = bytes[0] as char;
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return false;
    }
    for ch in tag.chars() {
        if !(ch.is_ascii_lowercase()
            || ch.is_ascii_digit()
            || ch == '_'
            || ch == '-'
            || ch == '/')
        {
            return false;
        }
    }
    true
}

fn is_valid_slug(value: &str) -> bool {
    if value.is_empty() || value.len() > 120 {
        return false;
    }
    let first = value.chars().next().unwrap();
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return false;
    }
    value.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

fn sanitize_relative_path(path: &str) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return Err("fixed_path must be relative".to_string());
    }
    for component in candidate.components() {
        if let std::path::Component::ParentDir = component {
            return Err("fixed_path cannot contain ..".to_string());
        }
    }
    Ok(candidate)
}

fn are_unique(items: &[String]) -> bool {
    let mut seen = HashSet::new();
    for item in items {
        if !seen.insert(item) {
            return false;
        }
    }
    true
}

fn union_strings(mut left: Vec<String>, right: Vec<String>) -> Vec<String> {
    for item in right {
        if !left.contains(&item) {
            left.push(item);
        }
    }
    left
}

fn deterministic_note_id(title: &str, date: &str, topic: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(title.as_bytes());
    hasher.update(date.as_bytes());
    hasher.update(topic.as_bytes());
    let hash = hex::encode(hasher.finalize());
    format!("note-{}", &hash[..12])
}

fn hash_payload(
    note: &SynthNote,
    classification: &Option<SynthClassificationHint>,
    options: &SynthUpsertOptions,
    idempotency_key: Option<&str>,
) -> String {
    let value = serde_json::json!({
        "note": note,
        "classification": classification,
        "options": options,
        "idempotency_key": idempotency_key,
    });
    sha256_hex(value.to_string().as_bytes())
}

fn hash_payload_classify(
    note: &ClassifyNote,
    taxonomy_context: &Option<TaxonomyContext>,
    options: &ClassifyOptions,
    idempotency_key: Option<&str>,
) -> String {
    let value = serde_json::json!({
        "note": note,
        "taxonomy_context": taxonomy_context,
        "options": options,
        "idempotency_key": idempotency_key,
    });
    sha256_hex(value.to_string().as_bytes())
}

fn load_idempotency_entry(vault_id: &str, key: &str) -> Result<Option<IdempotencyEntry>> {
    let store = load_idempotency_store()?;
    let full_key = format!("{vault_id}:{key}");
    Ok(store.get(&full_key).cloned())
}

fn save_idempotency_entry<T: Serialize>(
    vault_id: &str,
    key: &str,
    payload_hash: String,
    response: &T,
) -> Result<()> {
    let mut store = load_idempotency_store()?;
    let full_key = format!("{vault_id}:{key}");
    let value = serde_json::to_value(response)?;
    store.insert(
        full_key,
        IdempotencyEntry {
            payload_hash,
            response: value,
        },
    );
    save_idempotency_store(&store)
}

fn load_idempotency_store() -> Result<HashMap<String, IdempotencyEntry>> {
    let path = idempotency_path()?;
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let content = fs::read_to_string(&path)?;
    let store = serde_json::from_str(&content)?;
    Ok(store)
}

fn save_idempotency_store(store: &HashMap<String, IdempotencyEntry>) -> Result<()> {
    let path = idempotency_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let payload = serde_json::to_string_pretty(store)?;
    fs::write(&path, payload)?;
    Ok(())
}

fn idempotency_path() -> Result<PathBuf> {
    Ok(app_data_dir()?.join("mdidx-idempotency.json"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassificationOutput {
    classification: ClassificationResult,
    tags: TagResult,
    scores: ScoreResult,
    alternatives: Option<Vec<ClassificationAlternative>>,
    rationale: Option<Vec<String>>,
}

fn classify_note(
    note: &ClassifyNote,
    taxonomy_context: Option<&TaxonomyContext>,
    options: &ClassifyOptions,
    warnings: &mut Vec<String>,
) -> ClassificationOutput {
    let text = format!(
        "{} {} {}",
        note.title,
        note.summary.clone().unwrap_or_default(),
        note.markdown_body
    )
    .to_lowercase();

    let mut scores: HashMap<String, f32> = [
        ("00_Inbox", 0.1),
        ("10_Topics", 0.1),
        ("20_Playbooks", 0.1),
        ("30_Architecture", 0.1),
        ("90_Reference", 0.1),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), *v))
    .collect();

    if text.contains("how to") || text.contains("steps") || text.contains("checklist") {
        *scores.get_mut("20_Playbooks").unwrap() += 0.6;
    }
    if text.contains("architecture") || text.contains("design") || text.contains("components") {
        *scores.get_mut("30_Architecture").unwrap() += 0.6;
    }
    if text.contains("reference") || text.contains("api") || text.contains("spec") {
        *scores.get_mut("90_Reference").unwrap() += 0.5;
    }
    if text.contains("overview") || text.contains("concept") || text.contains("introduction") {
        *scores.get_mut("10_Topics").unwrap() += 0.4;
    }
    if matches!(note.status.as_deref(), Some("draft"))
        || matches!(note.confidence.as_deref(), Some("low"))
    {
        *scores.get_mut("00_Inbox").unwrap() += 0.3;
    }

    let mut sorted: Vec<(String, f32)> = scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (mut category, best_score) = sorted[0].clone();
    let sum: f32 = scores.values().copied().sum();
    let mut confidence = if sum > 0.0 { best_score / sum } else { 0.0 };

    if confidence < 0.6 {
        category = "00_Inbox".to_string();
        confidence = 0.59;
        warnings.push("low classification confidence; routing to 00_Inbox".to_string());
    }

    if let Some(strict) = &options.strict_category
        && !strict.contains(&category)
    {
        category = strict
            .first()
            .cloned()
            .unwrap_or_else(|| "00_Inbox".to_string());
        warnings.push("strict_category enforced".to_string());
    }

    let (domain, is_new_domain) = resolve_domain(note, taxonomy_context, options);
    let subpath = slug_to_title(&slugify(&note.title));
    let resolved_path = if subpath.is_empty() {
        format!("{category}/{domain}")
    } else {
        format!("{category}/{domain}/{subpath}")
    };

    let (tags, tag_score) = resolve_tags(note, taxonomy_context, options);

    let alternatives = if options.return_alternatives {
        let mut alts = Vec::new();
        for (idx, (cat, score)) in sorted.iter().enumerate() {
            if idx == 0 {
                continue;
            }
            let path = format!("{cat}/{domain}");
            alts.push(ClassificationAlternative {
                category: cat.clone(),
                resolved_path: path,
                score: *score,
            });
            if alts.len() >= options.alternatives_limit {
                break;
            }
        }
        Some(alts)
    } else {
        None
    };

    let rationale = if options.include_rationale {
        Some(vec![
            format!("category chosen: {category}"),
            "scores derived from keyword heuristics".to_string(),
        ])
    } else {
        None
    };

    ClassificationOutput {
        classification: ClassificationResult {
            category,
            domain: domain.clone(),
            subpath,
            path_confidence: confidence,
            resolved_path,
            is_new_domain,
        },
        tags,
        scores: ScoreResult {
            category_scores: scores,
            domain_score: if is_new_domain { 0.6 } else { 0.85 },
            tag_score,
        },
        alternatives,
        rationale,
    }
}

fn resolve_domain(
    note: &ClassifyNote,
    taxonomy_context: Option<&TaxonomyContext>,
    options: &ClassifyOptions,
) -> (String, bool) {
    let topic = note.topic.trim();
    if let Some(ctx) = taxonomy_context
        && let Some(domains) = &ctx.known_domains
    {
        for domain in domains {
            if domain.eq_ignore_ascii_case(topic) {
                return (domain.clone(), false);
            }
            if topic.to_lowercase().contains(&domain.to_lowercase())
                || domain.to_lowercase().contains(&topic.to_lowercase())
            {
                return (domain.clone(), false);
            }
        }
        if options.domain_strategy == "prefer_existing" && !domains.is_empty() {
            return (domains[0].clone(), false);
        }
    }
    (slug_to_title(&slugify(topic)), true)
}

fn resolve_tags(
    note: &ClassifyNote,
    taxonomy_context: Option<&TaxonomyContext>,
    options: &ClassifyOptions,
) -> (TagResult, f32) {
    let mut normalized: Vec<String> = Vec::new();
    let mut selected: Vec<String> = Vec::new();
    let mut dropped: Vec<String> = Vec::new();

    let mut candidates: Vec<String> = note
        .existing_tags
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|tag| normalize_tag(&tag))
        .filter(|tag| !tag.is_empty())
        .collect();

    candidates.push(normalize_tag(&note.topic));

    for tag in &candidates {
        if normalized.contains(tag) {
            continue;
        }
        normalized.push(tag.clone());
    }

    let dictionary = taxonomy_context.and_then(|ctx| ctx.tag_dictionary.clone()).unwrap_or_default();
    let dict_set: HashSet<String> = dictionary
        .into_iter()
        .map(|tag| normalize_tag(&tag))
        .collect();

    for tag in normalized.iter() {
        if options.enforce_tag_dictionary && !dict_set.contains(tag) {
            dropped.push(tag.clone());
            continue;
        }
        if !options.allow_new_tags && !dict_set.contains(tag) {
            dropped.push(tag.clone());
            continue;
        }
        if !selected.contains(tag) && selected.len() < options.max_tags {
            selected.push(tag.clone());
        }
    }

    while selected.len() < options.min_tags && !normalized.is_empty() {
        let tag = normalized
            .iter()
            .find(|tag| !selected.contains(tag))
            .cloned();
        if let Some(tag) = tag {
            selected.push(tag);
        } else {
            break;
        }
    }

    if selected.is_empty() {
        selected.push("general".to_string());
    }

    let score = if selected.len() >= options.min_tags { 0.85 } else { 0.5 };

    (
        TagResult {
            selected,
            normalized,
            dropped,
        },
        score,
    )
}

fn normalize_tag(value: &str) -> String {
    let mut out = String::new();
    for ch in value.to_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if ch == ' ' || ch == '-' || ch == '_' || ch == '/' {
            out.push('-');
        }
    }
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    out.trim_matches('-').to_string()
}

fn slug_to_title(slug: &str) -> String {
    if slug.is_empty() {
        return "General".to_string();
    }
    slug.split('-')
        .filter(|s| !s.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => format!(
                    "{}{}",
                    first.to_uppercase().collect::<String>(),
                    chars.as_str()
                ),
                None => "".to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
