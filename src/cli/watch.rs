use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use lancedb::table::Table;
use lancedb::connect;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use tokio::time::{sleep_until, Instant};

use mdidx::{
    EmbeddingModelChoice, EmbeddingProviderChoice, IndexConfig, OpenAIModelChoice, chunk_schema,
    file_schema, load_index_config, save_index_config,
};

use super::args::WatchArgs;
use super::common::resolve_db_path;
use super::fts::{FTS_COLUMN, ensure_fts_index, refresh_fts_index, wait_for_fts_tick};
use super::indexing::{
    Embedder, IndexFileParams, build_embedder, canonical_path_string, delete_file_entries,
    ensure_table, index_file, is_markdown_path, modified_nanos, resolve_dim, sha256_hex,
    verify_vector_dim,
};

struct WatchIndexContext<'a> {
    chunks_table: &'a Table,
    files_table: &'a Table,
    embedder: &'a mut Embedder,
    provider: EmbeddingProviderChoice,
    model: EmbeddingModelChoice,
    dim: i32,
    chunk_size: usize,
    chunk_overlap: usize,
    batch_size: usize,
    quiet: bool,
}

pub async fn watch_command(args: WatchArgs) -> Result<()> {
    /*
    Watch mode keeps the index in sync with filesystem events.
    What: listen for create/update/delete events, debounce them, and apply batch updates.
    Why: reduce redundant re-embeds during rapid edits while keeping the DB consistent.
    How: collect paths into a pending set, flush on a debounce timer, and optionally refresh FTS on a periodic tick.
    */
    if args.chunk_size == 0 {
        anyhow::bail!("chunk-size must be > 0");
    }
    if args.chunk_overlap >= args.chunk_size {
        anyhow::bail!("chunk-overlap must be < chunk-size");
    }
    if args.batch_size == 0 {
        anyhow::bail!("batch-size must be > 0");
    }

    let db_path = resolve_db_path(args.db)?;
    fs::create_dir_all(&db_path)
        .with_context(|| format!("create directory {}", db_path.display()))?;

    let existing = load_index_config(&db_path)?;
    let user_override = args.provider.is_some()
        || args.model.is_some()
        || args.openai_model.is_some()
        || args.dim.is_some();
    let provider = args
        .provider
        .or(existing.as_ref().map(|config| config.provider))
        .unwrap_or(EmbeddingProviderChoice::Local);
    let model = args
        .model
        .or(existing.as_ref().map(|config| config.model))
        .unwrap_or(EmbeddingModelChoice::Nomic);
    let openai_model = args
        .openai_model
        .or(existing.as_ref().map(|config| config.openai_model))
        .unwrap_or(OpenAIModelChoice::TextEmbedding3Small);
    let dim = if let Some(value) = args.dim {
        resolve_dim(provider, model, openai_model, Some(value))?
    } else if let Some(config) = existing.as_ref() {
        if user_override {
            resolve_dim(provider, model, openai_model, None)?
        } else {
            config.dim
        }
    } else {
        resolve_dim(provider, model, openai_model, None)?
    };
    let config = IndexConfig {
        provider,
        model,
        openai_model,
        dim,
    };

    if let Some(existing) = existing {
        if existing != config {
            anyhow::bail!(
                "Embedding settings already chosen for this database at {}. Changing them requires re-creating the database (use --reset --confirm) or choose a different --db. Existing: {:?}, requested: {:?}.",
                db_path.display(),
                existing,
                config
            );
        }
    } else {
        save_index_config(&db_path, &config)?;
    }

    let db_uri = db_path.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;
    let chunk_schema = chunk_schema(dim);
    let file_schema = file_schema();
    let chunks_table = ensure_table(&db, "chunks", chunk_schema.clone()).await?;
    let files_table = ensure_table(&db, "files", file_schema.clone()).await?;
    verify_vector_dim(&chunks_table, dim).await?;

    let mut embedder = build_embedder(
        provider,
        model,
        openai_model,
        args.show_download_progress,
        dim,
    )?;
    let mut watch_ctx = WatchIndexContext {
        chunks_table: &chunks_table,
        files_table: &files_table,
        embedder: &mut embedder,
        provider,
        model,
        dim,
        chunk_size: args.chunk_size,
        chunk_overlap: args.chunk_overlap,
        batch_size: args.batch_size,
        quiet: args.quiet,
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let mut watcher: RecommendedWatcher = notify::recommended_watcher(move |res| {
        match res {
            Ok(event) => {
                let _ = tx.send(event);
            }
            Err(err) => {
                eprintln!("watch error: {err}");
            }
        }
    })?;

    for path in &args.paths {
        watcher.watch(path, RecursiveMode::Recursive)?;
        if !args.quiet {
            println!("watch {}", path.display());
        }
    }

    let debounce = Duration::from_millis(args.debounce_ms);
    let mut fts_interval = if args.fts_refresh_interval_secs == 0 {
        None
    } else {
        let mut interval = tokio::time::interval(Duration::from_secs(args.fts_refresh_interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        Some(interval)
    };
    let mut fts_index_name: Option<String> = None;
    let mut pending_fts_refresh = false;
    let mut pending: HashSet<PathBuf> = HashSet::new();
    let mut last_event: Option<Instant> = None;

    loop {
        if last_event.is_some() {
            let deadline = last_event.unwrap() + debounce;
            tokio::select! {
                Some(event) = rx.recv() => {
                    for path in event.paths {
                        pending.insert(path);
                    }
                    if debounce.is_zero() {
                        process_pending_changes(&mut pending, &mut watch_ctx).await?;
                        last_event = None;
                    } else {
                        last_event = Some(Instant::now());
                    }
                }
                _ = sleep_until(deadline) => {
                    if !pending.is_empty() {
                        process_pending_changes(&mut pending, &mut watch_ctx).await?;
                    }
                    if pending_fts_refresh {
                        let index_name = ensure_watch_fts_index(
                            &chunks_table,
                            &mut fts_index_name,
                            args.quiet,
                        )
                        .await?;
                        refresh_fts_index(&chunks_table, &index_name, args.quiet).await?;
                        pending_fts_refresh = false;
                    }
                    last_event = None;
                }
                _ = wait_for_fts_tick(&mut fts_interval) => {
                    if last_event.is_some() || !pending.is_empty() {
                        pending_fts_refresh = true;
                    } else {
                        let index_name = ensure_watch_fts_index(
                            &chunks_table,
                            &mut fts_index_name,
                            args.quiet,
                        )
                        .await?;
                        refresh_fts_index(&chunks_table, &index_name, args.quiet).await?;
                    }
                }
            }
        } else {
            tokio::select! {
                event = rx.recv() => {
                    match event {
                        Some(event) => {
                            for path in event.paths {
                                pending.insert(path);
                            }
                            if debounce.is_zero() {
                                process_pending_changes(&mut pending, &mut watch_ctx).await?;
                                if pending_fts_refresh {
                                    let index_name = ensure_watch_fts_index(
                                        &chunks_table,
                                        &mut fts_index_name,
                                        args.quiet,
                                    )
                                    .await?;
                                    refresh_fts_index(&chunks_table, &index_name, args.quiet).await?;
                                    pending_fts_refresh = false;
                                }
                                last_event = None;
                            } else {
                                last_event = Some(Instant::now());
                            }
                        }
                        None => break,
                    }
                }
                _ = wait_for_fts_tick(&mut fts_interval) => {
                    if last_event.is_some() || !pending.is_empty() {
                        pending_fts_refresh = true;
                    } else {
                        let index_name = ensure_watch_fts_index(
                            &chunks_table,
                            &mut fts_index_name,
                            args.quiet,
                        )
                        .await?;
                        refresh_fts_index(&chunks_table, &index_name, args.quiet).await?;
                    }
                }
            }
        }
    }

    Ok(())
}

async fn process_pending_changes(
    pending: &mut HashSet<PathBuf>,
    context: &mut WatchIndexContext<'_>,
) -> Result<()> {
    /*
    Apply a batch of pending paths.
    Files that exist are reindexed in place; missing paths are treated as deletions and removed
    from both chunks and files tables. Sorting improves determinism and makes logs easier to scan.
    */
    if pending.is_empty() {
        return Ok(());
    }

    let mut paths: Vec<PathBuf> = pending.drain().collect();
    paths.sort();

    let chunks_table = context.chunks_table;
    let files_table = context.files_table;
    let provider = context.provider;
    let model = context.model;
    let dim = context.dim;
    let chunk_size = context.chunk_size;
    let chunk_overlap = context.chunk_overlap;
    let batch_size = context.batch_size;
    let quiet = context.quiet;

    for path in paths {
        if !is_markdown_path(&path) {
            continue;
        }

        if path.exists() && path.is_file() {
            let canonical_path = canonical_path_string(&path)?;
            let metadata = fs::metadata(&path).with_context(|| format!("metadata: {canonical_path}"))?;
            let mtime = modified_nanos(&metadata)?;
            let content = fs::read_to_string(&path)
                .with_context(|| format!("read: {canonical_path}"))?;
            let checksum = sha256_hex(content.as_bytes());
            if !quiet {
                println!("index {}", canonical_path);
            }
            let index_params = IndexFileParams {
                chunks_table,
                files_table,
                embedder: &mut *context.embedder,
                provider,
                model,
                file_path: &canonical_path,
                content: &content,
                checksum,
                mtime,
                dim,
                chunk_size,
                chunk_overlap,
                batch_size,
            };
            index_file(index_params).await?;
        } else {
            let canonical_path = canonical_path_string(&path)?;
            delete_file_entries(chunks_table, files_table, &canonical_path).await?;
            if !quiet {
                println!("remove {}", canonical_path);
            }
        }
    }

    Ok(())
}

async fn ensure_watch_fts_index(
    table: &Table,
    index_name: &mut Option<String>,
    quiet: bool,
) -> Result<String> {
    if index_name.is_none() {
        *index_name = Some(ensure_fts_index(table, FTS_COLUMN, quiet).await?);
    }
    Ok(index_name
        .as_ref()
        .expect("index name missing")
        .to_string())
}
