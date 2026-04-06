use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use arrow_array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use lancedb::table::Table;
use lancedb::{connect, Error as LanceError};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use mdidx::{
    ChunkRow, EmbeddingModelChoice, EmbeddingProviderChoice, IndexConfig, OpenAIModelChoice,
    app_data_dir, build_chunk_batch, chunk_schema, file_schema, load_index_config, save_index_config,
};

use super::args::{ChangeAlgorithm, IndexArgs};
use super::common::resolve_db_path;
use super::fts::{FTS_COLUMN, ensure_fts_index, refresh_fts_index};

#[derive(Debug, Clone)]
struct FileEntry {
    checksum: String,
    mtime: i64,
}

pub(crate) enum Embedder {
    Local {
        inner: Box<TextEmbedding>,
    },
    OpenAI(OpenAIClient),
}

pub(crate) struct OpenAIClient {
    http: reqwest::Client,
    api_key: String,
    organization: Option<String>,
    project: Option<String>,
    model: OpenAIModelChoice,
    dimensions: Option<i32>,
}

#[derive(serde::Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<i32>,
    encoding_format: &'a str,
}

#[derive(serde::Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

#[derive(serde::Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

pub async fn index_command(args: IndexArgs) -> Result<()> {
    /*
    Indexing pipeline.
    What: scan one or more roots, update chunks/files tables, and optionally refresh FTS.
    Why: keep the index incrementally consistent while locking embedding config per database to avoid mixed vectors.
    How: resolve or load config, walk roots, reindex changed files (mtime/checksum), delete missing files, then refresh FTS if requested.
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

    if let Some(existing) = existing
        && existing != config
    {
        if args.reset {
            // Overwrite after reset.
        } else {
            anyhow::bail!(
                "Embedding settings already chosen for this database at {}. Changing them requires re-creating the database (use --reset --confirm) or choose a different --db. Existing: {:?}, requested: {:?}.",
                db_path.display(),
                existing,
                config
            );
        }
    }

    let db_uri = db_path.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;

    if !args.quiet {
        let roots = args
            .paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!("Scanning {} for .md files...", roots);
    }

    if args.reset && !args.confirm {
        anyhow::bail!("refusing to reset without --confirm");
    }
    if args.reset {
        reset_tables(&db).await?;
    }

    save_index_config(&db_path, &config)?;

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

    let existing_files = load_file_index(&files_table).await?;
    let mut seen_paths: HashSet<String> = HashSet::new();

    let mut indexed_files = 0usize;
    let mut skipped_files = 0usize;
    let mut removed_files = 0usize;
    let mut chunk_rows = 0usize;

    for root in &args.paths {
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            if entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("md"))
                != Some(true)
            {
                continue;
            }

            let path = entry.path();
            let canonical_path = canonical_path_string(path)?;
            let canonical_path_display = canonical_path.clone();
            let metadata = fs::metadata(path).with_context(|| format!("metadata: {canonical_path}"))?;
            let mtime = modified_nanos(&metadata)?;

            let existing = existing_files.get(&canonical_path);

            if args.algorithm == ChangeAlgorithm::Mtime {
                if !should_reindex(ChangeAlgorithm::Mtime, existing, mtime, None)? {
                    if !args.quiet {
                        println!("skip (mtime unchanged) {}", canonical_path_display);
                    }
                    seen_paths.insert(canonical_path);
                    skipped_files += 1;
                    continue;
                }

                let content = fs::read_to_string(path)
                    .with_context(|| format!("read: {canonical_path}"))?;
                let checksum = sha256_hex(content.as_bytes());
                if !args.quiet {
                    println!("index {}", canonical_path_display);
                }
                let index_params = IndexFileParams {
                    chunks_table: &chunks_table,
                    files_table: &files_table,
                    embedder: &mut embedder,
                    provider,
                    model,
                    file_path: &canonical_path,
                    content: &content,
                    checksum,
                    mtime,
                    dim,
                    chunk_size: args.chunk_size,
                    chunk_overlap: args.chunk_overlap,
                    batch_size: args.batch_size,
                };
                index_file(index_params).await?;
                indexed_files += 1;
                chunk_rows += count_chunks(&content, args.chunk_size, args.chunk_overlap);
                seen_paths.insert(canonical_path);
            } else {
                let content = fs::read_to_string(path)
                    .with_context(|| format!("read: {canonical_path}"))?;
                let checksum = sha256_hex(content.as_bytes());
                if !should_reindex(ChangeAlgorithm::Checksum, existing, mtime, Some(&checksum))? {
                    if !args.quiet {
                        println!("skip (checksum unchanged) {}", canonical_path_display);
                    }
                    seen_paths.insert(canonical_path);
                    skipped_files += 1;
                    continue;
                }

                if !args.quiet {
                    println!("index {}", canonical_path_display);
                }
                let index_params = IndexFileParams {
                    chunks_table: &chunks_table,
                    files_table: &files_table,
                    embedder: &mut embedder,
                    provider,
                    model,
                    file_path: &canonical_path,
                    content: &content,
                    checksum,
                    mtime,
                    dim,
                    chunk_size: args.chunk_size,
                    chunk_overlap: args.chunk_overlap,
                    batch_size: args.batch_size,
                };
                index_file(index_params).await?;
                indexed_files += 1;
                chunk_rows += count_chunks(&content, args.chunk_size, args.chunk_overlap);
                seen_paths.insert(canonical_path);
            }
        }
    }

    for (path, _) in existing_files.iter() {
        if !seen_paths.contains(path) {
            delete_file_entries(&chunks_table, &files_table, path).await?;
            removed_files += 1;
            if !args.quiet {
                println!("remove {}", path);
            }
        }
    }

    if args.fts_refresh {
        let index_name = ensure_fts_index(&chunks_table, FTS_COLUMN, args.quiet).await?;
        refresh_fts_index(&chunks_table, &index_name, args.quiet).await?;
    }

    println!(
        "Indexed {indexed_files} file(s) ({chunk_rows} chunk rows), skipped {skipped_files}, removed {removed_files}."
    );

    Ok(())
}

pub(crate) async fn index_file(
    params: IndexFileParams<'_>,
) -> Result<()> {
    /*
    File-level upsert uses "delete then insert" rather than partial updates.
    This guarantees we never leave stale chunks behind after a file shrinks and keeps the files
    table authoritative for incremental scans. Chunk rows are written in batches to control
    memory usage and API limits.
    */
    let IndexFileParams {
        chunks_table,
        files_table,
        embedder,
        provider,
        model,
        file_path,
        content,
        checksum,
        mtime,
        dim,
        chunk_size,
        chunk_overlap,
        batch_size,
    } = params;
    delete_file_entries(chunks_table, files_table, file_path).await?;

    let chunks = chunk_text(content, chunk_size, chunk_overlap);
    let chunk_count = chunks.len() as i32;
    if chunks.is_empty() {
        let file_batch = build_file_batch(file_path, &checksum, mtime, chunk_count)?;
        files_table.add(file_batch).execute().await?;
        return Ok(());
    }
    let embedding_inputs = prepare_document_inputs(provider, model, &chunks);
    let embeddings = embed_texts(embedder, embedding_inputs, batch_size)
        .await
        .context("embedding failed")?;

    if embeddings.len() != chunks.len() {
        anyhow::bail!(
            "embedding count mismatch: expected {}, got {}",
            chunks.len(),
            embeddings.len()
        );
    }

    let mut rows: Vec<ChunkRow> = Vec::with_capacity(batch_size.min(chunks.len()));
    for (index, (chunk, vector)) in chunks.into_iter().zip(embeddings.into_iter()).enumerate() {
        rows.push(ChunkRow {
            file_path: file_path.to_string(),
            chunk_index: index as i32,
            content: chunk,
            checksum: checksum.clone(),
            mtime,
            vector,
        });

        if rows.len() >= batch_size {
            add_chunk_rows(chunks_table, &rows, dim).await?;
            rows.clear();
        }
    }

    if !rows.is_empty() {
        add_chunk_rows(chunks_table, &rows, dim).await?;
    }

    let file_batch = build_file_batch(file_path, &checksum, mtime, chunk_count)?;
    files_table.add(file_batch).execute().await?;

    Ok(())
}

pub(crate) async fn ensure_table(
    db: &lancedb::connection::Connection,
    name: &str,
    schema: Arc<Schema>,
) -> Result<Table> {
    match db.open_table(name).execute().await {
        Ok(table) => Ok(table),
        Err(err) => match err {
            LanceError::TableNotFound { .. } => Ok(db.create_empty_table(name, schema).execute().await?),
            other => Err(other.into()),
        },
    }
}

pub(crate) async fn delete_file_entries(
    chunks_table: &Table,
    files_table: &Table,
    file_path: &str,
) -> Result<()> {
    let predicate = format!("file_path = '{}'", sql_escape(file_path));
    chunks_table.delete(&predicate).await?;
    files_table.delete(&predicate).await?;
    Ok(())
}

pub(crate) async fn verify_vector_dim(table: &Table, expected_dim: i32) -> Result<()> {
    let schema = table.schema().await?;
    let field = schema.field_with_name("vector")?;
    if let DataType::FixedSizeList(_, size) = field.data_type() {
        if *size != expected_dim {
            anyhow::bail!(
                "vector dimension mismatch: table has {size}, but --dim {expected_dim} was requested"
            );
        }
    } else {
        anyhow::bail!("vector column is not FixedSizeList");
    }
    Ok(())
}

pub(crate) fn canonical_path_string(path: &Path) -> Result<String> {
    let canonical = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    Ok(canonical.to_string_lossy().to_string())
}

pub(crate) fn modified_nanos(metadata: &fs::Metadata) -> Result<i64> {
    let modified = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let duration = modified.duration_since(UNIX_EPOCH).unwrap_or_default();
    Ok(duration.as_nanos() as i64)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

pub(crate) fn is_markdown_path(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("md"))
        == Some(true)
}

async fn load_file_index(table: &Table) -> Result<HashMap<String, FileEntry>> {
    /*
    Preload file metadata into memory so change detection is O(1) per file during the scan.
    This avoids repeated table lookups in the tight loop over filesystem entries.
    */
    let mut map = HashMap::new();
    let stream = table.query().execute().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    for batch in batches {
        let file_col = batch
            .column_by_name("file_path")
            .context("files table missing file_path column")?;
        let checksum_col = batch
            .column_by_name("checksum")
            .context("files table missing checksum column")?;
        let mtime_col = batch
            .column_by_name("mtime")
            .context("files table missing mtime column")?;

        let file_arr = file_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("file_path is not StringArray")?;
        let checksum_arr = checksum_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("checksum is not StringArray")?;
        let mtime_arr = mtime_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("mtime is not Int64Array")?;

        for i in 0..batch.num_rows() {
            let path = file_arr.value(i).to_string();
            let checksum = checksum_arr.value(i).to_string();
            let mtime = mtime_arr.value(i);
            map.insert(path, FileEntry { checksum, mtime });
        }
    }

    Ok(map)
}

async fn add_chunk_rows(table: &Table, rows: &[ChunkRow], dim: i32) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let batch = build_chunk_batch(rows, dim)?;
    table.add(batch).execute().await?;
    Ok(())
}

pub(crate) struct IndexFileParams<'a> {
    pub(crate) chunks_table: &'a Table,
    pub(crate) files_table: &'a Table,
    pub(crate) embedder: &'a mut Embedder,
    pub(crate) provider: EmbeddingProviderChoice,
    pub(crate) model: EmbeddingModelChoice,
    pub(crate) file_path: &'a str,
    pub(crate) content: &'a str,
    pub(crate) checksum: String,
    pub(crate) mtime: i64,
    pub(crate) dim: i32,
    pub(crate) chunk_size: usize,
    pub(crate) chunk_overlap: usize,
    pub(crate) batch_size: usize,
}

fn build_file_batch(file_path: &str, checksum: &str, mtime: i64, chunk_count: i32) -> Result<RecordBatch> {
    let schema = file_schema();
    let file_path_array = StringArray::from_iter_values([file_path]);
    let checksum_array = StringArray::from_iter_values([checksum]);
    let mtime_array = Int64Array::from_iter_values([mtime]);
    let chunk_count_array = Int32Array::from_iter_values([chunk_count]);
    let indexed_at_array = Int64Array::from_iter_values([now_epoch_seconds()]);

    let columns = vec![
        Arc::new(file_path_array) as _ ,
        Arc::new(checksum_array) as _ ,
        Arc::new(mtime_array) as _ ,
        Arc::new(chunk_count_array) as _ ,
        Arc::new(indexed_at_array) as _ ,
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    /*
    Chunk using character indices to avoid slicing UTF-8 mid-codepoint.
    Overlap is applied in character space to preserve continuity across adjacent chunks.
    */
    if text.is_empty() {
        return Vec::new();
    }

    let mut indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    indices.push(text.len());
    let char_count = indices.len().saturating_sub(1);

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < char_count {
        let end = (start + chunk_size).min(char_count);
        let chunk = &text[indices[start]..indices[end]];
        chunks.push(chunk.to_string());
        if end == char_count {
            break;
        }
        start = end.saturating_sub(overlap);
    }

    chunks
}

fn count_chunks(text: &str, chunk_size: usize, overlap: usize) -> usize {
    if text.is_empty() {
        return 0;
    }

    let mut indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    indices.push(text.len());
    let char_count = indices.len().saturating_sub(1);

    let mut count = 0usize;
    let mut start = 0usize;

    while start < char_count {
        let end = (start + chunk_size).min(char_count);
        count += 1;
        if end == char_count {
            break;
        }
        start = end.saturating_sub(overlap);
    }

    count
}

fn sql_escape(value: &str) -> String {
    value.replace('\'', "''")
}

fn now_epoch_seconds() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_secs() as i64
}

async fn reset_tables(db: &lancedb::connection::Connection) -> Result<()> {
    let names = db.table_names().execute().await?;
    if names.iter().any(|name| name == "chunks") {
        db.drop_table("chunks", &[]).await?;
    }
    if names.iter().any(|name| name == "files") {
        db.drop_table("files", &[]).await?;
    }
    Ok(())
}

pub(crate) fn resolve_dim(
    provider: EmbeddingProviderChoice,
    local_model: EmbeddingModelChoice,
    openai_model: OpenAIModelChoice,
    dim: Option<i32>,
) -> Result<i32> {
    match provider {
        EmbeddingProviderChoice::Local => {
            let model_dim = model_dim(local_model);
            match dim {
                Some(value) => {
                    if value <= 0 {
                        anyhow::bail!("dim must be > 0");
                    }
                    if value != model_dim {
                        anyhow::bail!(
                            "dim must match model output ({model_dim}) for {:?}",
                            local_model
                        );
                    }
                    Ok(value)
                }
                None => Ok(model_dim),
            }
        }
        EmbeddingProviderChoice::OpenAI => {
            let model_dim = openai_model_dim(openai_model);
            match dim {
                Some(value) => {
                    if value <= 0 {
                        anyhow::bail!("dim must be > 0");
                    }
                    if value > model_dim {
                        anyhow::bail!(
                            "dim must be <= model output ({model_dim}) for {:?}",
                            openai_model
                        );
                    }
                    Ok(value)
                }
                None => Ok(model_dim),
            }
        }
    }
}

fn model_dim(model: EmbeddingModelChoice) -> i32 {
    match model {
        EmbeddingModelChoice::Nomic => 768,
        EmbeddingModelChoice::BgeM3 => 1024,
    }
}

fn openai_model_dim(model: OpenAIModelChoice) -> i32 {
    match model {
        OpenAIModelChoice::TextEmbedding3Small => 1536,
        OpenAIModelChoice::TextEmbedding3Large => 3072,
    }
}

fn openai_model_id(model: OpenAIModelChoice) -> &'static str {
    match model {
        OpenAIModelChoice::TextEmbedding3Small => "text-embedding-3-small",
        OpenAIModelChoice::TextEmbedding3Large => "text-embedding-3-large",
    }
}

pub(crate) fn build_embedder(
    provider: EmbeddingProviderChoice,
    local_model: EmbeddingModelChoice,
    openai_model: OpenAIModelChoice,
    show_download_progress: bool,
    dim: i32,
) -> Result<Embedder> {
    /*
    Local embeddings are cached in the user data directory to avoid polluting the repo and to
    reuse model artifacts across runs. OpenAI embeddings rely on environment credentials and
    only request non-default dimensions when explicitly configured.
    */
    match provider {
        EmbeddingProviderChoice::Local => {
            let fastembed_model = match local_model {
                EmbeddingModelChoice::Nomic => EmbeddingModel::NomicEmbedTextV15,
                EmbeddingModelChoice::BgeM3 => EmbeddingModel::BGEM3,
            };

            let mut options = InitOptions::new(fastembed_model);
            if show_download_progress {
                options = options.with_show_download_progress(true);
            }
            let cache_dir = app_data_dir()?.join("fastembed");
            fs::create_dir_all(&cache_dir)
                .with_context(|| format!("create directory {}", cache_dir.display()))?;
            options = options.with_cache_dir(cache_dir);

            Ok(Embedder::Local {
                inner: Box::new(TextEmbedding::try_new(options)?),
            })
        }
        EmbeddingProviderChoice::OpenAI => {
            Ok(Embedder::OpenAI(OpenAIClient::new(openai_model, dim)?))
        }
    }
}

fn prepare_document_inputs(
    provider: EmbeddingProviderChoice,
    local_model: EmbeddingModelChoice,
    chunks: &[String],
) -> Vec<String> {
    match provider {
        EmbeddingProviderChoice::Local => match local_model {
            EmbeddingModelChoice::Nomic => chunks
                .iter()
                .map(|chunk| format!("search_document: {chunk}"))
                .collect(),
            EmbeddingModelChoice::BgeM3 => chunks.to_vec(),
        },
        EmbeddingProviderChoice::OpenAI => chunks.to_vec(),
    }
}

async fn embed_texts(
    embedder: &mut Embedder,
    inputs: Vec<String>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    match embedder {
        Embedder::Local { inner } => Ok(inner.embed(inputs.as_slice(), Some(batch_size))?),
        Embedder::OpenAI(client) => client.embed(&inputs, batch_size).await,
    }
}

impl OpenAIClient {
    fn new(model: OpenAIModelChoice, dim: i32) -> Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY is not set")?;
        let organization = env::var("OPENAI_ORG_ID").ok();
        let project = env::var("OPENAI_PROJECT_ID").ok();
        let default_dim = openai_model_dim(model);
        let dimensions = if dim == default_dim { None } else { Some(dim) };

        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
            organization,
            project,
            model,
            dimensions,
        })
    }

    async fn embed(&self, inputs: &[String], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        let mut output = Vec::with_capacity(inputs.len());
        let chunk_size = batch_size.max(1);

        for chunk in inputs.chunks(chunk_size) {
            let request = OpenAIEmbeddingRequest {
                model: openai_model_id(self.model),
                input: chunk,
                dimensions: self.dimensions,
                encoding_format: "float",
            };

            let mut builder = self
                .http
                .post("https://api.openai.com/v1/embeddings")
                .bearer_auth(&self.api_key)
                .json(&request);

            if let Some(org) = &self.organization {
                builder = builder.header("OpenAI-Organization", org);
            }
            if let Some(project) = &self.project {
                builder = builder.header("OpenAI-Project", project);
            }

            let response = builder.send().await?;
            let status = response.status();
            let body = response.text().await?;
            if !status.is_success() {
                anyhow::bail!("OpenAI embeddings failed ({status}): {body}");
            }

            let mut parsed: OpenAIEmbeddingResponse = serde_json::from_str(&body)?;
            parsed.data.sort_by_key(|item| item.index);
            for item in parsed.data {
                output.push(item.embedding);
            }
        }

        Ok(output)
    }
}

fn should_reindex(
    algorithm: ChangeAlgorithm,
    existing: Option<&FileEntry>,
    mtime: i64,
    checksum: Option<&str>,
) -> Result<bool> {
    /*
    Two change-detection modes are supported: fast mtime checks or accurate checksums.
    The checksum is supplied by the caller so we only compute it when needed.
    */
    match algorithm {
        ChangeAlgorithm::Mtime => Ok(match existing {
            Some(entry) => entry.mtime != mtime,
            None => true,
        }),
        ChangeAlgorithm::Checksum => {
            let checksum = checksum.context("checksum required for checksum algorithm")?;
            Ok(match existing {
                Some(entry) => entry.checksum != checksum,
                None => true,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_text_with_overlap() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunk_text(text, 10, 2);
        assert_eq!(chunks, vec![
            "abcdefghij",
            "ijklmnopqr",
            "qrstuvwxyz",
        ]);
        assert_eq!(chunks.len(), count_chunks(text, 10, 2));
    }

    #[test]
    fn chunk_text_handles_short_input() {
        let text = "short";
        let chunks = chunk_text(text, 10, 3);
        assert_eq!(chunks, vec!["short"]);
        assert_eq!(chunks.len(), count_chunks(text, 10, 3));
    }

    #[test]
    fn should_reindex_mtime() -> Result<()> {
        let entry = FileEntry {
            checksum: "abc".to_string(),
            mtime: 123,
        };
        assert!(!should_reindex(ChangeAlgorithm::Mtime, Some(&entry), 123, None)?);
        assert!(should_reindex(ChangeAlgorithm::Mtime, Some(&entry), 124, None)?);
        assert!(should_reindex(ChangeAlgorithm::Mtime, None, 0, None)?);
        Ok(())
    }

    #[test]
    fn should_reindex_checksum() -> Result<()> {
        let entry = FileEntry {
            checksum: "abc".to_string(),
            mtime: 123,
        };
        assert!(!should_reindex(ChangeAlgorithm::Checksum, Some(&entry), 123, Some("abc"))?);
        assert!(should_reindex(ChangeAlgorithm::Checksum, Some(&entry), 123, Some("def"))?);
        assert!(should_reindex(ChangeAlgorithm::Checksum, None, 0, Some("abc"))?);
        Ok(())
    }
}
