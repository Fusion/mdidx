use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::Schema;
use lancedb::table::Table;
use lancedb::{connect, Error as LanceError};
use serde::{Deserialize, Serialize};

use crate::embedding::{
    EmbeddingModelChoice, EmbeddingProviderChoice, OpenAIModelChoice, build_embedder,
    embed_texts, prepare_document_inputs,
};
use crate::lancedb_util::verify_vector_dim;
use crate::schema::{ChunkRow, build_chunk_batch, chunk_schema, file_schema};
use crate::util::{modified_nanos, now_epoch_seconds, sha256_hex, sql_escape};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSingleFileParams {
    pub db: PathBuf,
    pub file_path: PathBuf,
    pub provider: EmbeddingProviderChoice,
    pub model: EmbeddingModelChoice,
    pub openai_model: OpenAIModelChoice,
    pub dim: i32,
    pub show_download_progress: bool,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSingleFileResult {
    pub file_path: String,
    pub chunks_upserted: usize,
}

pub async fn index_single_file(params: IndexSingleFileParams) -> Result<IndexSingleFileResult> {
    /*
    Single-file indexing is used by tooling that updates one path at a time (e.g. MCP).
    It enforces the same chunking and embedding rules as the CLI pipeline and performs a full
    replace of the file's chunks to avoid stale data when content changes.
    */
    if params.chunk_size == 0 {
        anyhow::bail!("chunk_size must be > 0");
    }
    if params.chunk_overlap >= params.chunk_size {
        anyhow::bail!("chunk_overlap must be < chunk_size");
    }
    if params.batch_size == 0 {
        anyhow::bail!("batch_size must be > 0");
    }

    let db_uri = params.db.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;
    let chunk_schema = chunk_schema(params.dim);
    let file_schema = file_schema();
    let chunks_table = ensure_table(&db, "chunks", chunk_schema.clone()).await?;
    let files_table = ensure_table(&db, "files", file_schema.clone()).await?;

    verify_vector_dim(&chunks_table, params.dim).await?;

    let mut embedder = build_embedder(
        params.provider,
        params.model,
        params.openai_model,
        params.show_download_progress,
        params.dim,
    )?;

    let content = fs::read_to_string(&params.file_path)
        .with_context(|| format!("read: {}", params.file_path.display()))?;
    let metadata = fs::metadata(&params.file_path)
        .with_context(|| format!("metadata: {}", params.file_path.display()))?;
    let mtime = modified_nanos(&metadata)?;
    let checksum = sha256_hex(content.as_bytes());

    let canonical_path = params
        .file_path
        .canonicalize()
        .unwrap_or_else(|_| params.file_path.clone());
    let canonical_path = canonical_path.to_string_lossy().to_string();

    let index_params = IndexFileParams {
        chunks_table: &chunks_table,
        files_table: &files_table,
        embedder: &mut embedder,
        provider: params.provider,
        model: params.model,
        file_path: &canonical_path,
        content: &content,
        checksum,
        mtime,
        dim: params.dim,
        chunk_size: params.chunk_size,
        chunk_overlap: params.chunk_overlap,
        batch_size: params.batch_size,
    };
    index_file(index_params).await?;

    Ok(IndexSingleFileResult {
        file_path: canonical_path,
        chunks_upserted: count_chunks(&content, params.chunk_size, params.chunk_overlap),
    })
}

async fn ensure_table(
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

async fn delete_file_entries(chunks_table: &Table, files_table: &Table, file_path: &str) -> Result<()> {
    let predicate = format!("file_path = '{}'", sql_escape(file_path));
    chunks_table.delete(&predicate).await?;
    files_table.delete(&predicate).await?;
    Ok(())
}

async fn add_chunk_rows(table: &Table, rows: &[ChunkRow], dim: i32) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let batch = build_chunk_batch(rows, dim)?;
    table.add(batch).execute().await?;
    Ok(())
}

struct IndexFileParams<'a> {
    chunks_table: &'a Table,
    files_table: &'a Table,
    embedder: &'a mut crate::embedding::Embedder,
    provider: EmbeddingProviderChoice,
    model: EmbeddingModelChoice,
    file_path: &'a str,
    content: &'a str,
    checksum: String,
    mtime: i64,
    dim: i32,
    chunk_size: usize,
    chunk_overlap: usize,
    batch_size: usize,
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

async fn index_file(
    params: IndexFileParams<'_>,
) -> Result<()> {
    /*
    Replace-all write path: delete existing rows, chunk, embed, then append new chunks and file metadata.
    This keeps chunk rows and the files table consistent without requiring partial update semantics.
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
    // Delete first so reindexing cannot leave stale chunks when file content shrinks.
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

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    /*
    Chunking uses character indices to preserve valid UTF-8 slicing.
    Overlap is expressed in characters so adjacent chunks share context for embeddings.
    */
    if text.is_empty() {
        return Vec::new();
    }

    // Work in character indices to avoid slicing invalid UTF-8 boundaries.
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
    /*
    Mirror `chunk_text` without allocating chunk strings.
    This lets us report chunk counts cheaply for stats and metadata.
    */
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
