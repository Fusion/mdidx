use std::path::PathBuf;

use anyhow::{Context, Result};
use arrow_array::{Float32Array, Float64Array, Int32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use lance_index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::connect;
use serde::{Deserialize, Serialize};

use crate::embedding::{
    EmbeddingModelChoice, EmbeddingProviderChoice, OpenAIModelChoice, build_embedder,
    embed_texts, prepare_query_input, resolve_dim,
};
use crate::lancedb_util::verify_vector_dim;
use crate::util::truncate_chars;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub query: String,
    pub db: PathBuf,
    pub provider: EmbeddingProviderChoice,
    pub model: EmbeddingModelChoice,
    pub openai_model: OpenAIModelChoice,
    pub dim: Option<i32>,
    pub show_download_progress: bool,
    pub limit: usize,
    pub filter: Option<String>,
    pub postfilter: bool,
    pub max_content_chars: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResults {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub chunk_index: i32,
    pub distance: Option<f64>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchParams {
    pub query: String,
    pub db: PathBuf,
    pub provider: EmbeddingProviderChoice,
    pub model: EmbeddingModelChoice,
    pub openai_model: OpenAIModelChoice,
    pub dim: Option<i32>,
    pub show_download_progress: bool,
    pub limit: usize,
    pub filter: Option<String>,
    pub postfilter: bool,
    pub max_content_chars: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HybridSearchResults {
    pub results: Vec<HybridSearchResult>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HybridSearchResult {
    pub file_path: String,
    pub chunk_index: i32,
    pub distance: Option<f64>,
    pub score: Option<f64>,
    pub relevance_score: Option<f64>,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsSearchParams {
    pub query: String,
    pub db: PathBuf,
    pub limit: usize,
    pub filter: Option<String>,
    pub max_content_chars: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FtsSearchResults {
    pub results: Vec<FtsSearchResult>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FtsSearchResult {
    pub file_path: String,
    pub chunk_index: i32,
    pub score: Option<f64>,
    pub content: String,
}

pub async fn search(params: SearchParams) -> Result<SearchResults> {
    /*
    Semantic search pipeline.
    What: embed the query, run a vector nearest-neighbor search, then parse results.
    Why: semantic matching surfaces relevant chunks beyond exact keyword overlap.
    How: resolve embedding dims, verify table schema, embed the query, and execute the LanceDB vector query.
    */
    if params.limit == 0 {
        anyhow::bail!("limit must be > 0");
    }

    let dim = resolve_dim(
        params.provider,
        params.model,
        params.openai_model,
        params.dim,
    )?;

    let db_uri = params.db.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;
    let chunks_table = db.open_table("chunks").execute().await?;

    verify_vector_dim(&chunks_table, dim).await?;

    let mut embedder = build_embedder(
        params.provider,
        params.model,
        params.openai_model,
        params.show_download_progress,
        dim,
    )?;
    let query_input = prepare_query_input(params.provider, params.model, &params.query);
    let embeddings = embed_texts(&mut embedder, vec![query_input], 1).await?;
    let vector = embeddings.into_iter().next().context("no embedding returned")?;

    let mut query = chunks_table
        .query()
        .nearest_to(vector.as_slice())?
        .limit(params.limit);

    if let Some(filter) = params.filter.as_deref() {
        query = query.only_if(filter);
    }
    if params.postfilter {
        query = query.postfilter();
    }

    let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
    let results = collect_search_results(&batches, params.max_content_chars)?;

    Ok(SearchResults { results })
}

pub async fn search_fts(params: FtsSearchParams) -> Result<FtsSearchResults> {
    /*
    BM25 search pipeline.
    What: run a full-text search over the chunks table.
    Why: exact term/phrase matches are valuable when semantics are too loose.
    How: execute LanceDB FTS over the content column and parse results.
    */
    if params.limit == 0 {
        anyhow::bail!("limit must be > 0");
    }

    let db_uri = params.db.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;
    let chunks_table = db.open_table("chunks").execute().await?;

    // Assumes an FTS index exists on the chunks table; creation is handled elsewhere.
    let mut query = chunks_table
        .query()
        .full_text_search(FullTextSearchQuery::new(params.query))
        .limit(params.limit);

    if let Some(filter) = params.filter.as_deref() {
        query = query.only_if(filter);
    }

    let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
    let results = collect_fts_results(&batches, params.max_content_chars)?;

    Ok(FtsSearchResults { results })
}

pub async fn search_hybrid(params: HybridSearchParams) -> Result<HybridSearchResults> {
    /*
    Hybrid search pipeline.
    What: combine vector similarity with BM25 and let LanceDB fuse via RRF.
    Why: improves recall by capturing both semantic and lexical signals.
    How: embed the query, run a hybrid query, and preserve distance/score/relevance fields.
    */
    if params.limit == 0 {
        anyhow::bail!("limit must be > 0");
    }

    let dim = resolve_dim(
        params.provider,
        params.model,
        params.openai_model,
        params.dim,
    )?;

    let db_uri = params.db.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;
    let chunks_table = db.open_table("chunks").execute().await?;

    verify_vector_dim(&chunks_table, dim).await?;

    let mut embedder = build_embedder(
        params.provider,
        params.model,
        params.openai_model,
        params.show_download_progress,
        dim,
    )?;
    let query_input = prepare_query_input(params.provider, params.model, &params.query);
    let embeddings = embed_texts(&mut embedder, vec![query_input], 1).await?;
    let vector = embeddings.into_iter().next().context("no embedding returned")?;

    // LanceDB fuses vector + BM25 results with RRF; we keep all scoring fields for callers.
    let mut query = chunks_table
        .query()
        .nearest_to(vector.as_slice())?
        .full_text_search(FullTextSearchQuery::new(params.query))
        .limit(params.limit);

    if let Some(filter) = params.filter.as_deref() {
        query = query.only_if(filter);
    }
    if params.postfilter {
        query = query.postfilter();
    }

    let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
    let results = collect_hybrid_results(&batches, params.max_content_chars)?;

    Ok(HybridSearchResults { results })
}

fn collect_search_results(batches: &[RecordBatch], max_content_chars: usize) -> Result<Vec<SearchResult>> {
    /*
    RecordBatch extraction.
    What: convert Arrow columns into typed fields and apply content truncation.
    Why: LanceDB returns Arrow batches; we expose a stable, JSON-friendly shape.
    How: downcast columns, read row values, and optionally trim content.
    */
    let mut total = 0usize;
    let mut results = Vec::new();
    for batch in batches {
        let file_col = batch
            .column_by_name("file_path")
            .context("missing file_path column")?;
        let chunk_col = batch
            .column_by_name("chunk_index")
            .context("missing chunk_index column")?;
        let content_col = batch
            .column_by_name("content")
            .context("missing content column")?;

        let file_arr = file_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("file_path is not StringArray")?;
        let chunk_arr = chunk_col
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("chunk_index is not Int32Array")?;
        let content_arr = content_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("content is not StringArray")?;

        let distance_arr = batch.column_by_name("_distance");
        for i in 0..batch.num_rows() {
            total += 1;
            let file_path = file_arr.value(i);
            let chunk_index = chunk_arr.value(i);
            let content = content_arr.value(i);
            let distance = distance_arr
                .and_then(|col| extract_distance(col, i).ok())
                .map(|d| (d * 10000.0).round() / 10000.0);

            let content = if max_content_chars == 0 {
                content.to_string()
            } else {
                truncate_chars(content, max_content_chars)
            };

            results.push(SearchResult {
                file_path: file_path.to_string(),
                chunk_index,
                distance,
                content,
            });
        }
    }

    if total == 0 {
        return Ok(Vec::new());
    }

    Ok(results)
}

fn collect_fts_results(batches: &[RecordBatch], max_content_chars: usize) -> Result<Vec<FtsSearchResult>> {
    let mut total = 0usize;
    let mut results = Vec::new();
    for batch in batches {
        let file_col = batch
            .column_by_name("file_path")
            .context("missing file_path column")?;
        let chunk_col = batch
            .column_by_name("chunk_index")
            .context("missing chunk_index column")?;
        let content_col = batch
            .column_by_name("content")
            .context("missing content column")?;

        let file_arr = file_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("file_path is not StringArray")?;
        let chunk_arr = chunk_col
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("chunk_index is not Int32Array")?;
        let content_arr = content_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("content is not StringArray")?;

        let score_arr = batch.column_by_name("_score");
        for i in 0..batch.num_rows() {
            total += 1;
            let file_path = file_arr.value(i);
            let chunk_index = chunk_arr.value(i);
            let content = content_arr.value(i);
            let score = score_arr
                .and_then(|col| extract_score(col, i).ok())
                .map(|s| (s * 10000.0).round() / 10000.0);

            let content = if max_content_chars == 0 {
                content.to_string()
            } else {
                truncate_chars(content, max_content_chars)
            };

            results.push(FtsSearchResult {
                file_path: file_path.to_string(),
                chunk_index,
                score,
                content,
            });
        }
    }

    if total == 0 {
        return Ok(Vec::new());
    }

    Ok(results)
}

fn collect_hybrid_results(batches: &[RecordBatch], max_content_chars: usize) -> Result<Vec<HybridSearchResult>> {
    let mut total = 0usize;
    let mut results = Vec::new();
    for batch in batches {
        let file_col = batch
            .column_by_name("file_path")
            .context("missing file_path column")?;
        let chunk_col = batch
            .column_by_name("chunk_index")
            .context("missing chunk_index column")?;
        let content_col = batch
            .column_by_name("content")
            .context("missing content column")?;

        let file_arr = file_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("file_path is not StringArray")?;
        let chunk_arr = chunk_col
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("chunk_index is not Int32Array")?;
        let content_arr = content_col
            .as_any()
            .downcast_ref::<StringArray>()
            .context("content is not StringArray")?;

        let distance_arr = batch.column_by_name("_distance");
        let score_arr = batch.column_by_name("_score");
        let relevance_arr = batch.column_by_name("_relevance_score");
        for i in 0..batch.num_rows() {
            total += 1;
            let file_path = file_arr.value(i);
            let chunk_index = chunk_arr.value(i);
            let content = content_arr.value(i);
            let distance = distance_arr
                .and_then(|col| extract_distance(col, i).ok())
                .map(|d| (d * 10000.0).round() / 10000.0);
            let score = score_arr
                .and_then(|col| extract_score(col, i).ok())
                .map(|s| (s * 10000.0).round() / 10000.0);
            let relevance_score = relevance_arr
                .and_then(|col| extract_score(col, i).ok())
                .map(|s| (s * 10000.0).round() / 10000.0);

            let content = if max_content_chars == 0 {
                content.to_string()
            } else {
                truncate_chars(content, max_content_chars)
            };

            results.push(HybridSearchResult {
                file_path: file_path.to_string(),
                chunk_index,
                distance,
                score,
                relevance_score,
                content,
            });
        }
    }

    if total == 0 {
        return Ok(Vec::new());
    }

    Ok(results)
}

fn extract_distance(col: &arrow_array::ArrayRef, row: usize) -> Result<f64> {
    // LanceDB can expose float columns as f32 or f64 depending on the backend.
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return Ok(arr.value(row) as f64);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return Ok(arr.value(row));
    }
    anyhow::bail!("_distance column has unsupported type");
}

fn extract_score(col: &arrow_array::ArrayRef, row: usize) -> Result<f64> {
    // LanceDB can expose float columns as f32 or f64 depending on the backend.
    if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
        return Ok(arr.value(row) as f64);
    }
    if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
        return Ok(arr.value(row));
    }
    anyhow::bail!("_score column has unsupported type");
}
