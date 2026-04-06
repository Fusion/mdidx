use std::path::PathBuf;

use anyhow::{Context, Result};
use arrow_array::{Int64Array, RecordBatch};
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::query::ExecutableQuery;
use lancedb::{connect, Error as LanceError};
use serde::{Deserialize, Serialize};

use crate::config::{IndexConfig, load_index_config};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsParams {
    pub db: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsResults {
    pub files: usize,
    pub chunks: usize,
    pub last_indexed_at: Option<i64>,
    pub config: Option<IndexConfig>,
}

pub async fn stats(params: StatsParams) -> Result<StatsResults> {
    let db_uri = params.db.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;
    let config = load_index_config(&params.db)?;

    let files_table = match db.open_table("files").execute().await {
        Ok(table) => table,
        Err(LanceError::TableNotFound { .. }) => {
            return Ok(StatsResults {
                files: 0,
                chunks: 0,
                last_indexed_at: None,
                config,
            });
        }
        Err(err) => return Err(err.into()),
    };

    let chunks_table = match db.open_table("chunks").execute().await {
        Ok(table) => table,
        Err(LanceError::TableNotFound { .. }) => {
            return Ok(StatsResults {
                files: 0,
                chunks: 0,
                last_indexed_at: None,
                config,
            });
        }
        Err(err) => return Err(err.into()),
    };

    let file_count = files_table.count_rows(None).await?;
    let chunk_count = chunks_table.count_rows(None).await?;
    let last_indexed = load_last_indexed_at(&files_table).await?;

    Ok(StatsResults {
        files: file_count,
        chunks: chunk_count,
        last_indexed_at: last_indexed,
        config,
    })
}

pub fn format_epoch_seconds(value: i64) -> String {
    if let Some(timestamp) = DateTime::<Utc>::from_timestamp(value, 0) {
        return timestamp.to_rfc3339();
    }
    value.to_string()
}

async fn load_last_indexed_at(table: &lancedb::table::Table) -> Result<Option<i64>> {
    let stream = table.query().execute().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    let mut latest: Option<i64> = None;

    for batch in batches {
        let indexed_at_col = batch
            .column_by_name("indexed_at")
            .context("files table missing indexed_at column")?;
        let indexed_at_arr = indexed_at_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("indexed_at is not Int64Array")?;

        for i in 0..batch.num_rows() {
            let value = indexed_at_arr.value(i);
            latest = Some(latest.map_or(value, |current| current.max(value)));
        }
    }

    Ok(latest)
}
