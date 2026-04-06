use std::sync::Arc;

use anyhow::Result;
use arrow_array::{FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub file_path: String,
    pub chunk_index: i32,
    pub content: String,
    pub checksum: String,
    pub mtime: i64,
    pub vector: Vec<f32>,
}

pub fn chunk_schema(dim: i32) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("checksum", DataType::Utf8, false),
        Field::new("mtime", DataType::Int64, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dim,
            ),
            false,
        ),
    ]))
}

pub fn file_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("file_path", DataType::Utf8, false),
        Field::new("checksum", DataType::Utf8, false),
        Field::new("mtime", DataType::Int64, false),
        Field::new("chunk_count", DataType::Int32, false),
        Field::new("indexed_at", DataType::Int64, false),
    ]))
}

pub fn build_chunk_batch(rows: &[ChunkRow], dim: i32) -> Result<RecordBatch> {
    let schema = chunk_schema(dim);

    let file_path_array = StringArray::from_iter_values(rows.iter().map(|r| r.file_path.as_str()));
    let chunk_index_array = Int32Array::from_iter_values(rows.iter().map(|r| r.chunk_index));
    let content_array = StringArray::from_iter_values(rows.iter().map(|r| r.content.as_str()));
    let checksum_array = StringArray::from_iter_values(rows.iter().map(|r| r.checksum.as_str()));
    let mtime_array = Int64Array::from_iter_values(rows.iter().map(|r| r.mtime));

    for row in rows {
        if row.vector.len() != dim as usize {
            anyhow::bail!(
                "vector length mismatch: expected {}, got {}",
                dim,
                row.vector.len()
            );
        }
    }

    let flat_values: Vec<f32> = rows
        .iter()
        .flat_map(|row| row.vector.iter().copied())
        .collect();
    let values = Float32Array::from_iter_values(flat_values);
    let vector_field = Arc::new(Field::new("item", DataType::Float32, false));
    let vector_array = FixedSizeListArray::try_new(vector_field, dim, Arc::new(values), None)?;

    let columns = vec![
        Arc::new(file_path_array) as _,
        Arc::new(chunk_index_array) as _,
        Arc::new(content_array) as _,
        Arc::new(checksum_array) as _,
        Arc::new(mtime_array) as _,
        Arc::new(vector_array) as _,
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}
