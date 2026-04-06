use mdidx::{ChunkRow, build_chunk_batch, chunk_schema};
use arrow_schema::DataType;

#[test]
fn build_chunk_batch_schema_matches() {
    let rows = vec![ChunkRow {
        file_path: "doc.md".to_string(),
        chunk_index: 0,
        content: "hello world".to_string(),
        checksum: "abc".to_string(),
        mtime: 123,
        vector: vec![0.1, 0.2, 0.3],
    }];

    let dim = 3;
    let batch = build_chunk_batch(&rows, dim).expect("build batch");
    let schema = chunk_schema(dim);

    assert_eq!(batch.schema(), schema);

    let batch_schema = batch.schema();
    let vector_field = batch_schema
        .field_with_name("vector")
        .expect("vector field");
    match vector_field.data_type() {
        DataType::FixedSizeList(inner, size) => {
            assert_eq!(*size, dim);
            assert!(!inner.is_nullable());
            assert_eq!(inner.data_type(), &DataType::Float32);
        }
        other => panic!("unexpected vector field type: {other:?}"),
    }

    let vector_array = batch
        .column_by_name("vector")
        .expect("vector column");
    assert_eq!(vector_array.data_type(), vector_field.data_type());
}
