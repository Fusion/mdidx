use anyhow::Result;
use lancedb::table::Table;

pub(crate) async fn verify_vector_dim(table: &Table, expected_dim: i32) -> Result<()> {
    let schema = table.schema().await?;
    let field = schema.field_with_name("vector")?;
    if let arrow_schema::DataType::FixedSizeList(_, size) = field.data_type() {
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
