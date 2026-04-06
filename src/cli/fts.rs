use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use futures::future;
use lancedb::index::{Index, IndexType};
use lancedb::table::{OptimizeAction, Table};
use lancedb::{connect, Error as LanceError};

use super::args::FtsArgs;
use super::common::resolve_db_path;

pub const FTS_COLUMN: &str = "content";

pub async fn fts_build_command(args: FtsArgs) -> Result<()> {
    let db_path = resolve_db_path(args.db)?;
    let chunks_table = open_chunks_table(&db_path).await?;
    let index_name = ensure_fts_index(&chunks_table, FTS_COLUMN, args.quiet).await?;
    let interval = Duration::from_secs(args.progress_interval_secs);
    let timeout = Duration::from_secs(args.wait_timeout_secs);
    run_with_progress(
        &chunks_table,
        &index_name,
        interval,
        args.quiet,
        async {
            chunks_table
                .wait_for_index(&[index_name.as_str()], timeout)
                .await?;
            Ok(())
        },
    )
    .await?;
    if !args.quiet {
        print_fts_stats(&chunks_table, &index_name).await?;
    }
    Ok(())
}

pub async fn fts_refresh_command(args: FtsArgs) -> Result<()> {
    let db_path = resolve_db_path(args.db)?;
    let chunks_table = open_chunks_table(&db_path).await?;
    let index_name = ensure_fts_index(&chunks_table, FTS_COLUMN, args.quiet).await?;
    let interval = Duration::from_secs(args.progress_interval_secs);
    run_with_progress(
        &chunks_table,
        &index_name,
        interval,
        args.quiet,
        async { refresh_fts_index(&chunks_table, &index_name, args.quiet).await },
    )
    .await?;
    if !args.quiet {
        print_fts_stats(&chunks_table, &index_name).await?;
    }
    Ok(())
}

pub(crate) async fn ensure_fts_index(
    table: &Table,
    column: &str,
    quiet: bool,
) -> Result<String> {
    /*
    FTS creation is idempotent: if an index already exists we reuse it, otherwise we create one.
    We re-query the index list after creation so we can return the concrete name assigned by LanceDB.
    */
    if let Some(name) = find_fts_index_name(table, column).await? {
        if !quiet {
            println!("FTS index already exists: {name}");
        }
        return Ok(name);
    }

    if !quiet {
        println!("Creating FTS index on chunks.{column}...");
    }
    table
        .create_index(&[column], Index::FTS(Default::default()))
        .execute()
        .await?;

    if let Some(name) = find_fts_index_name(table, column).await? {
        if !quiet {
            println!("FTS index created: {name}");
        }
        return Ok(name);
    }

    Ok(format!("{column}_idx"))
}

async fn find_fts_index_name(table: &Table, column: &str) -> Result<Option<String>> {
    let indices = table.list_indices().await?;
    for index in indices {
        if index.index_type == IndexType::FTS && index.columns.iter().any(|col| col == column) {
            return Ok(Some(index.name));
        }
    }
    Ok(None)
}

pub(crate) async fn refresh_fts_index(table: &Table, index_name: &str, quiet: bool) -> Result<()> {
    /*
    Refresh only when LanceDB reports unindexed rows.
    This avoids expensive optimize calls when the FTS index is already current.
    */
    let stats = table.index_stats(index_name).await?;
    let needs_refresh = stats
        .as_ref()
        .map(|value| value.num_unindexed_rows > 0)
        .unwrap_or(true);

    if !needs_refresh {
        if !quiet {
            println!("FTS index is up to date: {index_name}");
        }
        return Ok(());
    }

    if !quiet {
        if let Some(value) = stats.as_ref() {
            println!(
                "Refreshing FTS index {index_name} ({} unindexed rows)...",
                value.num_unindexed_rows
            );
        } else {
            println!("Refreshing FTS index {index_name}...");
        }
    }

    table.optimize(OptimizeAction::Index(Default::default())).await?;

    if !quiet {
        println!("FTS index refresh complete: {index_name}");
    }

    Ok(())
}

async fn run_with_progress<T, F>(
    table: &Table,
    index_name: &str,
    interval: Duration,
    quiet: bool,
    op: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    /*
    Run an async index operation while periodically polling index stats.
    This keeps progress feedback responsive without coupling the core operation to logging.
    */
    if quiet || interval.is_zero() {
        return op.await;
    }

    let mut op = std::pin::pin!(op);
    let mut ticker = tokio::time::interval(interval);
    let mut last_unindexed: Option<usize> = None;
    let mut warned_unavailable = false;

    loop {
        tokio::select! {
            result = &mut op => {
                return result;
            }
            _ = ticker.tick() => {
                match table.index_stats(index_name).await {
                    Ok(Some(stats)) => {
                        let unindexed = stats.num_unindexed_rows;
                        if last_unindexed != Some(unindexed) {
                            println!("FTS indexing: {unindexed} unindexed rows remaining");
                            last_unindexed = Some(unindexed);
                        }
                    }
                    Ok(None) => {
                        if !warned_unavailable {
                            println!("FTS indexing: stats unavailable");
                            warned_unavailable = true;
                        }
                    }
                    Err(err) => {
                        if !warned_unavailable {
                            println!("FTS indexing: stats error: {err}");
                            warned_unavailable = true;
                        }
                    }
                }
            }
        }
    }
}

pub(crate) async fn print_fts_stats(table: &Table, index_name: &str) -> Result<()> {
    match table.index_stats(index_name).await? {
        Some(stats) => {
            println!(
                "FTS index {index_name}: {} indexed rows, {} unindexed rows",
                stats.num_indexed_rows, stats.num_unindexed_rows
            );
        }
        None => {
            println!("FTS index {index_name}: stats unavailable");
        }
    }
    Ok(())
}

async fn open_chunks_table(db_path: &Path) -> Result<Table> {
    let db_uri = db_path.to_string_lossy().to_string();
    let db = connect(&db_uri).execute().await?;
    match db.open_table("chunks").execute().await {
        Ok(table) => Ok(table),
        Err(LanceError::TableNotFound { .. }) => anyhow::bail!(
            "chunks table not found at {} (run `mdidx index` first)",
            db_path.display()
        ),
        Err(err) => Err(err.into()),
    }
}

pub(crate) async fn wait_for_fts_tick(fts_interval: &mut Option<tokio::time::Interval>) {
    if let Some(interval) = fts_interval {
        interval.tick().await;
    } else {
        future::pending::<()>().await;
    }
}
