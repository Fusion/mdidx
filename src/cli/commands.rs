use anyhow::Result;
use mdidx::{
    AiConfig, FtsSearchParams, FtsSearchResults, HybridSearchParams, HybridSearchResults,
    IndexConfig, SearchParams, SearchResult, SearchResults, load_ai_config, load_index_config,
    load_vault_config, search, search_fts, search_hybrid, set_classify_model, set_vault_path,
    stats, format_epoch_seconds, StatsParams, VaultConfig,
};

use super::args::{
    ConfigArgs, ConfigCommand, ConfigSetArgs, OutputFormat, SearchArgs, SearchMode, StatsArgs,
    VaultArgs, VaultCommand, VaultSetArgs,
};
use super::common::resolve_db_path;
use super::indexing::resolve_dim;

pub async fn search_command(args: SearchArgs) -> Result<()> {
    let db_path = resolve_db_path(args.db)?;
    let config = load_index_config(&db_path)?.unwrap_or(IndexConfig {
        provider: args.provider,
        model: args.model,
        openai_model: args.openai_model,
        dim: resolve_dim(args.provider, args.model, args.openai_model, args.dim)?,
    });
    match args.mode {
        SearchMode::Vector => {
            let params = SearchParams {
                query: args.query,
                db: db_path,
                provider: config.provider,
                model: config.model,
                openai_model: config.openai_model,
                dim: Some(config.dim),
                show_download_progress: args.show_download_progress,
                limit: args.limit,
                filter: args.filter,
                postfilter: args.postfilter,
                max_content_chars: args.max_content_chars,
            };
            let results = search(params).await?;
            print_search_results(&results, args.output)?;
        }
        SearchMode::Bm25 => {
            let params = FtsSearchParams {
                query: args.query,
                db: db_path,
                limit: args.limit,
                filter: args.filter,
                max_content_chars: args.max_content_chars,
            };
            let results = search_fts(params).await?;
            print_fts_results(&results, args.output)?;
        }
        SearchMode::Hybrid => {
            let params = HybridSearchParams {
                query: args.query,
                db: db_path,
                provider: config.provider,
                model: config.model,
                openai_model: config.openai_model,
                dim: Some(config.dim),
                show_download_progress: args.show_download_progress,
                limit: args.limit,
                filter: args.filter,
                postfilter: args.postfilter,
                max_content_chars: args.max_content_chars,
            };
            let results = search_hybrid(params).await?;
            print_hybrid_results(&results, args.output)?;
        }
    }
    Ok(())
}

pub async fn stats_command(args: StatsArgs) -> Result<()> {
    let db_path = resolve_db_path(args.db)?;
    let results = stats(StatsParams { db: db_path }).await?;
    println!("Files: {}", results.files);
    println!("Chunks: {}", results.chunks);
    match results.last_indexed_at {
        Some(value) => println!("Last indexed at: {}", format_epoch_seconds(value)),
        None => println!("Last indexed at: n/a"),
    }
    if let Some(config) = results.config {
        println!("Embedding provider: {:?}", config.provider);
        println!("Embedding model: {:?}", config.model);
        println!("OpenAI model: {:?}", config.openai_model);
        println!("Embedding dim: {}", config.dim);
    } else {
        println!("Embedding config: n/a");
    }
    Ok(())
}

pub async fn vault_command(args: VaultArgs) -> Result<()> {
    match args.command {
        VaultCommand::Set(args) => vault_set_command(args).await,
        VaultCommand::List => vault_list_command().await,
    }
}

async fn vault_set_command(args: VaultSetArgs) -> Result<()> {
    if !args.path.exists() {
        anyhow::bail!("vault path does not exist: {}", args.path.display());
    }
    if !args.path.is_dir() {
        anyhow::bail!("vault path is not a directory: {}", args.path.display());
    }
    let canonical = args.path.canonicalize().unwrap_or(args.path);
    set_vault_path(&args.id, &canonical)?;
    println!("vault {} -> {}", args.id, canonical.display());
    Ok(())
}

async fn vault_list_command() -> Result<()> {
    let VaultConfig { vaults } = load_vault_config()?;
    if vaults.is_empty() {
        println!("No vaults configured.");
        return Ok(());
    }
    for (id, path) in vaults {
        println!("{id}: {path}");
    }
    Ok(())
}

pub async fn config_command(args: ConfigArgs) -> Result<()> {
    match args.command {
        ConfigCommand::Set(args) => config_set_command(args).await,
        ConfigCommand::Show => config_show_command().await,
    }
}

async fn config_set_command(args: ConfigSetArgs) -> Result<()> {
    set_classify_model(args.classify_model)?;
    println!("AI config updated.");
    Ok(())
}

async fn config_show_command() -> Result<()> {
    let AiConfig { classify_model } = load_ai_config()?;
    match classify_model {
        Some(model) => println!("classify_model: {model}"),
        None => println!("classify_model: <default>"),
    }
    Ok(())
}

fn print_search_results(results: &SearchResults, output: OutputFormat) -> Result<()> {
    match output {
        OutputFormat::Human => print_search_results_human(&results.results),
        OutputFormat::Json => print_search_results_json(results),
    }
}

fn print_fts_results(results: &FtsSearchResults, output: OutputFormat) -> Result<()> {
    match output {
        OutputFormat::Human => print_fts_results_human(&results.results),
        OutputFormat::Json => print_fts_results_json(results),
    }
}

fn print_hybrid_results(results: &HybridSearchResults, output: OutputFormat) -> Result<()> {
    match output {
        OutputFormat::Human => print_hybrid_results_human(&results.results),
        OutputFormat::Json => print_hybrid_results_json(results),
    }
}

fn print_fts_results_human(results: &[mdidx::FtsSearchResult]) -> Result<()> {
    if results.is_empty() {
        println!("No results.");
        return Ok(());
    }
    for result in results {
        let score = result
            .score
            .map(|s| format!("{s:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{}:{} (score: {})",
            result.file_path, result.chunk_index, score
        );
        println!("{}", result.content);
        println!();
    }
    Ok(())
}

fn print_fts_results_json(results: &FtsSearchResults) -> Result<()> {
    let json = serde_json::to_string(results)?;
    println!("{json}");
    Ok(())
}

fn print_hybrid_results_human(results: &[mdidx::HybridSearchResult]) -> Result<()> {
    if results.is_empty() {
        println!("No results.");
        return Ok(());
    }
    for result in results {
        let distance = result
            .distance
            .map(|d| format!("{d:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        let score = result
            .score
            .map(|s| format!("{s:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        let relevance = result
            .relevance_score
            .map(|s| format!("{s:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "{}:{} (distance: {}, score: {}, relevance: {})",
            result.file_path, result.chunk_index, distance, score, relevance
        );
        println!("{}", result.content);
        println!();
    }
    Ok(())
}

fn print_hybrid_results_json(results: &HybridSearchResults) -> Result<()> {
    let json = serde_json::to_string(results)?;
    println!("{json}");
    Ok(())
}

fn print_search_results_human(results: &[SearchResult]) -> Result<()> {
    if results.is_empty() {
        println!("No results.");
        return Ok(());
    }
    for result in results {
        let distance = result
            .distance
            .map(|d| format!("{d:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!("{}:{} (distance: {})", result.file_path, result.chunk_index, distance);
        println!("{}", result.content);
        println!();
    }
    Ok(())
}

fn print_search_results_json(results: &SearchResults) -> Result<()> {
    let json = serde_json::to_string(results)?;
    println!("{json}");
    Ok(())
}
