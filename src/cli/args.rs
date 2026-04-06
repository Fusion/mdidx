use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use mdidx::{EmbeddingModelChoice, EmbeddingProviderChoice, OpenAIModelChoice};

pub const DEFAULT_CHUNK_SIZE: usize = 1000;
pub const DEFAULT_CHUNK_OVERLAP: usize = 100;
pub const DEFAULT_BATCH_SIZE: usize = 512;

#[derive(Parser)]
#[command(author, version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Index(IndexArgs),
    Watch(WatchArgs),
    FtsBuild(FtsArgs),
    FtsRefresh(FtsArgs),
    Search(SearchArgs),
    Stats(StatsArgs),
    Vault(VaultArgs),
    Config(ConfigArgs),
}

#[derive(Parser)]
pub struct IndexArgs {
    #[arg(required = true, num_args = 1..)]
    pub paths: Vec<PathBuf>,

    #[arg(long)]
    pub db: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = ChangeAlgorithm::Mtime)]
    pub algorithm: ChangeAlgorithm,

    #[arg(long, value_enum)]
    pub provider: Option<EmbeddingProviderChoice>,

    #[arg(long, value_enum)]
    pub model: Option<EmbeddingModelChoice>,

    #[arg(long, value_enum)]
    pub openai_model: Option<OpenAIModelChoice>,

    #[arg(long)]
    pub dim: Option<i32>,

    #[arg(long, default_value_t = false)]
    pub show_download_progress: bool,

    #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
    pub chunk_size: usize,

    #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
    pub chunk_overlap: usize,

    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    pub batch_size: usize,

    #[arg(long, default_value_t = false)]
    pub quiet: bool,

    #[arg(long, default_value_t = false)]
    pub reset: bool,

    #[arg(long, default_value_t = false)]
    pub confirm: bool,

    #[arg(long, default_value_t = false)]
    pub fts_refresh: bool,
}

#[derive(Parser)]
pub struct WatchArgs {
    #[arg(required = true, num_args = 1..)]
    pub paths: Vec<PathBuf>,

    #[arg(long)]
    pub db: Option<PathBuf>,

    #[arg(long, value_enum)]
    pub provider: Option<EmbeddingProviderChoice>,

    #[arg(long, value_enum)]
    pub model: Option<EmbeddingModelChoice>,

    #[arg(long, value_enum)]
    pub openai_model: Option<OpenAIModelChoice>,

    #[arg(long)]
    pub dim: Option<i32>,

    #[arg(long, default_value_t = false)]
    pub show_download_progress: bool,

    #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
    pub chunk_size: usize,

    #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
    pub chunk_overlap: usize,

    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 750)]
    pub debounce_ms: u64,

    #[arg(long, default_value_t = 0)]
    pub fts_refresh_interval_secs: u64,

    #[arg(long, default_value_t = false)]
    pub quiet: bool,
}

#[derive(Parser)]
pub struct SearchArgs {
    pub query: String,

    #[arg(long)]
    pub db: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = EmbeddingProviderChoice::Local)]
    pub provider: EmbeddingProviderChoice,

    #[arg(long, value_enum, default_value_t = EmbeddingModelChoice::Nomic)]
    pub model: EmbeddingModelChoice,

    #[arg(long, value_enum, default_value_t = OpenAIModelChoice::TextEmbedding3Small)]
    pub openai_model: OpenAIModelChoice,

    #[arg(long)]
    pub dim: Option<i32>,

    #[arg(long, default_value_t = false)]
    pub show_download_progress: bool,

    #[arg(long, default_value_t = 5)]
    pub limit: usize,

    #[arg(long)]
    pub filter: Option<String>,

    #[arg(long, default_value_t = false)]
    pub postfilter: bool,

    #[arg(long, default_value_t = 200)]
    pub max_content_chars: usize,

    #[arg(long, value_enum, default_value_t = SearchMode::Vector)]
    pub mode: SearchMode,

    #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
    pub output: OutputFormat,
}

#[derive(Parser)]
pub struct StatsArgs {
    #[arg(long)]
    pub db: Option<PathBuf>,
}

#[derive(Parser)]
pub struct VaultArgs {
    #[command(subcommand)]
    pub command: VaultCommand,
}

#[derive(Subcommand)]
pub enum VaultCommand {
    Set(VaultSetArgs),
    List,
}

#[derive(Parser)]
pub struct VaultSetArgs {
    pub id: String,
    pub path: PathBuf,
}

#[derive(Parser)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: ConfigCommand,
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    Set(ConfigSetArgs),
    Show,
}

#[derive(Parser)]
pub struct ConfigSetArgs {
    #[arg(long)]
    pub classify_model: Option<String>,
}

#[derive(Parser)]
pub struct FtsArgs {
    #[arg(long)]
    pub db: Option<PathBuf>,

    #[arg(long, default_value_t = 2)]
    pub progress_interval_secs: u64,

    #[arg(long, default_value_t = 3600)]
    pub wait_timeout_secs: u64,

    #[arg(long, default_value_t = false)]
    pub quiet: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum ChangeAlgorithm {
    Mtime,
    Checksum,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum OutputFormat {
    Human,
    Json,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum SearchMode {
    Vector,
    Bm25,
    Hybrid,
}
