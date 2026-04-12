#[path = "lib/ai_config.rs"]
mod ai_config;
#[path = "lib/config.rs"]
mod config;
#[path = "lib/embedding.rs"]
mod embedding;
#[path = "lib/file_content.rs"]
mod file_content;
#[path = "lib/indexing.rs"]
mod indexing;
#[path = "lib/lancedb_util.rs"]
mod lancedb_util;
#[path = "lib/paths.rs"]
mod paths;
#[path = "lib/schema.rs"]
mod schema;
#[path = "lib/search.rs"]
mod search;
#[path = "lib/stats.rs"]
mod stats;
#[path = "lib/util.rs"]
mod util;
#[path = "lib/vaults.rs"]
mod vaults;

#[path = "lib/mcp.rs"]
pub mod mcp;

pub use ai_config::{AiConfig, load_ai_config, save_ai_config, set_classify_model};
pub use config::{IndexConfig, load_index_config, save_index_config};
pub use embedding::{EmbeddingModelChoice, EmbeddingProviderChoice, OpenAIModelChoice};
pub use file_content::{FileContentParams, FileContentResult, get_file_content};
pub use indexing::{IndexSingleFileParams, IndexSingleFileResult, index_single_file};
pub use paths::{app_data_dir, default_db_path};
pub use schema::{ChunkRow, build_chunk_batch, chunk_schema, file_schema};
pub use search::{
    FtsSearchParams, FtsSearchResult, FtsSearchResults, HybridSearchParams, HybridSearchResult,
    HybridSearchResults, SearchParams, SearchResult, SearchResults, search, search_fts, search_hybrid,
};
pub use stats::{StatsParams, StatsResults, format_epoch_seconds, stats};
pub use vaults::{VaultConfig, get_vault_path, load_vault_config, save_vault_config, set_vault_path};

pub const MDIDX_VERSION: &str = env!("CARGO_PKG_VERSION");
