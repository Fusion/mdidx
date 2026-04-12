mod args;
mod commands;
mod common;
mod fts;
mod indexing;
mod watch;

use anyhow::Result;
use clap::Parser;

use args::{Cli, Command};

pub async fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Index(args) => indexing::index_command(args).await,
        Command::Watch(args) => watch::watch_command(args).await,
        Command::FtsBuild(args) => fts::fts_build_command(args).await,
        Command::FtsRefresh(args) => fts::fts_refresh_command(args).await,
        Command::Search(args) => commands::search_command(args).await,
        Command::Stats(args) => commands::stats_command(args).await,
        Command::Version => commands::version_command(),
        Command::Vault(args) => commands::vault_command(args).await,
        Command::Config(args) => commands::config_command(args).await,
    }
}
