use std::sync::Arc;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use mdidx::mcp::{add_tool_usage_hints, build_config, init_mcp_logger, set_mcp_log_level};
use model_context_protocol::protocol::{
    error_codes, JsonRpcId, JsonRpcMessage, JsonRpcResponse, LoggingLevel, SetLevelParams,
    ServerOutbound,
};
use model_context_protocol::server::{McpServer, ServerError, ServerStatus};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::RwLock;

#[derive(Parser)]
#[command(author, version, about = "mdidx MCP stdio server")]
struct Args {
    #[arg(long, value_enum, default_value_t = LogLevelArg::Info)]
    log_level: LogLevelArg,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "kebab_case")]
enum LogLevelArg {
    Error,
    Warning,
    Notice,
    Info,
    Debug,
    Critical,
    Alert,
    Emergency,
}

impl From<LogLevelArg> for LoggingLevel {
    fn from(value: LogLevelArg) -> Self {
        match value {
            LogLevelArg::Error => LoggingLevel::Error,
            LogLevelArg::Warning => LoggingLevel::Warning,
            LogLevelArg::Notice => LoggingLevel::Notice,
            LogLevelArg::Info => LoggingLevel::Info,
            LogLevelArg::Debug => LoggingLevel::Debug,
            LogLevelArg::Critical => LoggingLevel::Critical,
            LogLevelArg::Alert => LoggingLevel::Alert,
            LogLevelArg::Emergency => LoggingLevel::Emergency,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    /*
    Stdio MCP server loop.
    What: read JSON-RPC lines from stdin and write responses to stdout.
    Why: stdio is the most compatible transport for local MCP clients.
    How: intercept logging/setLevel and tools/list locally for low latency, forward all other traffic to the MCP server.
    */
    let args = Args::parse();
    let initial_level: LoggingLevel = args.log_level.into();
    let _ = init_mcp_logger();
    let _ = set_mcp_log_level(initial_level.clone());
    let (server, mut channels) = McpServer::new(build_config());
    let log_level = Arc::new(RwLock::new(initial_level));

    let stdout_handle = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(outbound) = channels.outbound_rx.recv().await {
            let json = match outbound.to_json() {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("Failed to serialize outbound message: {}", e);
                    continue;
                }
            };

            if stdout.write_all(json.as_bytes()).await.is_err() {
                break;
            }
            if stdout.write_all(b"\n").await.is_err() {
                break;
            }
            if stdout.flush().await.is_err() {
                break;
            }
        }
    });

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                match JsonRpcMessage::parse(trimmed) {
                    Ok(message) => {
                        if let JsonRpcMessage::Request(request) = &message {
                            if request.method == "logging/setLevel" {
                                let response = handle_set_level(request, &log_level).await;
                                let outbound = ServerOutbound::Response(response);
                                if channels.outbound_tx.send(outbound).await.is_err() {
                                    break;
                                }
                                continue;
                            }
                            if request.method == "tools/list" {
                                let tools = add_tool_usage_hints(server.list_tools());
                                let response = JsonRpcResponse::success(
                                    request.id.clone(),
                                    serde_json::json!({ "tools": tools }),
                                );
                                let outbound = ServerOutbound::Response(response);
                                if channels.outbound_tx.send(outbound).await.is_err() {
                                    break;
                                }
                                continue;
                            }
                        }

                        let inbound = message.into_client_inbound();
                        if channels.inbound_tx.send(inbound).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let error_response = JsonRpcResponse::error(
                            JsonRpcId::Null,
                            error_codes::PARSE_ERROR,
                            format!("Parse error: {}", e),
                            None,
                        );
                        let outbound = ServerOutbound::Response(error_response);
                        if channels.outbound_tx.send(outbound).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Err(err) => return Err(ServerError::Io(err).into()),
        }

        if server.status() != ServerStatus::Running {
            break;
        }
    }

    server.stop();
    let _ = stdout_handle.await;

    Ok(())
}

async fn handle_set_level(
    request: &model_context_protocol::protocol::JsonRpcRequest,
    log_level: &Arc<RwLock<LoggingLevel>>,
) -> JsonRpcResponse {
    let params = match request.params.clone() {
        Some(value) => value,
        None => {
            return JsonRpcResponse::error(
                request.id.clone(),
                error_codes::INVALID_PARAMS,
                "Missing params".to_string(),
                None,
            );
        }
    };

    let params: SetLevelParams = match serde_json::from_value(params) {
        Ok(value) => value,
        Err(err) => {
            return JsonRpcResponse::error(
                request.id.clone(),
                error_codes::INVALID_PARAMS,
                format!("Invalid params: {err}"),
                None,
            );
        }
    };

    let level = params.level.clone();
    let mut guard = log_level.write().await;
    *guard = level.clone();
    let _ = set_mcp_log_level(level);

    JsonRpcResponse::success(request.id.clone(), serde_json::json!({}))
}
