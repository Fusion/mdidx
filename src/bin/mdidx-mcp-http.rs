use std::sync::Arc;

use actix_web::{web, App, HttpRequest, HttpResponse, HttpServer};
use anyhow::Result;
use clap::Parser;
use mdidx::mcp::{add_tool_usage_hints, build_config_with_capabilities};
use model_context_protocol::protocol::{
    self, error_codes, ClientInbound, JsonRpcId, JsonRpcMessage, JsonRpcRequest, JsonRpcResponse,
    LoggingLevel, SetLevelParams,
};
use model_context_protocol::server::{McpServer, ServerError};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio::sync::RwLock;

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8080;
const DEFAULT_ENDPOINT: &str = "/mcp";
const HEADER_PROTOCOL_VERSION: &str = "MCP-Protocol-Version";
const HEADER_ORIGIN: &str = "Origin";

const SUPPORTED_PROTOCOL_VERSIONS: [&str; 2] = ["2025-11-25", "2025-03-26"];

#[derive(Parser)]
#[command(author, version, about = "mdidx MCP Streamable HTTP server")]
struct Args {
    #[arg(long, default_value = DEFAULT_HOST)]
    host: String,
    #[arg(long, default_value_t = DEFAULT_PORT)]
    port: u16,
    #[arg(long, default_value = DEFAULT_ENDPOINT)]
    endpoint: String,
}

struct AppState {
    server: Arc<McpServer>,
    inbound_tx: mpsc::Sender<ClientInbound>,
    endpoint: String,
    log_level: Arc<RwLock<LoggingLevel>>,
    capabilities: model_context_protocol::protocol::McpCapabilities,
}

#[actix_web::main]
async fn main() -> Result<()> {
    /*
    Streamable HTTP MCP server.
    What: expose MCP over plain HTTP POST/GET with JSON-RPC payloads.
    Why: enables modern MCP clients without SSE while keeping compatibility with stdio semantics.
    How: validate protocol headers, parse JSON-RPC (single or batch), and route to the MCP server.
    */
    let args = Args::parse();
    let (config, capabilities) = build_config_with_capabilities();
    let (server, mut channels) = McpServer::new(config);

    tokio::spawn(async move {
        while let Some(outbound) = channels.outbound_rx.recv().await {
            match &outbound {
                protocol::ServerOutbound::Notification(n) => {
                    eprintln!("[MCP] Notification: {}", n.method);
                }
                protocol::ServerOutbound::Request(r) => {
                    eprintln!("[MCP] Server request: {}", r.method);
                }
                _ => {}
            }
        }
    });

    let state = web::Data::new(AppState {
        server,
        inbound_tx: channels.inbound_tx.clone(),
        endpoint: args.endpoint.clone(),
        log_level: Arc::new(RwLock::new(LoggingLevel::Info)),
        capabilities,
    });
    let endpoint = args.endpoint.clone();

    HttpServer::new(move || {
        let state = state.clone();
        let endpoint = endpoint.clone();
        App::new()
            .app_data(state)
            .route(&endpoint, web::post().to(handle_mcp_post))
            .route(&endpoint, web::get().to(handle_mcp_get))
            .route("/rpc", web::post().to(handle_mcp_post))
            .route("/tools", web::get().to(handle_tools_list))
            .route("/call", web::post().to(handle_tool_call))
            .route("/health", web::get().to(handle_health))
            .route("/.well-known/mcp.json", web::get().to(handle_well_known))
            .route("/", web::get().to(handle_root))
    })
    .bind((args.host.as_str(), args.port))
    .map_err(|e| ServerError::Io(std::io::Error::new(std::io::ErrorKind::AddrInUse, e)))?
    .run()
    .await
    .map_err(|e| ServerError::Io(std::io::Error::other(e)))?;

    Ok(())
}

async fn handle_mcp_get(req: HttpRequest) -> HttpResponse {
    if let Some(resp) = validate_origin(&req) {
        return resp;
    }
    if let Some(resp) = validate_protocol_version(&req) {
        return resp;
    }
    HttpResponse::MethodNotAllowed().finish()
}

async fn handle_mcp_post(
    req: HttpRequest,
    body: String,
    state: web::Data<AppState>,
) -> HttpResponse {
    /*
    POST handler accepts either a single JSON-RPC message or a batch array.
    We validate protocol headers early, then parse and route messages to the MCP server,
    returning JSON-RPC error responses for malformed requests.
    */
    if let Some(resp) = validate_origin(&req) {
        return resp;
    }
    if let Some(resp) = validate_protocol_version(&req) {
        return resp;
    }

    let value: serde_json::Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(err) => {
            let error_response = JsonRpcResponse::error(
                JsonRpcId::Null,
                error_codes::PARSE_ERROR,
                format!("Parse error: {err}"),
                None,
            );
            return HttpResponse::BadRequest().json(error_response);
        }
    };

    match value {
        serde_json::Value::Array(items) => handle_batch(state, items).await,
        other => handle_single(state, other).await,
    }
}

async fn handle_single(state: web::Data<AppState>, value: serde_json::Value) -> HttpResponse {
    let message: JsonRpcMessage = match serde_json::from_value(value) {
        Ok(message) => message,
        Err(err) => {
            let error_response = JsonRpcResponse::error(
                JsonRpcId::Null,
                error_codes::INVALID_REQUEST,
                format!("Invalid request: {err}"),
                None,
            );
            return HttpResponse::BadRequest().json(error_response);
        }
    };

    match message {
        JsonRpcMessage::Request(request) => {
            let response = handle_request_directly(&state, request).await;
            HttpResponse::Ok().json(response)
        }
        JsonRpcMessage::Notification(notification) => {
            let _ = state
                .inbound_tx
                .send(ClientInbound::Notification(notification))
                .await;
            HttpResponse::Accepted().finish()
        }
        JsonRpcMessage::Response(_) => HttpResponse::Accepted().finish(),
    }
}

async fn handle_batch(state: web::Data<AppState>, items: Vec<serde_json::Value>) -> HttpResponse {
    if items.is_empty() {
        let error_response = JsonRpcResponse::error(
            JsonRpcId::Null,
            error_codes::INVALID_REQUEST,
            "Invalid request: empty batch".to_string(),
            None,
        );
        return HttpResponse::BadRequest().json(error_response);
    }

    let mut responses: Vec<JsonRpcResponse> = Vec::new();
    let mut saw_request = false;

    for item in items {
        let message: JsonRpcMessage = match serde_json::from_value(item) {
            Ok(message) => message,
            Err(err) => {
                responses.push(JsonRpcResponse::error(
                    JsonRpcId::Null,
                    error_codes::INVALID_REQUEST,
                    format!("Invalid request: {err}"),
                    None,
                ));
                saw_request = true;
                continue;
            }
        };

        match message {
            JsonRpcMessage::Request(request) => {
                saw_request = true;
                responses.push(handle_request_directly(&state, request).await);
            }
            JsonRpcMessage::Notification(notification) => {
                let _ = state
                    .inbound_tx
                    .send(ClientInbound::Notification(notification))
                    .await;
            }
            JsonRpcMessage::Response(_) => {}
        }
    }

    if !saw_request {
        return HttpResponse::Accepted().finish();
    }

    HttpResponse::Ok().json(responses)
}

async fn handle_request_directly(state: &AppState, request: JsonRpcRequest) -> JsonRpcResponse {
    /*
    Fast-path local handling for core MCP methods.
    This avoids a full round-trip through the inbound channel for initialize/tools/list/logging setLevel,
    while still delegating tool calls and other requests to the server.
    */
    match request.method.as_str() {
        "initialize" => JsonRpcResponse::success(
            request.id,
            serde_json::json!({
                "protocolVersion": model_context_protocol::MCP_PROTOCOL_VERSION,
                "serverInfo": state.server.server_info(),
                "capabilities": state.capabilities.clone()
            }),
        ),
        "tools/list" => {
            let tools = add_tool_usage_hints(state.server.list_tools());
            JsonRpcResponse::success(request.id, serde_json::json!({ "tools": tools }))
        }
        "tools/call" => {
            let params = match request.params {
                Some(p) => p,
                None => {
                    return JsonRpcResponse::error(
                        request.id,
                        error_codes::INVALID_PARAMS,
                        "Missing params".to_string(),
                        None,
                    );
                }
            };

            let name = match params.get("name").and_then(|n| n.as_str()) {
                Some(n) => n,
                None => {
                    return JsonRpcResponse::error(
                        request.id,
                        error_codes::INVALID_PARAMS,
                        "Missing tool name".to_string(),
                        None,
                    );
                }
            };

            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));

            match state.server.call_tool(name, arguments).await {
                Ok(content) => JsonRpcResponse::success(
                    request.id,
                    serde_json::json!({
                        "content": content,
                        "isError": false
                    }),
                ),
                Err(err) => JsonRpcResponse::success(
                    request.id,
                    serde_json::json!({
                        "content": [{ "type": "text", "text": err.to_string() }],
                        "isError": true
                    }),
                ),
            }
        }
        "ping" => JsonRpcResponse::success(request.id, serde_json::json!({})),
        "logging/setLevel" => handle_set_level(&state.log_level, request).await,
        _ => JsonRpcResponse::error(
            request.id,
            error_codes::METHOD_NOT_FOUND,
            format!("Method not found: {}", request.method),
            None,
        ),
    }
}

async fn handle_tools_list(state: web::Data<AppState>) -> HttpResponse {
    let tools = add_tool_usage_hints(state.server.list_tools());
    HttpResponse::Ok().json(tools)
}

#[derive(Deserialize)]
struct CallToolRequest {
    name: String,
    arguments: serde_json::Value,
}

async fn handle_tool_call(
    state: web::Data<AppState>,
    body: web::Json<CallToolRequest>,
) -> HttpResponse {
    match state
        .server
        .call_tool(&body.name, body.arguments.clone())
        .await
    {
        Ok(content) => HttpResponse::Ok().json(content),
        Err(err) => HttpResponse::InternalServerError().json(serde_json::json!({
            "error": err.to_string()
        })),
    }
}

async fn handle_health(state: web::Data<AppState>) -> HttpResponse {
    let status = state.server.status();
    HttpResponse::Ok().json(serde_json::json!({
        "status": format!("{:?}", status),
        "name": state.server.name(),
        "version": state.server.version()
    }))
}

async fn handle_root(state: web::Data<AppState>) -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "name": state.server.name(),
        "version": state.server.version(),
        "protocolVersion": model_context_protocol::MCP_PROTOCOL_VERSION,
        "endpoints": {
            "mcp": state.endpoint.as_str(),
            "rpc": "/rpc",
            "tools": "/tools",
            "call": "/call",
            "health": "/health"
        }
    }))
}

async fn handle_well_known(state: web::Data<AppState>) -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "name": state.server.name(),
        "version": state.server.version(),
        "protocolVersion": model_context_protocol::MCP_PROTOCOL_VERSION,
        "transport": "streamable_http",
        "endpoint": state.endpoint.as_str()
    }))
}

async fn handle_set_level(
    log_level: &Arc<RwLock<LoggingLevel>>,
    request: JsonRpcRequest,
) -> JsonRpcResponse {
    let params = match request.params.clone() {
        Some(value) => value,
        None => {
            return JsonRpcResponse::error(
                request.id,
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
                request.id,
                error_codes::INVALID_PARAMS,
                format!("Invalid params: {err}"),
                None,
            );
        }
    };

    let mut guard = log_level.write().await;
    *guard = params.level;

    JsonRpcResponse::success(request.id, serde_json::json!({}))
}

fn validate_origin(req: &HttpRequest) -> Option<HttpResponse> {
    let origin = req.headers().get(HEADER_ORIGIN)?;
    let origin = match origin.to_str() {
        Ok(value) => value,
        Err(_) => {
            return Some(
                HttpResponse::Forbidden().json(serde_json::json!({
                    "error": "Invalid Origin header"
                })),
            );
        }
    };

    if is_allowed_origin(origin) {
        return None;
    }

    Some(HttpResponse::Forbidden().json(serde_json::json!({
        "error": "Origin not allowed"
    })))
}

fn validate_protocol_version(req: &HttpRequest) -> Option<HttpResponse> {
    let header = req.headers().get(HEADER_PROTOCOL_VERSION)?;

    let value = match header.to_str() {
        Ok(value) => value,
        Err(_) => {
            return Some(
                HttpResponse::BadRequest().json(serde_json::json!({
                    "error": "Invalid MCP-Protocol-Version header"
                })),
            );
        }
    };

    if SUPPORTED_PROTOCOL_VERSIONS.contains(&value) {
        return None;
    }

    Some(
        HttpResponse::BadRequest().json(serde_json::json!({
            "error": format!("Unsupported MCP protocol version: {value}"),
            "supported": SUPPORTED_PROTOCOL_VERSIONS,
        })),
    )
}

fn is_allowed_origin(origin: &str) -> bool {
    let origin = origin.trim();
    let origin = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"));
    let Some(rest) = origin else {
        return false;
    };
    let host_port = rest.split('/').next().unwrap_or(rest);
    let host = host_port.split(':').next().unwrap_or(host_port);
    matches!(host, "localhost" | "127.0.0.1")
}
