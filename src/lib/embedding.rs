use std::env;

use anyhow::{Context, Result};
use clap::ValueEnum;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use serde::{Deserialize, Serialize};

use crate::paths::{app_data_dir, ensure_dir};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingProviderChoice {
    #[serde(rename = "local")]
    Local,
    #[value(name = "openai")]
    #[serde(rename = "openai")]
    OpenAI,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingModelChoice {
    #[serde(rename = "nomic")]
    Nomic,
    #[value(name = "bge-m3")]
    #[serde(rename = "bge-m3")]
    BgeM3,
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpenAIModelChoice {
    #[value(name = "text-embedding-3-small")]
    #[serde(rename = "text-embedding-3-small")]
    TextEmbedding3Small,
    #[value(name = "text-embedding-3-large")]
    #[serde(rename = "text-embedding-3-large")]
    TextEmbedding3Large,
}

pub(crate) enum Embedder {
    Local { inner: Box<TextEmbedding> },
    OpenAI(OpenAIClient),
}

pub(crate) struct OpenAIClient {
    http: reqwest::Client,
    api_key: String,
    organization: Option<String>,
    project: Option<String>,
    model: OpenAIModelChoice,
    dimensions: Option<i32>,
}

#[derive(Serialize)]
struct OpenAIEmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<i32>,
    encoding_format: &'a str,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

#[derive(Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

pub(crate) fn resolve_dim(
    provider: EmbeddingProviderChoice,
    local_model: EmbeddingModelChoice,
    openai_model: OpenAIModelChoice,
    dim: Option<i32>,
) -> Result<i32> {
    /*
    Enforce provider/model constraints on embedding dimensionality.
    Local models have fixed output sizes; OpenAI allows smaller dims but never larger than the model cap.
    This protects the index from mixed vector sizes that would make similarity search invalid.
    */
    match provider {
        EmbeddingProviderChoice::Local => {
            let model_dim = model_dim(local_model);
            match dim {
                Some(value) => {
                    if value <= 0 {
                        anyhow::bail!("dim must be > 0");
                    }
                    if value != model_dim {
                        anyhow::bail!(
                            "dim must match model output ({model_dim}) for {:?}",
                            local_model
                        );
                    }
                    Ok(value)
                }
                None => Ok(model_dim),
            }
        }
        EmbeddingProviderChoice::OpenAI => {
            let model_dim = openai_model_dim(openai_model);
            match dim {
                Some(value) => {
                    if value <= 0 {
                        anyhow::bail!("dim must be > 0");
                    }
                    if value > model_dim {
                        anyhow::bail!(
                            "dim must be <= model output ({model_dim}) for {:?}",
                            openai_model
                        );
                    }
                    Ok(value)
                }
                None => Ok(model_dim),
            }
        }
    }
}

pub(crate) fn build_embedder(
    provider: EmbeddingProviderChoice,
    local_model: EmbeddingModelChoice,
    openai_model: OpenAIModelChoice,
    show_download_progress: bool,
    dim: i32,
) -> Result<Embedder> {
    /*
    Construct the embedding backend with cache + network behavior aligned to the CLI.
    Local models are cached under the app data directory to keep the repo clean and reuse downloads.
    OpenAI uses environment credentials and only requests explicit dimensions when they differ from defaults.
    */
    match provider {
        EmbeddingProviderChoice::Local => {
            let fastembed_model = match local_model {
                EmbeddingModelChoice::Nomic => EmbeddingModel::NomicEmbedTextV15,
                EmbeddingModelChoice::BgeM3 => EmbeddingModel::BGEM3,
            };

            let mut options = InitOptions::new(fastembed_model);
            if show_download_progress {
                options = options.with_show_download_progress(true);
            }
            let cache_dir = app_data_dir()?.join("fastembed");
            ensure_dir(&cache_dir)?;
            options = options.with_cache_dir(cache_dir);

            Ok(Embedder::Local {
                inner: Box::new(TextEmbedding::try_new(options)?),
            })
        }
        EmbeddingProviderChoice::OpenAI => Ok(Embedder::OpenAI(OpenAIClient::new(
            openai_model, dim,
        )?)),
    }
}

pub(crate) fn prepare_query_input(
    provider: EmbeddingProviderChoice,
    local_model: EmbeddingModelChoice,
    query: &str,
) -> String {
    match provider {
        EmbeddingProviderChoice::Local => match local_model {
            // Nomic models are trained on prefixed inputs for query/document separation.
            EmbeddingModelChoice::Nomic => format!("search_query: {query}"),
            EmbeddingModelChoice::BgeM3 => query.to_string(),
        },
        EmbeddingProviderChoice::OpenAI => query.to_string(),
    }
}

pub(crate) fn prepare_document_inputs(
    provider: EmbeddingProviderChoice,
    local_model: EmbeddingModelChoice,
    chunks: &[String],
) -> Vec<String> {
    match provider {
        EmbeddingProviderChoice::Local => match local_model {
            // Nomic models are trained on prefixed inputs for query/document separation.
            EmbeddingModelChoice::Nomic => chunks
                .iter()
                .map(|chunk| format!("search_document: {chunk}"))
                .collect(),
            EmbeddingModelChoice::BgeM3 => chunks.to_vec(),
        },
        EmbeddingProviderChoice::OpenAI => chunks.to_vec(),
    }
}

pub(crate) async fn embed_texts(
    embedder: &mut Embedder,
    inputs: Vec<String>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    /*
    Single entry point for embedding across providers.
    The batch size is honored for local models and used to chunk OpenAI calls to avoid request limits.
    */
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    match embedder {
        Embedder::Local { inner } => Ok(inner.embed(inputs.as_slice(), Some(batch_size))?),
        Embedder::OpenAI(client) => client.embed(&inputs, batch_size).await,
    }
}

fn model_dim(model: EmbeddingModelChoice) -> i32 {
    match model {
        EmbeddingModelChoice::Nomic => 768,
        EmbeddingModelChoice::BgeM3 => 1024,
    }
}

fn openai_model_dim(model: OpenAIModelChoice) -> i32 {
    match model {
        OpenAIModelChoice::TextEmbedding3Small => 1536,
        OpenAIModelChoice::TextEmbedding3Large => 3072,
    }
}

fn openai_model_id(model: OpenAIModelChoice) -> &'static str {
    match model {
        OpenAIModelChoice::TextEmbedding3Small => "text-embedding-3-small",
        OpenAIModelChoice::TextEmbedding3Large => "text-embedding-3-large",
    }
}

impl OpenAIClient {
    fn new(model: OpenAIModelChoice, dim: i32) -> Result<Self> {
        let api_key = env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is not set")?;
        let organization = env::var("OPENAI_ORG_ID").ok();
        let project = env::var("OPENAI_PROJECT_ID").ok();
        let default_dim = openai_model_dim(model);
        let dimensions = if dim == default_dim { None } else { Some(dim) };

        Ok(Self {
            http: reqwest::Client::new(),
            api_key,
            organization,
            project,
            model,
            dimensions,
        })
    }

    async fn embed(&self, inputs: &[String], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        /*
        OpenAI can return embeddings out of order and limits inputs per request.
        We chunk requests, then sort by index to restore input order for downstream consumers.
        */
        let mut output = Vec::with_capacity(inputs.len());
        let chunk_size = batch_size.max(1);

        for chunk in inputs.chunks(chunk_size) {
            let request = OpenAIEmbeddingRequest {
                model: openai_model_id(self.model),
                input: chunk,
                dimensions: self.dimensions,
                encoding_format: "float",
            };

            let mut builder = self
                .http
                .post("https://api.openai.com/v1/embeddings")
                .bearer_auth(&self.api_key)
                .json(&request);

            if let Some(org) = &self.organization {
                builder = builder.header("OpenAI-Organization", org);
            }
            if let Some(project) = &self.project {
                builder = builder.header("OpenAI-Project", project);
            }

            let response = builder.send().await?;
            let status = response.status();
            let body = response.text().await?;
            if !status.is_success() {
                anyhow::bail!("OpenAI embeddings failed ({status}): {body}");
            }

            let mut parsed: OpenAIEmbeddingResponse = serde_json::from_str(&body)?;
            // OpenAI can return embeddings out of order; sort by index to preserve input order.
            parsed.data.sort_by_key(|item| item.index);
            for item in parsed.data {
                output.push(item.embedding);
            }
        }

        Ok(output)
    }
}
