//! Common types for LLM providers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::models::EmbeddingModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderType {
    Ollama,
    MistralRs,
    #[cfg(any(target_os = "macos", feature = "coreml"))]
    CoreML,
}

impl ProviderType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderType::Ollama => "ollama",
            ProviderType::MistralRs => "mistralrs",
            #[cfg(any(target_os = "macos", feature = "coreml"))]
            ProviderType::CoreML => "coreml",
        }
    }
}

/// Errors that can occur when interacting with a provider.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),
    
    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("API error: {0}")]
    Api(String),
    
    #[error("Provider error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ProviderError>;

/// Provider trait for LLM backends.
///
/// Implementations provide chat completions and embeddings through
/// different backends (Ollama, mistral.rs, etc.).
#[async_trait]
pub trait Provider: Send + Sync {
    /// Stream a chat completion.
    ///
    /// The callback is invoked for each chunk of the response.
    async fn chat<'a>(
        &'a self,
        request: ChatRequest,
        callback: Box<dyn FnMut(ChatResponse) + Send + 'a>,
    ) -> Result<()>;
    
    /// Generate an embedding vector for the given text.
    async fn embed(&self, text: &str, model: &EmbeddingModel) -> Result<Vec<f32>>;
    
    /// Generate embeddings for multiple texts in batch.
    /// Default implementation calls embed() sequentially.
    async fn embed_batch(&self, texts: &[&str], model: &EmbeddingModel) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text, model).await?);
        }
        Ok(embeddings)
    }
}

/// Request for chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f64,
    pub tools: Option<Vec<Tool>>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: 0.7,
            tools: None,
        }
    }
    
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }
    
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }
}

/// Response from chat completion (streaming chunk).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub model: String,
    pub content: String,
    pub done: bool,
    pub message: Message,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    /// Context, generally pulled from RAG
    pub context: Option<String>,
    /// Message input from the user
    pub content: String,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    pub fn system(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            context: Some(context).unwrap_or(None),
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }
    
    pub fn user(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            context: Some(context).unwrap_or(None),
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }
    
    pub fn assistant(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            context: Some(context).unwrap_or(None),
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }
    
    pub fn tool(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            context: Some(context).unwrap_or(None),
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }
}

/// Tool specification for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub function: ToolCallFunction,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Request for generating embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: String,
}

/// Response containing embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub model: String,
    
    #[serde(default)]
    pub embeddings: Vec<Vec<f32>>,
}
