//! LLM provider abstraction layer.
//!
//! This module defines a common interface for different LLM backends
//! (Ollama, mistral.rs, etc.) to provide chat completions and embeddings.
use async_trait::async_trait;

use crate::ollama::OllamaError;

pub type ProviderError = OllamaError;
pub type ProviderResult<T> = Result<T, ProviderError>;

#[async_trait]
pub trait Provider: Send + Sync {

    async fn chat(
        &self,
        request: ChatRequest,
    );
}

/// Common request/response types for providers
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f32,
    pub tools: Option<Vec<Tool>>,
}

pub struct ChatResponse {
    /// Accumulated or chunk content
    pub content: String,
    /// If this is the final chunk
    pub done: bool,
    pub tool_calls: Option<Vec<ToolCall>>,
}

pub struct Message {
    // e.g. "system", "user", "assistant", "tool"
    pub role: String,
    pub content: String,
    pub tool_calls: Option<Vec<ToolCall>>,
}

pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,  // JSON schema
}

pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

