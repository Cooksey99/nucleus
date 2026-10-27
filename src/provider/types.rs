//! Common types for LLM providers.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::models::EmbeddingModel;

/// Errors that can occur when interacting with a provider.
#[derive(Debug, Error)]
pub enum ProviderError {
    /// The HTTP request itself failed.
    #[cfg(feature = "hosted")]
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    /// A response body could not be parsed.
    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    /// The provider returned an API-level error.
    #[error("API error: {0}")]
    Api(String),

    /// A parameter was invalid for this provider.
    #[error("Invalid parameter: {0}")]
    InvalidParam(String),

    /// Any other provider failure.
    #[error("Provider error: {0}")]
    Other(String),
}

/// Result type for provider operations.
pub type Result<T> = std::result::Result<T, ProviderError>;

/// Provider trait for LLM backends.
///
/// Implementations provide chat completions and embeddings through
/// different backends.
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
    /// Model identifier.
    pub model: String,
    /// Conversation messages to send.
    pub messages: Vec<Message>,
    /// Sampling temperature.
    pub temperature: f64,
    /// Tools the model may call.
    pub tools: Option<Vec<Tool>>,
    /// Requested structured output.
    pub structured_output: Option<StructuredOutput>,
    /// Reasoning effort, where the provider supports it.
    pub reasoning_effort: Option<String>,
}

impl ChatRequest {
    /// A request for `model` carrying `messages`.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: 0.7,
            tools: None,
            structured_output: None,
            reasoning_effort: None,
        }
    }

    /// Ask for a structured output.
    pub fn with_structured_output(mut self, structured_output: StructuredOutput) -> Self {
        self.structured_output = Some(structured_output);
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the tools the model may call.
    pub fn with_tools(mut self, tools: Vec<Tool>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the reasoning effort.
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }
}

/// Response from chat completion (streaming chunk).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// Model that produced the chunk.
    pub model: String,
    /// Text content of the chunk.
    pub content: String,
    /// Whether this is the final chunk of the stream.
    pub done: bool,
    /// The message this chunk belongs to.
    pub message: Message,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Role name, e.g. `"user"`, `"assistant"`, `"system"`, `"tool"`.
    pub role: String,
    /// Context, generally pulled from RAG
    pub context: Option<String>,
    /// Message input from the user
    pub content: String,

    /// Base64 image payloads attached to the message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,

    /// Tool calls the assistant requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// A system message.
    pub fn system(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            context,
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }

    /// A user message.
    pub fn user(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            context,
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }

    /// An assistant message.
    pub fn assistant(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            context,
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }

    /// A tool-result message.
    pub fn tool(context: Option<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            context,
            content: content.into(),
            images: None,
            tool_calls: None,
        }
    }
}

/// Tool specification for function calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition.
    pub function: ToolFunction,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    /// The model-facing function name.
    pub name: String,
    /// What the function does.
    pub description: String,
    /// JSON Schema for the function's arguments.
    pub parameters: serde_json::Value,
}

/// Tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// The function being called and its arguments.
    pub function: ToolCallFunction,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    /// The function name.
    pub name: String,
    /// The function's arguments as JSON.
    pub arguments: serde_json::Value,
}

/// Requested structured output for a hosted call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredOutput {
    /// JSON schema to define the expected output structure
    pub schema: serde_json::Value,
    /// Optional description of what the structured output represents
    pub description: Option<String>,
    /// Optional example of the expected output format
    ///
    /// Adding this could improve the results
    pub example: Option<serde_json::Value>,
}

impl StructuredOutput {
    /// A structured output driven by `schema`.
    pub fn new(schema: serde_json::Value) -> Self {
        Self {
            schema,
            description: None,
            example: None,
        }
    }

    /// Attach a description of the output.
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Attach an example of the expected output.
    pub fn with_example(mut self, example: serde_json::Value) -> Self {
        self.example = Some(example);
        self
    }
}

impl From<serde_json::Value> for StructuredOutput {
    fn from(schema: serde_json::Value) -> Self {
        Self::new(schema)
    }
}

impl From<StructuredOutput> for serde_json::Value {
    fn from(output: StructuredOutput) -> Self {
        serde_json::to_value(output).unwrap_or_default()
    }
}
