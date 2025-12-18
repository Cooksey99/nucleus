//! LLM provider abstraction layer.
//!
//! This module defines a common interface for different LLM backends
//! (Ollama, mistral.rs, etc.) to provide chat completions and embeddings.

pub mod mistralrs;
pub mod ollama;
mod types;
mod utils;

#[cfg(feature = "coreml")]
pub mod coreml;

// Re-export common types
pub use types::{
    ChatRequest, ChatResponse, EmbedRequest, EmbedResponse, Message, Provider, ProviderError,
    Result, Tool, ToolCall, ToolCallFunction, ToolFunction,
};

// Re-export provider implementations
pub use mistralrs::MistralRsProvider;
pub use ollama::OllamaProvider;

#[cfg(feature = "coreml")]
pub use coreml::CoreMLProvider;
